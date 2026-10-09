"""Emotion classification experiment, separate from validated submission."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from peft import PeftConfig, PeftModel, get_peft_model

BASE = '/mnt/storage/models/Qwen2.5-7B-Instruct'
WARM = '/mnt/storage/results/emotion-qwen-7b-v2'
ROOT = Path('/mnt/storage/starter/emotion_qwen_7b')
OUT = Path('/mnt/storage/results/emotion-qwen-7b-v3')
DATA = Path('/mnt/storage/starter/prepared')
LABELS = ['anger', 'anxiety', 'care', 'disgust', 'fear', 'gratitude', 'helplessness', 'joy', 'loneliness', 'mixed', 'neutral', 'pride', 'relaxed', 'sadness', 'shame', 'surprise']
PROMPT = '结合对话历史，识别最后一条用户发言的主要情绪。不要把助手的语气当作用户情绪。候选情绪：' + ', '.join(LABELS)

def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok

def encode(tok, row):
    text = '\n'.join(t['role'] + ': ' + t['content'] for t in row['history'])
    ids = tok.apply_chat_template([{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
    trimmed = len(ids) > 2048
    if trimmed:
        ids = ids[:256] + ids[-1792:]
    return ids, trimmed

def prepare():
    ROOT.mkdir(exist_ok=True, parents=True)
    samples = read(DATA/'train_inference.jsonl')
    answers = {r['id']: r for r in read(DATA/'train_answers.jsonl')}
    convs = sorted({r['conversation_id'] for r in samples})
    random.Random(42).shuffle(convs)
    cal = set(convs[:len(convs)//5])
    tok = tokenizer()
    records = []
    trimmed = 0
    for row in samples:
        ids, changed = encode(tok, row)
        trimmed += changed
        records.append({'input_ids': ids, 'labels': LABELS.index(answers[row['id']]['emotion_label']), 'calibration': row['conversation_id'] in cal})
    torch.save(records, ROOT/'tokens.pt')
    report = {'samples': len(records), 'training_calibration_conversations': len(cal), 'calibration_samples': sum(r['calibration'] for r in records), 'trimmed_samples': trimmed, 'max_tokens': 2048, 'retained_when_long': 'first256 + last1792', 'labels': LABELS, 'prompt': PROMPT, 'seed': 42, 'validation_used_for_training': False, 'objective': 'unweighted single-label cross entropy', 'warm_start_adapter': WARM}
    for name in ['train_inference.jsonl', 'train_answers.jsonl']:
        report[name+'_sha256'] = hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    (ROOT/'data_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('PREPARED', {k:v for k,v in report.items() if k not in ['prompt', 'labels']}, flush=True)

def normalize_head_input(module, args):
    x = args[0].float()
    return ((x * torch.rsqrt(x.square().mean(-1, keepdim=True).clamp_min(1e-8))).to(module.weight.dtype),)

def base_model():
    from transformers import BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['score'])
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(LABELS), local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', quantization_config=quant, device_map={'':0})
    model.score.register_forward_pre_hook(normalize_head_input)
    model.config.pad_token_id = tokenizer().pad_token_id
    model.config.problem_type = 'single_label_classification'
    model.config.use_cache = False
    return model

def warm_classifier():
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(base_model(), gradient_checkpointing_kwargs={'use_reentrant':False})
    return PeftModel.from_pretrained(model, WARM, local_files_only=True, is_trainable=True)

def collate(batch):
    pad = tokenizer().pad_token_id if not hasattr(collate, 'pad') else collate.pad
    collate.pad = pad
    length = max(len(x['input_ids']) for x in batch)
    return {'input_ids': torch.tensor([r['input_ids']+[pad]*(length-len(r['input_ids'])) for r in batch]), 'attention_mask': torch.tensor([[1]*len(r['input_ids'])+[0]*(length-len(r['input_ids'])) for r in batch]), 'labels': torch.tensor([r['labels'] for r in batch])}

class Data(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

def train(steps):
    assert torch.cuda.is_available(), 'Refusing CPU training'
    assert not (OUT/'adapter_config.json').exists(), 'Do not overwrite completed adapter'
    set_seed(42)
    records = torch.load(ROOT/'tokens.pt', weights_only=True)
    model = warm_classifier()
    model.config.use_cache = False
    head, adapters = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (head if '.score.' in name else adapters).append(param)
    assert head and adapters
    assert sum(p.numel() for p in head) < 1000000
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([{'params': adapters, 'lr': 1e-5}, {'params': head, 'lr': 5e-5}], weight_decay=.01)
    OUT.mkdir(exist_ok=True, parents=True)
    args = TrainingArguments(output_dir=str(OUT), max_steps=steps, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-5, warmup_steps=40, save_strategy='steps', save_steps=400, save_total_limit=4, logging_steps=20, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, report_to='none', remove_unused_columns=False, dataloader_num_workers=0, seed=42)
    from transformers.trainer_utils import get_last_checkpoint
    trainer = Trainer(model=model, args=args, train_dataset=Data([r for r in records if not r['calibration']]), data_collator=collate, optimizers=(optimizer, None))
    (OUT/'experiment_config.json').write_text(json.dumps({'steps': steps, 'base': BASE, 'warm_start_adapter': WARM, 'task': 'SEQ_CLS', 'classifier_head_trainable': True, 'head_lr': 5e-5, 'adapter_lr': 1e-5, 'loss': 'unweighted_cross_entropy', 'development_labels_used_for_training': False}, indent=2))
    checkpoint = get_last_checkpoint(str(OUT))
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(OUT)); trainer.save_state(); tokenizer().save_pretrained(OUT)
    print('CLASSIFIER_TRAINING_FINISHED', flush=True)

def predict(model, records):
    predicted, confidence = [], []
    for start in range(0, len(records), 2):
        batch = collate(records[start:start+2])
        with torch.inference_mode():
            logits = model(**{k:v.to('cuda') for k,v in batch.items() if k != 'labels'}).logits
        probs = logits.float().softmax(-1)
        predicted.extend(probs.argmax(-1).cpu().tolist())
        confidence.extend(probs.max(-1).values.cpu().tolist())
    return predicted, confidence

def evaluate():
    samples_dev = read(DATA/'dev200_inference.jsonl')
    gold_dev = {r['id']: r for r in read(DATA/'dev200_answers.jsonl')}
    tok_dev = tokenizer()
    cal = [{'input_ids': encode(tok_dev, r)[0], 'labels': LABELS.index(gold_dev[r['id']]['emotion_label'])} for r in samples_dev]
    reports = []
    for name in ['checkpoint-'+str(i) for i in range(400,2601,400)] + ['final']:
        path = OUT/name if name != 'final' else OUT
        if not (path/'adapter_config.json').exists(): continue
        model = PeftModel.from_pretrained(base_model(), path, local_files_only=True).eval()
        model.config.use_cache = False
        pred, _ = predict(model, cal)
        acc = sum(p==r['labels'] for p,r in zip(pred,cal))/len(cal)
        reports.append({'checkpoint': str(path), 'development_accuracy': acc, 'selection_scope': 'fixed dev200 development accuracy'})
        del model; gc.collect(); torch.cuda.empty_cache()
        print('CHECKPOINT_DEV', name, acc, flush=True)
    selection = max(reports, key=lambda r:r['development_accuracy'])
    (OUT/'selection.json').write_text(json.dumps({'selected': selection, 'all_checkpoints': reports, 'selection_data': 'fixed dev200 development, not independent test', 'warning': 'Continued from v2 on identical training-only subset; development checkpoint selection is not independent test'}, indent=2))
    model = PeftModel.from_pretrained(base_model(), selection['checkpoint'], local_files_only=True).eval()
    model.config.use_cache = False
    samples = read(DATA/'dev200_inference.jsonl'); tok = tokenizer()
    encoded = [{'input_ids': encode(tok,r)[0], 'labels': 0} for r in samples]
    pred, confidence = predict(model, encoded)
    rows = [{'id': r['id'], 'emotion_label': LABELS[p], 'confidence': c} for r,p,c in zip(samples,pred,confidence)]
    (OUT/'dev200_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    gold = {r['id']:r for r in read(DATA/'dev200_answers.jsonl')}
    acc = sum(r['emotion_label']==gold[r['id']]['emotion_label'] for r in rows)/len(rows)
    (OUT/'dev200_metrics.json').write_text(json.dumps({'emotion_accuracy': acc, 'samples': len(rows), 'selection': selection, 'scope': 'Development comparison, not untouched test'}, indent=2))
    print('CLASSIFIER_DEV_ACCURACY', acc, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare','train','evaluate','run']); p.add_argument('--steps', type=int, default=1200); a=p.parse_args()
    if a.mode == 'run':
        torch.cuda.init()
        torch.cuda.get_device_properties(0)
        if not (ROOT/'tokens.pt').exists(): prepare()
        train(a.steps)
        gc.collect(); torch.cuda.empty_cache()
        evaluate()
    elif a.mode == 'train': train(a.steps)
    else: globals()[a.mode]()
