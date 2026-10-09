"""Separate structured prediction experiment; local metrics are not official scores."""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '/mnt/storage/aigc/participant')
import run_inference as official

BASE = '/mnt/storage/models/Qwen2.5-3B-Instruct'
ROOT = Path('/mnt/storage/starter/structured_v2')
ADAPTER = Path('/mnt/storage/results/structured-v2')
PROMPT = '''你是对话情绪和用户画像标注器。读取完整历史，判断最后一条用户消息的情绪，并提取用户画像。只输出一个JSON对象，不输出解释、回复或Markdown。
情绪只选一个标签：{emotions}。
画像以用户发言为依据，不把助手的表达风格当作用户风格；结合历史，不凭单次情绪杜撰稳定性格。没有足够证据时允许空数组。只能使用下列标签：
personality_traits：{personality}
interests：{interests}
style：{styles}
输出格式：{{"emotion_label":"标签","user_profile":{{"personality_traits":[],"interests":[],"style":[]}}}}
'''.format(
    emotions=', '.join(sorted(official.EMOTIONS)),
    personality=', '.join(sorted(official.PROFILE_LABELS['personality_traits'])),
    interests=', '.join(sorted(official.PROFILE_LABELS['interests'])),
    styles=', '.join(sorted(official.PROFILE_LABELS['style'])),
)


def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(text):
    obj = json.loads(text.strip())
    if set(obj) != {'emotion_label', 'user_profile'}:
        raise ValueError('incorrect_keys')
    if obj['emotion_label'] not in official.EMOTIONS:
        raise ValueError('invalid_emotion')
    profile = obj['user_profile']
    if not isinstance(profile, dict) or set(profile) != set(official.PROFILE_LABELS):
        raise ValueError('invalid_profile_fields')
    for field, allowed in official.PROFILE_LABELS.items():
        values = profile[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or value not in allowed for value in values
        ) or len(values) != len(set(values)):
            raise ValueError('invalid_' + field)
    return obj


def tokenizer():
    result = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    result.pad_token = result.eos_token
    result.padding_side = 'right'
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    examples, emotions = [], []
    skipped = 0
    for row in rows('/mnt/storage/starter/prepared/train_sft.jsonl'):
        target = json.loads(row['messages'][-1]['content'])
        answer = {key: target[key] for key in ['emotion_label', 'user_profile']}
        text = json.dumps(answer, ensure_ascii=False)
        validate(text)
        messages = [{'role': 'system', 'content': PROMPT}, row['messages'][1]]
        prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(
            messages + [{'role': 'assistant', 'content': text}],
            tokenize=True, add_generation_prompt=False,
        )
        if len(full) > 3072:
            skipped += 1
            continue
        if full[:len(prefix)] != prefix:
            raise RuntimeError('Chat template prefix mismatch')
        examples.append({'input_ids': full, 'labels': [-100] * len(prefix) + full[len(prefix):]})
        emotions.append(answer['emotion_label'])
    counts = Counter(emotions)
    weights = [min(4.0, (len(examples) / len(counts) / counts[e]) ** 0.5) for e in emotions]
    torch.save({'examples': examples, 'weights': weights}, ROOT / 'train_tokens.pt')
    metadata = {
        'eligible_samples': len(examples), 'skipped_over_3072_tokens': skipped,
        'emotion_counts': dict(counts), 'sampling': 'inverse_sqrt_emotion_frequency_capped_at_4',
        'seed': 42, 'max_length': 3072, 'base_model': BASE,
        'system_prompt': PROMPT, 'target_fields': ['emotion_label', 'user_profile'],
        'validation_used_for_training': False,
    }
    (ROOT / 'data_report.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('PREPARED', json.dumps({k: v for k, v in metadata.items() if k != 'system_prompt'}), flush=True)


def train(steps):
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; refusing CPU training')
    if (ADAPTER / 'adapter_config.json').exists():
        raise RuntimeError('Completed adapter already exists; do not overwrite')
    set_seed(42)
    prepared = torch.load(ROOT / 'train_tokens.pt', weights_only=True)
    examples, weights = prepared['examples'], prepared['weights']
    tok = tokenizer()

    class Data(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            return examples[index]

    def collate(batch):
        length = max(len(item['input_ids']) for item in batch)
        return {
            'input_ids': torch.tensor([item['input_ids'] + [tok.pad_token_id] * (length - len(item['input_ids'])) for item in batch]),
            'attention_mask': torch.tensor([[1] * len(item['input_ids']) + [0] * (length - len(item['input_ids'])) for item in batch]),
            'labels': torch.tensor([item['labels'] + [-100] * (length - len(item['labels'])) for item in batch]),
        }

    class BalancedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return WeightedRandomSampler(weights, len(weights), replacement=True)

    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation='sdpa',
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type='CAUSAL_LM', r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules='all-linear', bias='none',
    ))
    model.print_trainable_parameters()
    ADAPTER.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(ADAPTER), max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=1e-4, warmup_steps=30,
        logging_steps=10, save_strategy='steps', save_steps=200, save_total_limit=3,
        bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, optim='adamw_torch',
        report_to='none', remove_unused_columns=False, dataloader_num_workers=0, seed=42,
    )
    trainer = BalancedTrainer(model=model, args=training, train_dataset=Data(), data_collator=collate)
    checkpoint = get_last_checkpoint(str(ADAPTER))
    (ADAPTER / 'experiment_config.json').write_text(json.dumps({
        'steps': steps, 'rank': 16, 'target_modules': 'all-linear',
        'learning_rate': 1e-4, 'gradient_accumulation_steps': 4,
        'data_report': str(ROOT / 'data_report.json'), 'resume_checkpoint': checkpoint,
    }, indent=2))
    print('RESUME', checkpoint, flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    trainer.save_state()
    print('TRAINING_FINISHED', flush=True)


def evaluate(adapter, input_file, answers_file, output):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tok = tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to('cuda')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    gold = {row['id']: row for row in rows(answers_file)}
    samples = rows(input_file)
    assert len(samples) == len(gold) and {r['id'] for r in samples} == set(gold)
    predictions, failures, latencies = [], [], []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for index, sample in enumerate(samples, 1):
            messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': official.conversation_text(sample)}]
            inp = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True).to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
            try:
                obj = validate(raw)
            except (ValueError, TypeError, KeyError) as error:
                failures.append({'id': sample['id'], 'error': str(error), 'raw': raw})
                obj = {'emotion_label': '', 'user_profile': {field: [] for field in official.PROFILE_LABELS}}
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prediction = {'id': sample['id'], **obj, 'raw': raw, 'seconds': elapsed}
            handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            handle.flush()
            predictions.append(prediction)
            latencies.append(elapsed)
            print(f'[{index}/{len(samples)}] {sample["id"]} {elapsed:.2f}s {obj["emotion_label"]}', flush=True)
    by_id = {row['id']: row for row in predictions}
    correct = sum(by_id[sid]['emotion_label'] == answer['emotion_label'] for sid, answer in gold.items())
    summary = {'samples': len(samples), 'emotion_accuracy': correct / len(samples), 'invalid_outputs': len(failures), 'profiles': {}, 'classification_latency_seconds': sum(latencies) / len(latencies), 'note': 'Local structured-prediction diagnostics only; response generation and official total score are not evaluated.', 'input_file': str(input_file), 'adapter': adapter}
    for field in official.PROFILE_LABELS:
        tp = fp = fn = exact = empty = nonempty_exact = nonempty_total = 0
        for sid, answer in gold.items():
            expected, actual = set(answer['user_profile'][field]), set(by_id[sid]['user_profile'][field])
            tp += len(expected & actual)
            fp += len(actual - expected)
            fn += len(expected - actual)
            exact += expected == actual
            empty += not actual
            if expected:
                nonempty_total += 1
                nonempty_exact += expected == actual
        summary['profiles'][field] = {'micro_precision': tp / max(1, tp + fp), 'micro_recall': tp / max(1, tp + fn), 'micro_f1': 2 * tp / max(1, 2 * tp + fp + fn), 'set_exact': exact / len(samples), 'predicted_empty': empty, 'nonempty_gold_exact': nonempty_exact, 'nonempty_gold_count': nonempty_total}
    (output / 'failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    (output / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('METRICS', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--adapter')
    parser.add_argument('--input', default='/mnt/storage/starter/prepared/smoke100_inference.jsonl')
    parser.add_argument('--answers', default='/mnt/storage/starter/prepared/smoke100_answers.jsonl')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'train':
        train(args.steps)
    else:
        if not args.output:
            parser.error('--output is required for evaluation')
        evaluate(args.adapter, args.input, args.answers, args.output)
