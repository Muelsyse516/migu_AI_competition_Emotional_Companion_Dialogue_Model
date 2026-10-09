import argparse, hashlib, json, os, statistics, subprocess, sys, time, tempfile
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import official_helpers as official

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / 'configs/inference_config.json').read_text(encoding='utf-8'))
parser = argparse.ArgumentParser()
parser.add_argument('input'); parser.add_argument('output')
args = parser.parse_args()
source, output = Path(args.input), Path(args.output)
samples = official.read_jsonl(official.validate_test_file(source))
output.mkdir(parents=True, exist_ok=True)
if not torch.cuda.is_available():
    raise RuntimeError('GPU unavailable; CPU fallback disabled')
scratch_context = tempfile.TemporaryDirectory(prefix='competition-inference-')
scratch = Path(scratch_context.name)
subprocess.run([sys.executable, str(ROOT/'classical_worker.py'), str(source), str(scratch/'classical_predictions.jsonl'), str(scratch/'cpu_timing.json')], check=True)
cpu = {r['id']: r for r in official.read_jsonl_predictions(scratch/'classical_predictions.jsonl')} if hasattr(official, 'read_jsonl_predictions') else {r['id']: r for line in (scratch/'classical_predictions.jsonl').read_text().splitlines() for r in [json.loads(line)]}
cpu_ms = json.loads((scratch/'cpu_timing.json').read_text())['inference_ms'] / len(samples)
base = str(ROOT / config['base_model'])
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
def load(adapter=None, model_base=None):
    model = AutoModelForCausalLM.from_pretrained(model_base or base, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    if adapter:
        model = PeftModel.from_pretrained(model, str(ROOT/adapter), local_files_only=True).merge_and_unload()
    model.config.use_cache = True
    return model.to('cuda').eval()
structured_base = str(ROOT/config.get('structured_base_model',config['base_model']))
structured_tokenizer = AutoTokenizer.from_pretrained(structured_base,local_files_only=True)
def normalize_head_input(module,args):
    x=args[0].float()
    return ((x*torch.rsqrt(x.square().mean(-1,keepdim=True).clamp_min(1e-8))).to(module.weight.dtype),)
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['score'])
structured=AutoModelForSequenceClassification.from_pretrained(structured_base,num_labels=16,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',quantization_config=quant,device_map={'':0})
structured.config.pad_token_id=structured_tokenizer.eos_token_id
structured.score.register_forward_pre_hook(normalize_head_input)
structured=prepare_model_for_kbit_training(structured,use_gradient_checkpointing=False)
structured=PeftModel.from_pretrained(structured,str(ROOT/config['classifier_adapter']),local_files_only=True).eval()
structured.config.use_cache=False
reply = load(config.get('reply_adapter'))
from profile_runtime import ProfileEncoder, EmotionEncoder
profile_encoder = ProfileEncoder(ROOT/config['profile_encoder']) if config.get('profile_encoder') else None
emotion_encoder = EmotionEncoder(ROOT/config['emotion_encoder']) if config.get('emotion_encoder') else None
emotion_overrides = []
reply_prompt = config.get('reply_prompt') or official.SYSTEM_PROMPT.format(emotions=', '.join(sorted(official.EMOTIONS)), personality=', '.join(sorted(official.PROFILE_LABELS['personality_traits'])), interests=', '.join(sorted(official.PROFILE_LABELS['interests'])), styles=', '.join(sorted(official.PROFILE_LABELS['style'])))
def generate(model, prompt, row, limit, repetition_penalty=1.0, generation_tokenizer=None):
    tok = generation_tokenizer or tokenizer
    messages = [{'role':'system','content':prompt}, {'role':'user','content':official.conversation_text(row)}]
    inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True).to('cuda')
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=limit, do_sample=False, repetition_penalty=repetition_penalty, pad_token_id=tok.eos_token_id)
    return tok.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
latencies, failures = [], []
baseline_components, extra_encoder_ms = [], []
with (output/'submission.jsonl').open('w', encoding='utf-8') as handle, (scratch/'raw_outputs.jsonl').open('w', encoding='utf-8') as raw_handle:
    for index, sample in enumerate(samples, 1):
        torch.cuda.synchronize(); started = time.perf_counter()
        history = '\n'.join(t['role']+': '+t['content'] for t in sample['history'])
        ids = structured_tokenizer.apply_chat_template([{'role':'system','content':config['classifier_prompt']},{'role':'user','content':history}],tokenize=True,add_generation_prompt=True)
        if len(ids)>2048:ids=ids[:256]+ids[-1792:]
        inp=torch.tensor([ids],device='cuda')
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):logits=structured(input_ids=inp,attention_mask=torch.ones_like(inp)).logits
        emotion=sorted(official.EMOTIONS)[int(logits.argmax(-1).item())]
        labels={'emotion_label':emotion,'user_profile':cpu[sample['id']]['user_profile']}
        raw_labels=json.dumps(labels,ensure_ascii=False)
        baseline_components.append({'id':sample['id'],'emotion_label':labels['emotion_label'],'user_profile':json.loads(json.dumps(labels['user_profile']))})
        additional_ms = 0.0
        if emotion_encoder is not None:
            torch.cuda.synchronize(); encoder_started = time.perf_counter()
            encoded_emotion = emotion_encoder.predict(sample)
            torch.cuda.synchronize(); additional_ms += (time.perf_counter()-encoder_started)*1000
            if encoded_emotion['confidence'] >= config['emotion_override_threshold'] and encoded_emotion['emotion_label'] != labels['emotion_label']:
                emotion_overrides.append({'id':sample['id'],'original':labels['emotion_label'],'replacement':encoded_emotion['emotion_label'],'score':encoded_emotion['confidence']})
                labels['emotion_label'] = encoded_emotion['emotion_label']
        raw_reply = generate(reply, reply_prompt, sample, config.get('reply_max_tokens',1200),config.get('reply_repetition_penalty',1.0))
        if config.get('reply_adapter'):
            text, failed = raw_reply.strip(), False
        else:
            parsed, failed = official.parse_prediction(sample['id'], raw_reply)
            text = parsed['response_text']
        if not isinstance(text,str) or not text.strip():
            text = '我在听，你可以继续说说让你最在意的部分。'
            failures.append({'id':sample['id'], 'stage':'empty_reply'})
        if failed: failures.append({'id':sample['id'], 'stage':'reply_parse'})
        c = cpu[sample['id']]['user_profile']
        result = {'id':sample['id'], 'response_text':text, 'emotion_label':labels['emotion_label'], 'user_profile':{'personality_traits':c['personality_traits'], 'interests':labels['user_profile']['interests'], 'style':c['style']}, 'memory_refs':[]}
        for field in config.get('generated_profile_fields',[]):
            result['user_profile'][field] = labels['user_profile'][field]
        if profile_encoder is not None:
            torch.cuda.synchronize(); encoder_started = time.perf_counter()
            encoded_profile = profile_encoder.predict(sample)
            torch.cuda.synchronize(); additional_ms += (time.perf_counter()-encoder_started)*1000
            for field in config['profile_encoder_fields']:
                result['user_profile'][field] = encoded_profile[field]
        handle.write(json.dumps(result,ensure_ascii=False)+'\n'); handle.flush()
        raw_handle.write(json.dumps({'id':sample['id'],'labels':raw_labels,'reply':raw_reply},ensure_ascii=False)+'\n'); raw_handle.flush()
        torch.cuda.synchronize(); elapsed = (time.perf_counter()-started)*1000+cpu[sample['id']]['cpu_inference_ms']
        if index<=100:
            latencies.append(elapsed)
            extra_encoder_ms.append(additional_ms)
        print(f'[{index}/{len(samples)}] {sample["id"]} {elapsed:.1f}ms', flush=True)
report = {'backend':'transformers', 'version':config['version'], 'timing_scope':'emotion classification + reply generation + profile encoder + emotion encoder + parsing; individual CPU history/vectorization/prediction time added; CPU executed before GPU phase; model loading excluded', 'complete':len(latencies)==100, 'required_rounds':100, 'rounds':len(latencies), 'samples':len(samples), 'average_latency_ms':statistics.fmean(latencies), 'median_latency_ms':statistics.median(latencies), 'p95_latency_ms':official.percentile95(latencies), 'min_latency_ms':min(latencies), 'max_latency_ms':max(latencies), 'latencies_ms':latencies, 'hardware_label':torch.cuda.get_device_name(0), 'parse_failure_count':len({r['id'] for r in failures}), 'parse_failure_ids':sorted({r['id'] for r in failures}), 'fallback_events':failures, 'baseline_components':baseline_components, 'paired_comparison_scope':'Candidate only; compare with separately measured unchanged baseline. Auxiliary labels are not old-structured predictions.', 'extra_encoder_mean_ms':statistics.fmean(extra_encoder_ms), 'extra_encoder_ms':extra_encoder_ms, 'emotion_override_count':len(emotion_overrides), 'emotion_overrides':emotion_overrides, 'test_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'submission_sha256':hashlib.sha256((output/'submission.jsonl').read_bytes()).hexdigest()}
(output/'performance_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('INFERENCE_FINISHED',flush=True)

scratch_context.cleanup()
