"""Generate replies without reading reference answers; score separately."""
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
sys.path.insert(0, '/root/participant_v3')
import official_helpers as official
base = '/root/participant_v3/assets/base'
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
original = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
adapted = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
adapted = PeftModel.from_pretrained(adapted, '/mnt/storage/results/reply-v3', local_files_only=True).merge_and_unload().to('cuda').eval()
prompt = official.SYSTEM_PROMPT.format(emotions=', '.join(sorted(official.EMOTIONS)), personality=', '.join(sorted(official.PROFILE_LABELS['personality_traits'])), interests=', '.join(sorted(official.PROFILE_LABELS['interests'])), styles=', '.join(sorted(official.PROFILE_LABELS['style'])))
new_prompt = json.loads(Path('/mnt/storage/starter/reply_v3/data_report.json').read_text())['system_prompt']
samples = official.read_jsonl(Path('/mnt/storage/starter/prepared/dev200_inference.jsonl'))
output = Path('/mnt/storage/results/reply-v3/dev200_replies.jsonl')
with output.open('w') as handle:
    for i, row in enumerate(samples, 1):
        result = {'id':row['id']}
        for name, model, system, limit in [('baseline', original, prompt, 1200), ('candidate', adapted, new_prompt, 512)]:
            messages = [{'role':'system','content':system}, {'role':'user','content':official.conversation_text(row)}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True).to('cuda')
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=limit, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            raw = tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            if name=='baseline':
                parsed, failed = official.parse_prediction(row['id'],raw)
                text = parsed['response_text']
            else: text, failed = raw.strip(), False
            result[name] = {'text':text,'raw':raw,'parse_failed':failed,'latency_ms':(time.perf_counter()-started)*1000,'generated_tokens':out.shape[1]-inputs['input_ids'].shape[1],'hit_token_limit':out.shape[1]-inputs['input_ids'].shape[1]>=limit}
        handle.write(json.dumps(result,ensure_ascii=False)+'\n'); handle.flush()
        print('REPLY_COMPARE',i,len(samples),flush=True)
print('REPLY_COMPARE_FINISHED',flush=True)
