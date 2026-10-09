"""Diagnose the exported Linux runtime without installing or changing models."""
import importlib, json, os, platform, sys, traceback
from pathlib import Path

root = Path(__file__).resolve().parent
errors = []
def emit(stage, **details):
    print(json.dumps({'stage':stage, **details}, ensure_ascii=False), flush=True)

emit('runtime', python=sys.executable, version=sys.version, platform=platform.platform(), cwd=os.getcwd())
if sys.version_info[:2] != (3,12): errors.append('Bundled CPU binaries require Python 3.12')
config = json.loads((root/'configs/inference_config.json').read_text(encoding='utf-8'))
for field in ['base_model','structured_base_model','classifier_adapter','reply_adapter','profile_encoder','classical_model']:
    value = config.get(field)
    if value and not (root/value).exists(): errors.append(f'Missing {field}: {root/value}')
for name in ['torch','transformers','accelerate','peft','bitsandbytes','huggingface_hub']:
    try:
        module=importlib.import_module(name)
        emit('dependency', name=name, version=getattr(module,'__version__',None), file=getattr(module,'__file__',None))
        expected={'transformers':'4.57.6','accelerate':'1.15.0','peft':'0.17.1','bitsandbytes':'0.48.2','huggingface_hub':'0.36.2'}
        if name in expected and getattr(module,'__version__',None)!=expected[name]:
            errors.append(f'Unexpected {name} version; expected {expected[name]}')
    except Exception:
        errors.append('Import failed: '+name)
        traceback.print_exc()
try:
    import torch
    available=torch.cuda.is_available()
    emit('cuda',available=available,torch_cuda=torch.version.cuda,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not available: errors.append('No accessible CUDA GPU')
    else:
        emit('gpu',name=torch.cuda.get_device_name(0),memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        torch.ones(1,device='cuda').add_(1)
        torch.cuda.synchronize()
except Exception:
    errors.append('CUDA computation failed')
    traceback.print_exc()
try:
    from transformers import AutoTokenizer,AutoConfig
    for field in ['base_model','structured_base_model','profile_encoder']:
        path=root/config[field]
        model_config=AutoConfig.from_pretrained(path,local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        emit('model_metadata',field=field,model_type=model_config.model_type,tokens=len(tokenizer))
except Exception:
    errors.append('Model metadata/tokenizer failed')
    traceback.print_exc()
emit('preflight_complete',errors=errors)
if errors: raise SystemExit(2)
