"""Compatibility entry for the official starter: dispatch the verified V5 pipeline."""
import os,sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
if sys.argv[1:]==['--check-contract']:
    config=json.loads((root/'configs/inference_config.json').read_text())
    assert config['backend']=='transformers'
    for key in ['model_path','base_model','structured_base_model','classifier_adapter','reply_adapter','profile_encoder','classical_model']:
        assert (root/config[key]).exists(),key
    print(json.dumps({'config':str(root/'configs/inference_config.json'),'entry':str(root/'launch_verified.sh'),'backend':config['backend'],'status':'contract-files-present'}))
    raise SystemExit(0)
if len(sys.argv)!=3:raise SystemExit('Usage: run_inference.py TEST_FILE RESULT_DIR')
os.execv('/bin/bash',['/bin/bash',str(root/'launch_verified.sh'),*sys.argv[1:]])
