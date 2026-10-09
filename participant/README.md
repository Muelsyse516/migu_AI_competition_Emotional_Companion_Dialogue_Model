V5 verified pipeline, official layout compatibility repair 2026-10-08.
Canonical configuration: configs/inference_config.json.
Entry: bash /root/participant/start.sh TEST_FILE RESULT_DIR.
Compatibility Python entry: run_inference.py dispatches launch_verified.sh.
All model assets use paths relative to participant; no network downloads.
Official execution failure fix requires platform retest; local static checks alone do not confirm official success.
