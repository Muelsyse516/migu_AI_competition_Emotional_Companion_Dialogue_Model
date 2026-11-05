FROM migu-next-runtime:20261025
COPY python /workspace/work/python
COPY venv /workspace/work/.venv
COPY participant /root/participant
ENV PARTICIPANT_PYTHON=/workspace/work/.venv/bin/python
ENV PATH=/workspace/work/.venv/bin:$PATH
ENV PYTHONPATH=/root/participant/vendor:/root/participant/runtime_vendor
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1 OMP_NUM_THREADS=8
WORKDIR /root/participant
ENTRYPOINT ["bash", "/root/participant/start.sh"]
