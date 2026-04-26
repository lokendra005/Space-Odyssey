# ─────────────────────────────────────────────────────────────────────────────
# Odyssey Station — GPU-ready HF Space Dockerfile (v5.1)
#
# Base image already has torch 2.4.0 + CUDA 12.1, so we ONLY add the LLM stack
# on top. No silent fallback: if Unsloth/bnb don't install we want to see it.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
ENV APP_VERSION="v5.3-no-build-import"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Demo-runtime deps (small, fast layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LLM stack. We install unsloth ALONGSIDE torch/torchvision/torchaudio so pip
# resolves a self-consistent set in a single transaction (last build failed
# because pip upgraded torch to 2.10 but left torchvision at 0.19 — Unsloth
# refused to import on that mismatched pair). Other deps come along.
RUN pip install --no-cache-dir --upgrade \
        torch torchvision torchaudio \
        unsloth \
        bitsandbytes \
        xformers

# Build-time sanity check — DOES NOT import unsloth here because HF Spaces'
# build infrastructure has no GPU; unsloth_zoo raises:
#   "Unsloth cannot find any torch accelerator? You need a GPU."
# The runtime GPU container imports unsloth normally.
RUN python -c "import torch, bitsandbytes, peft, transformers; \
    print(f'[BUILD-OK] torch={torch.__version__} cuda={torch.version.cuda} ' \
          f'bnb={bitsandbytes.__version__} peft={peft.__version__} ' \
          f'transformers={transformers.__version__}')"

COPY . .

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "demo/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
