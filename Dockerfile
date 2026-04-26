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
ENV APP_VERSION="v5.1-gpu"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Demo-runtime deps (small, fast layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LLM stack — no `|| echo` fallback so any failure is visible in build logs.
# Installed in dependency order: bnb → transformers/peft/accelerate/trl → unsloth.
RUN pip install --no-cache-dir \
        "bitsandbytes>=0.43.1" \
        "transformers>=4.45.0,<4.50.0" \
        "peft>=0.13.0,<0.15.0" \
        "accelerate>=0.31.0" \
        "datasets>=2.20.0" \
        "trl>=0.10.0,<0.15.0" \
        "xformers"

# Unsloth — try PyPI wheel first, then fall back to git HEAD if PyPI lags.
RUN pip install --no-cache-dir unsloth \
    || pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git@main"

# Build-time sanity check; prints version line into the build log.
RUN python -c "import torch, unsloth, bitsandbytes, peft, transformers; \
    print(f'[BUILD-OK] torch={torch.__version__} cuda={torch.version.cuda} ' \
          f'unsloth={getattr(unsloth, \"__version__\", \"unknown\")} ' \
          f'bnb={bitsandbytes.__version__} peft={peft.__version__} ' \
          f'transformers={transformers.__version__}')"

COPY . .

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "demo/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
