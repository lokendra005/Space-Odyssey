# ─────────────────────────────────────────────────────────────────────────────
# Odyssey Station — GPU-ready HF Space Dockerfile
# Works on T4/A10G GPU runtimes (full LLM stack) AND on CPU runtimes
# (Unsloth/bitsandbytes install best-effort; demo gracefully falls back to
# the heuristic policy when CUDA is absent).
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Bust HF Spaces build cache when this string changes.
ENV APP_VERSION="v5.0-gpu-sft"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Project deps first (cacheable layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU stack — install best-effort. On CPU runtimes the wheels may not exist;
# the `|| true` keeps the build green and demo/app.py falls back to heuristic.
RUN pip install --no-cache-dir bitsandbytes \
    && pip install --no-cache-dir peft accelerate \
    && pip install --no-cache-dir "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git" \
    || echo "GPU stack install partial — runtime will fall back if CUDA absent."

COPY . .

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "demo/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
