FROM registry.rcp.epfl.ch/rcp-courses/2026-ee-559:latest

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and basic system packages
RUN apt update && apt install -y \
    software-properties-common \
    git \
    wget \
    curl \
    build-essential \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv

# Use virtual environment by default
ENV PATH="/opt/venv/bin:$PATH"

# Hugging Face cache location.
# During RunAI jobs, override this to /scratch/hf_cache if /scratch is mounted.
ENV HF_HOME=/scratch/hf_cache
ENV TRANSFORMERS_CACHE=/scratch/hf_cache
ENV HF_HUB_CACHE=/scratch/hf_cache/hub

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Sanity check: make sure Qwen3-VL import works
RUN python -c "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; print('Qwen3-VL import OK')"

WORKDIR /workspace