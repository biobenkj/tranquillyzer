FROM mambaorg/micromamba:1.5.5 AS builder

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:$PATH

WORKDIR /build
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /build/environment.yml

# Install into base instead of creating a new env
RUN micromamba install -y -n base -f /build/environment.yml && \
    micromamba clean --all --yes

# GPU TF + addons into base
RUN micromamba run -n base \
      pip install "tensorflow[and-cuda]==2.15.1" --extra-index-url https://pypi.nvidia.com && \
    micromamba run -n base pip install "tensorflow-addons==0.22.*"

# Stage 2: runtime image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Copy conda env from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    curl git wget bzip2 libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN if [ -n "$MODEL_ZIP_URL" ]; then \
        echo "Downloading model bundle from $MODEL_ZIP_URL"; \
        mkdir -p models && \
        curl -L "$MODEL_ZIP_URL" -o /tmp/models.zip && \
        unzip /tmp/models.zip -d models && \
        rm /tmp/models.zip; \
    else \
        echo "MODEL_ZIP_URL not provided; skipping model download."; \
    fi

RUN rm -rf *.egg-info && pip install --no-cache-dir -e .

CMD ["tranquillyzer", "--help"]
