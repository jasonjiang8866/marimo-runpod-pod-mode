# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.7
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE}

ARG RUNTIME_PROFILE=cu128
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/jasonjiang8866/marimo-runpod-pod-mode" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

# Wheels supply user-space CUDA; the host supplies the driver. No nvcc or SSH.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUNTIME_PROFILE=${RUNTIME_PROFILE} \
    MARIMO_PORT=8080

WORKDIR /opt/marimo
COPY requirements.txt profiles.json ./
COPY scripts/install_runtime.py scripts/install_runtime.py
RUN uv venv /opt/venv \
    && python scripts/install_runtime.py "${RUNTIME_PROFILE}" \
    && uv pip install --no-cache -r requirements.txt \
    && uv pip check \
    && python -c "import marimo, torch; print(marimo.__version__, torch.__version__, torch.version.cuda)" \
    && uv pip freeze > build-packages.txt
COPY scripts/ scripts/
COPY start.sh ./
COPY notebooks/ examples/
RUN chmod +x start.sh && mkdir -p /workspace

WORKDIR /workspace
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "/opt/marimo/scripts/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["/opt/marimo/start.sh"]
