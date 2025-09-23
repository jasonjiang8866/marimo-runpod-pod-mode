# CUDA-enabled base so HF libs can use GPU on NVIDIA pods
FROM runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204

# Platform niceties that Runpod containers expect
# (Jupyter/SSH help debugging; not required for Marimo itself but common in Runpod templates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server nginx && rm -rf /var/lib/apt/lists/*


# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Python deps
WORKDIR /workspace
COPY requirements.txt /workspace/
# activate a virtual environment
RUN uv venv
# install into the virtual environment all dependencies
RUN uv pip install -r requirements.txt

# Marimo port
ENV MARIMO_PORT=8080
EXPOSE 8080

# Start Marimo in headless mode, served on 0.0.0.0
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/bin/bash", "-lc", "/start.sh"]
