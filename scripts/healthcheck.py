"""Check the unauthenticated liveness endpoint without exposing the editor."""
import json
import os
from urllib.request import ProxyHandler, build_opener


def main():
    port = int(os.environ.get("MARIMO_PORT", "8080"))
    base = os.environ.get("MARIMO_BASE_URL", "").rstrip("/")
    host = os.environ.get("MARIMO_HOST", "0.0.0.0")
    host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    if ":" in host:
        host = f"[{'::1' if host == '::' else host}]"
    # Ignore proxy environment variables for this loopback request.
    with build_opener(ProxyHandler({})).open(f"http://{host}:{port}{base}/health", timeout=3) as response:
        if response.status != 200 or json.load(response).get("status") != "healthy":
            raise SystemExit("marimo health endpoint is not healthy")


if __name__ == "__main__":
    main()
