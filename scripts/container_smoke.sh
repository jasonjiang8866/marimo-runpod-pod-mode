#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:?Pass an image tag}"
WORKSPACE="$(mktemp -d)"
cleanup() {
  docker rm -f marimo-smoke >/dev/null 2>&1 || true
  # The default RunPod user is root; return bind-mounted test files to the host
  # user before removing this newly-created temporary directory.
  docker run --rm --user 0:0 --entrypoint /bin/chown \
    -v "$WORKSPACE:/cleanup" "$IMAGE" -R "$(id -u):$(id -g)" /cleanup
  rm -rf -- "$WORKSPACE"
}
trap cleanup EXIT
printf '# retained user file\n' > "$WORKSPACE/user.py"
# A hosted CI runner has no GPU. This tests imports/server boot only, never CUDA.
docker run -d --name marimo-smoke \
  -e REQUIRE_GPU=0 -e SEED_EXAMPLES=1 \
  -e MARIMO_TOKEN_PASSWORD=ci-only-test-password \
  -v "$WORKSPACE:/workspace" "$IMAGE"
for _ in $(seq 1 90); do
  STATUS="$(docker inspect --format='{{.State.Health.Status}}' marimo-smoke)"
  [[ "$STATUS" != healthy ]] || break
  [[ "$(docker inspect --format='{{.State.Running}}' marimo-smoke)" == true ]] || {
    docker logs marimo-smoke; exit 1;
  }
  sleep 2
done
[[ "$STATUS" == healthy ]] || { docker logs marimo-smoke; exit 1; }
docker exec marimo-smoke python -c 'import marimo, torch; print(marimo.__version__, torch.__version__, torch.version.cuda)'
docker exec -i marimo-smoke python - <<'PY'
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
assert Path('/workspace/user.py').read_text() == '# retained user file\n'
assert Path('/workspace/examples/hardware_check.py').exists()
url = 'http://127.0.0.1:8080/api/files/roots'
try:
    urlopen(url)
except HTTPError as exc:
    assert exc.code in (401, 403), exc.code
else:
    raise AssertionError('Editor is reachable without authentication')
with urlopen(Request(url, headers={'Authorization': 'Bearer ci-only-test-password'})) as response:
    assert response.status == 200
PY
# Check a clean restart preserves the workspace and examples.
docker restart marimo-smoke >/dev/null
for _ in $(seq 1 45); do
  if docker exec marimo-smoke python /opt/marimo/scripts/healthcheck.py 2>/dev/null; then
    break
  fi
  sleep 2
done
docker exec marimo-smoke python /opt/marimo/scripts/healthcheck.py
# tini forwards termination to the server process group.
docker stop --time 10 marimo-smoke >/dev/null
[[ "$(docker inspect --format='{{.State.ExitCode}}' marimo-smoke)" != 137 ]]
