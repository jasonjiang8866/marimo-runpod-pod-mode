"""Exercise the actual pinned marimo CLI and CPU torch; no RunPod account needed."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hardware_check import check


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.app = self.path / "app"
        self.app.mkdir()
        shutil.copy(ROOT / "start.sh", self.app)
        shutil.copytree(ROOT / "scripts", self.app / "scripts")
        shutil.copytree(ROOT / "notebooks", self.app / "examples")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.env = {
            "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
            "HOME": str(self.path), "RUNTIME_PROFILE": "cpu", "REQUIRE_GPU": "0",
            "NOTEBOOK_ROOT": str(self.path / "workspace with spaces"),
            "CACHE_ROOT": str(self.path / "cache"), "MARIMO_PORT": str(self.port),
            "MARIMO_HOST": "127.0.0.1", "PYTHONUNBUFFERED": "1",
        }
        self.opener = build_opener(ProxyHandler({}))

    def capture(self, **settings):
        """Use the real bootstrap but inspect its final process arguments."""
        bindir = self.path / "bin"
        bindir.mkdir(exist_ok=True)
        stub = bindir / "marimo"
        stub.write_text(f'''#!{sys.executable}
import json, os, sys
args = sys.argv[1:]
print(json.dumps({{"argv": args, "password_in_env": "MARIMO_TOKEN_PASSWORD" in os.environ,
    "stdin": sys.stdin.read() if "-" in args else ""}}))
''')
        stub.chmod(0o755)
        env = {**self.env, **settings, "PATH": f"{bindir}:{self.env['PATH']}"}
        return subprocess.run(["bash", str(self.app / "start.sh")], env=env, text=True, capture_output=True, timeout=30)

    def test_default_auth_and_no_wildcard(self):
        result = self.capture()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])["argv"]
        self.assertIn("--token", args)
        self.assertNotIn("--no-token", args)
        self.assertNotIn("--allow-origins", args)
        self.assertIn("--no-sandbox", args)
        self.assertEqual(args[-1], self.env["NOTEBOOK_ROOT"])

    def test_proxy_origins_and_password_not_in_argv(self):
        result = self.capture(MARIMO_TOKEN_PASSWORD="test-only-password", RUNPOD_POD_ID="test-pod",
                              ALLOW_ORIGINS="https://one.example,https://two.example")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout.splitlines()[-1])
        self.assertNotIn("test-only-password", data["argv"])
        self.assertFalse(data["password_in_env"])
        self.assertEqual(data["stdin"].strip(), "test-only-password")
        self.assertIn(f"test-pod-{self.port}.proxy.runpod.net:443", data["argv"])
        self.assertEqual(data["argv"].count("--allow-origins"), 2)

    def test_public_host_overrides_pod_id(self):
        result = self.capture(PUBLIC_HOST="custom.example:443", RUNPOD_POD_ID="test-pod")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])["argv"]
        self.assertIn("custom.example:443", args)
        self.assertNotIn(f"test-pod-{self.port}.proxy.runpod.net:443", args)

    def test_invalid_settings_fail_before_launch(self):
        for settings in ({"MARIMO_PORT": "0"}, {"MARIMO_PORT": "65536"}, {"MARIMO_PORT": "x"},
                         {"MARIMO_MODE": "shell"}, {"MARIMO_MODE": "run"},
                         {"NOTEBOOK_ROOT": "relative"}, {"REQUIRE_GPU": "maybe"},
                         {"MIN_GPU_MEMORY_GB": "nan"}, {"MARIMO_TOKEN_PASSWORD_FILE": "/missing"},
                         {"MARIMO_TOKEN_PASSWORD": "x", "MARIMO_TOKEN_PASSWORD_FILE": "/missing"}):
            with self.subTest(settings=settings):
                result = self.capture(**settings)
                self.assertNotEqual(result.returncode, 0)

    def test_seed_does_not_overwrite(self):
        self.assertEqual(self.capture(SEED_EXAMPLES="1").returncode, 0)
        sample = Path(self.env["NOTEBOOK_ROOT"]) / "examples/hardware_check.py"
        self.assertIn("import torch", sample.read_text())
        sample.write_text("# user data\n")
        self.assertEqual(self.capture(SEED_EXAMPLES="1").returncode, 0)
        self.assertEqual(sample.read_text(), "# user data\n")
        self.assertFalse((sample.parent / "train_grpo.py").exists())

    def start_server(self, **settings):
        env = {**self.env, **settings}
        log = open(self.path / "server.log", "w+")
        self.addCleanup(log.close)
        proc = subprocess.Popen(["bash", str(self.app / "start.sh")], env=env, stdout=log, stderr=log)
        def stop():
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.addCleanup(stop)
        base = settings.get("MARIMO_BASE_URL", "").rstrip("/")
        self.url = f"http://127.0.0.1:{self.port}{base}"
        for _ in range(150):
            if proc.poll() is not None:
                log.seek(0)
                self.fail(f"Server exited: {log.read()}")
            try:
                with self.opener.open(self.url + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(0.1)
        else:
            self.fail("Server did not become healthy")
        subprocess.run([sys.executable, str(self.app / "scripts/healthcheck.py")], env=env, check=True, timeout=5)
        return proc

    def assert_auth(self, password):
        with self.assertRaises(HTTPError) as error:
            self.opener.open(self.url + "/api/files/roots")
        self.assertIn(error.exception.code, (401, 403))
        request = Request(self.url + "/api/files/roots", headers={"Authorization": f"Bearer {password}"})
        with self.opener.open(request) as response:
            self.assertEqual(response.status, 200)

    def test_real_editor_auth_and_health(self):
        proc = self.start_server(MARIMO_TOKEN_PASSWORD="integration-test-password")
        self.assert_auth("integration-test-password")
        if Path(f"/proc/{proc.pid}/cmdline").exists():
            self.assertNotIn(b"integration-test-password", Path(f"/proc/{proc.pid}/cmdline").read_bytes())

    def test_real_token_file_proxy_base_url(self):
        token = self.path / "token"
        token.write_text("integration-file-password\n")
        self.start_server(MARIMO_TOKEN_PASSWORD_FILE=str(token), MARIMO_BASE_URL="/notebooks",
                          PUBLIC_HOST="test.example:443")
        self.assert_auth("integration-file-password")

    def test_real_random_token_blocks_editor(self):
        self.start_server()
        with self.assertRaises(HTTPError) as error:
            self.opener.open(self.url + "/api/files/roots")
        self.assertIn(error.exception.code, (401, 403))

    def test_real_run_mode(self):
        self.start_server(MARIMO_MODE="run", NOTEBOOK_FILE=str(self.app / "examples/hardware_check.py"),
                          MARIMO_TOKEN_PASSWORD="run-test-password")
        # Run mode cannot expose edit endpoints, even with the correct password.
        request = Request(self.url + "/api/files/roots", headers={"Authorization": "Bearer run-test-password"})
        with self.assertRaises(HTTPError) as error:
            self.opener.open(request)
        self.assertIn(error.exception.code, (401, 403, 404))

    def test_real_cpu_smoke_and_gpu_failure(self):
        import torch
        result = check(torch, False)
        self.assertEqual(result["devices"][0]["smoke"], "passed")
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                check(torch, True)
            result = self.capture(RUNTIME_PROFILE="cu128", REQUIRE_GPU="1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA is unavailable", result.stderr)

    def test_mig_memory_gate_before_allocation(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="13.2"),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                get_device_properties=lambda _: SimpleNamespace(name="MIG slice", total_memory=24 * 1024**3, major=12, minor=0)))
        with self.assertRaisesRegex(RuntimeError, "below MIN_GPU_MEMORY_GB=48"):
            check(fake, True, 48)


if __name__ == "__main__":
    unittest.main()
