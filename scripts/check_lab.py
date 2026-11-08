"""Check Jupyter authentication inside an ephemeral, isolated project container."""

import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

token = Path("/run/secrets/jupyter_token").read_text().strip()
with tempfile.TemporaryFile() as logs:
    process = subprocess.Popen(
        ["jupyter", "lab", "--config=/etc/jupyter/jupyter_server_config.py"],
        stdout=logs,
        stderr=logs,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError(
                    "Jupyter exited before readiness; logs intentionally not printed"
                )
            try:
                urllib.request.urlopen("http://127.0.0.1:8888/login", timeout=1).close()
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        else:
            raise TimeoutError("Jupyter did not become ready")
        url = "http://127.0.0.1:8888/api/contents"
        try:
            urllib.request.urlopen(url, timeout=3).close()
        except urllib.error.HTTPError as error:
            assert error.code in (401, 403)
        else:
            raise AssertionError("Anonymous contents access was accepted")
        request = urllib.request.Request(url, headers={"Authorization": f"token {token}"})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 200
        print(
            json.dumps(
                {"lab_started": True, "anonymous_rejected": True, "authenticated_contents": True}
            )
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
