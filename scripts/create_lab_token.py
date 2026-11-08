"""Create a local, ignored Jupyter token without displaying it."""

import os
import secrets
from pathlib import Path

directory = Path(".local")
directory.mkdir(exist_ok=True)
(directory / "runs").mkdir(exist_ok=True)
token = directory / "jupyter_token"
if not token.exists():
    descriptor = os.open(token, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(secrets.token_urlsafe(32))
print("Jupyter token is available in .local/jupyter_token; view it locally at login.")
