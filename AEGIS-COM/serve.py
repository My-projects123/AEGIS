"""Launch the AEGIS-COM website.

    python serve.py
    open http://127.0.0.1:8080

On a Raspberry Pi, the same process is reachable as http://<pi-ip>:8080
because it binds 0.0.0.0. Override with HOST / PORT if needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    print(f"AEGIS-COM dashboard → http://127.0.0.1:{port}")
    print(f"LAN / Raspberry Pi  → http://<this-machine-ip>:{port}")
    print("Laptop software-in-the-loop POC. Not physical ANC.")
    uvicorn.run("backend.api:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
