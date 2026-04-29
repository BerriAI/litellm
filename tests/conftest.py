import os
from pathlib import Path


if "LITELLM_UI_PATH" not in os.environ:
    worker = os.environ.get("PYTEST_XDIST_WORKER", f"pid-{os.getpid()}")
    os.environ["LITELLM_UI_PATH"] = str(
        Path(__file__).resolve().parents[1]
        / ".pytest_cache"
        / f"litellm-ui-{worker}"
    )
