import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import litellm


def main() -> None:
    assert "site-packages" in Path(litellm.__file__).resolve().parts, "install the reviewed wheel first"
    spec: Final = importlib.util.find_spec("litellm.rust_bridge._native")
    assert spec is not None and spec.origin is not None, "wheel must contain the native extension"
    native: Final = Path(spec.origin)
    hidden: Final = native.with_suffix(native.suffix + ".disabled")
    assert not hidden.exists()
    script: Final = Path(__file__).resolve().parents[1] / "ocr" / "sdk_callback_contract.py"
    environment: Final = {
        **{key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")},
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    with tempfile.TemporaryDirectory(prefix="ocr-sdk-callbacks-") as directory:
        subprocess.run(
            [sys.executable, str(script), "--installed"], cwd=directory, env=environment, check=True, timeout=180
        )
        native.rename(hidden)
        try:
            subprocess.run(
                [sys.executable, str(script), "--installed", "--without-native"],
                cwd=directory,
                env=environment,
                check=True,
                timeout=60,
            )
        finally:
            hidden.rename(native)
    sys.stdout.write("Installed-wheel callback parity and unavailable-extension fallback passed\n")


if __name__ == "__main__":
    main()
