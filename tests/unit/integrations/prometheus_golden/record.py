from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[4]
SUITE: Final = "tests/unit/integrations/test_prometheus_native_parity.py"
GOLDEN: Final = Path(__file__).with_name("python_outputs.json")


def main() -> int:
    GOLDEN.unlink(missing_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)
        patch.setenv("PROMETHEUS_GOLDEN_RECORD", "1")
        return int(pytest.main(["-q", "-p", "no:cacheprovider", SUITE]))


if __name__ == "__main__":
    raise SystemExit(main())
