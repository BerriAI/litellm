from pathlib import Path
from typing import Final

import pytest

TEST_FILE: Final = "tests/unit/proxy/hooks/test_model_capacity_native.py"
GOLDEN_FILE: Final = Path(__file__).resolve().parent.parent / "test_model_capacity_native.golden.json"


def main() -> int:
    GOLDEN_FILE.unlink(missing_ok=True)
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("MODEL_CAPACITY_RECORD_GOLDEN", "1")
        return int(pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-p", "no:randomly", TEST_FILE]))


if __name__ == "__main__":
    raise SystemExit(main())
