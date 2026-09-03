from __future__ import annotations

import os
from pathlib import Path
from typing import Final

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
DEFAULT_FIXTURE_DIRECTORY: Final = Path(__file__).with_name("data")


def configured_fixture_directory() -> Path:
    configured: Final = os.environ.get(FIXTURE_DIR_ENV)
    return Path(configured).expanduser() if configured is not None else DEFAULT_FIXTURE_DIRECTORY
