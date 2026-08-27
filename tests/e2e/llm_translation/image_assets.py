"""Image assets the vision tests send, kept in-repo so no third-party host can fail a run."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Final

FIXTURES_DIR: Final = Path(__file__).parent / "fixtures"
CAT_IMAGE: Final = FIXTURES_DIR / "cat.jpg"


def cat_image_data_url() -> str:
    return "data:image/jpeg;base64," + base64.b64encode(CAT_IMAGE.read_bytes()).decode()
