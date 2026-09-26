from importlib.resources import files
from typing import Final


def _load(name: str) -> str:
    return files(__name__).joinpath(f"{name}.sql").read_text(encoding="utf-8")


KEY_AUTH_COMBINED_VIEW: Final = _load("key_auth_combined_view")
