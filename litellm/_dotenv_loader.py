import os
from typing import Final

_TRUTHY_VALUES: Final = frozenset({"1", "true", "t", "yes", "y"})


def should_load_dotenv() -> bool:
    disable_dotenv: Final = os.getenv("LITELLM_DISABLE_DOTENV", "").strip().casefold()
    return os.getenv("LITELLM_MODE", "DEV") == "DEV" and disable_dotenv not in _TRUTHY_VALUES
