import os
from collections.abc import Mapping


def should_hide_default_credentials_hint(general_settings: Mapping[str, object]) -> bool:
    return (
        os.getenv("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", "false").lower() == "true"
        or general_settings.get("hide_default_credentials_hint", False) is True
        or bool(os.getenv("UI_PASSWORD"))
    )
