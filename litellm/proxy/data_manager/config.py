from typing import Final

from litellm.secret_managers.main import get_secret_bool

LITELLM_DATA_MANAGER_ENABLED_ENV: Final = "LITELLM_DATA_MANAGER_ENABLED"


def data_manager_enabled() -> bool:
    return get_secret_bool(LITELLM_DATA_MANAGER_ENABLED_ENV, False) is True
