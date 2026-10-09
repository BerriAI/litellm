import os
from typing import Final

from litellm.secret_managers.main import get_secret_bool

LITELLM_DATA_MANAGER_ENABLED_ENV: Final = "LITELLM_DATA_MANAGER_ENABLED"
DATA_MANAGER_JOB_ROLE: Final = "data_manager"


def data_manager_enabled() -> bool:
    return get_secret_bool(LITELLM_DATA_MANAGER_ENABLED_ENV, False) is True


def running_as_data_manager() -> bool:
    return os.environ.get("LITELLM_JOB_ROLE") == DATA_MANAGER_JOB_ROLE


def proxy_skips_spend_log_cleanup() -> bool:
    return data_manager_enabled() and not running_as_data_manager()
