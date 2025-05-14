"""
Handles checking if spend logs should be deleted based on maximum retention period
"""

from typing import Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_server import general_settings
from litellm.secret_managers.main import str_to_bool


def _should_delete_spend_logs() -> bool:
    """
    Checks if the Pod should delete spend logs based on maximum retention period

    This setting enables automatic deletion of old spend logs to manage database size
    """
    _maximum_retention_period: Optional[Union[int, str]] = general_settings.get(
        "maximum_retention_period", None
    )
    if isinstance(_maximum_retention_period, str):
        try:
            _maximum_retention_period = int(_maximum_retention_period)
        except ValueError:
            verbose_proxy_logger.error(
                f"Invalid maximum_retention_period value: {_maximum_retention_period}"
            )
            return False
    if _maximum_retention_period is None:
        return False
    return True 