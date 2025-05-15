"""
Handles checking if spend logs should be deleted based on maximum retention period
"""

from typing import Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.proxy_server import general_settings


def _should_delete_spend_logs() -> bool:
    """
    Checks if the Pod should delete spend logs based on maximum retention period

    This setting enables automatic deletion of old spend logs to manage database size.
    The maximum_spend_logs_retention_period can be specified in:
    - Days (e.g., "30d")
    - Hours (e.g., "24h")
    - Minutes (e.g., "60m")
    - Seconds (e.g., "3600s" or just "3600")
    """
    _maximum_spend_logs_retention_period: Optional[Union[int, str]] = general_settings.get(
        "maximum_spend_logs_retention_period", None
    )
    
    if _maximum_spend_logs_retention_period is None:
        return False

    try:
        if isinstance(_maximum_spend_logs_retention_period, int):
            _maximum_spend_logs_retention_period = str(_maximum_spend_logs_retention_period)
        duration_in_seconds(_maximum_spend_logs_retention_period)
        return True
    except ValueError as e:
        verbose_proxy_logger.error(
            f"Invalid maximum_spend_logs_retention_period value: {_maximum_spend_logs_retention_period}, error: {str(e)}"
        )
        return False 