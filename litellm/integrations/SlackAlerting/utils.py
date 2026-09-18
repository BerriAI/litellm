"""
Utils used for slack alerting
"""

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import AlertType
from litellm.secret_managers.main import get_secret

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _Logging

    Logging = _Logging
else:
    Logging = Any


def process_slack_alerting_variables(
    alert_to_webhook_url: dict[AlertType, list[str] | str] | None,
) -> dict[AlertType, list[str] | str] | None:
    """
    process alert_to_webhook_url
    - check if any urls are set as os.environ/SLACK_WEBHOOK_URL_1 read env var and set the correct value
    """
    if alert_to_webhook_url is None:
        return None

    for alert_type, webhook_urls in alert_to_webhook_url.items():
        if isinstance(webhook_urls, list):
            _webhook_values: list[str] = []
            for webhook_url in webhook_urls:
                if "os.environ/" in webhook_url:
                    _env_value = get_secret(secret_name=webhook_url)
                    if not isinstance(_env_value, str):
                        raise ValueError(f"Invalid webhook url value for: {webhook_url}. Got type={type(_env_value)}")
                    _webhook_values.append(_env_value)
                else:
                    _webhook_values.append(webhook_url)

            alert_to_webhook_url[alert_type] = _webhook_values
        else:
            _webhook_value_str: str = webhook_urls
            if "os.environ/" in webhook_urls:
                _env_value = get_secret(secret_name=webhook_urls)
                if not isinstance(_env_value, str):
                    raise ValueError(f"Invalid webhook url value for: {webhook_urls}. Got type={type(_env_value)}")
                _webhook_value_str = _env_value
            else:
                _webhook_value_str = webhook_urls

            alert_to_webhook_url[alert_type] = _webhook_value_str

    return alert_to_webhook_url


async def _add_langfuse_trace_id_to_alert(
    request_data: dict | None = None,
) -> str | None:
    """
    Returns langfuse trace url

    - check:
    -> existing_trace_id
    -> trace_id
    -> litellm_call_id
    """
    from litellm.integrations.langfuse.langfuse import LangFuseLogger, resolve_langfuse_host

    callbacks: Final[list[CustomLogger | Callable[..., object] | str]] = (
        litellm.logging_callback_manager._get_all_callbacks()
    )
    if not any(callback == "langfuse" or isinstance(callback, LangFuseLogger) for callback in callbacks):
        return None

    if request_data is None or request_data.get("litellm_logging_obj", None) is None:
        return None

    litellm_logging_obj: Final[Logging] = request_data["litellm_logging_obj"]
    instance_host: Final = next(
        (callback.langfuse_host for callback in callbacks if isinstance(callback, LangFuseLogger)), None
    )
    host: Final = resolve_langfuse_host(
        litellm_logging_obj.standard_callback_dynamic_params.get("langfuse_host") or instance_host
    )
    for _ in range(3):
        if (trace_id := litellm_logging_obj._get_trace_id(service_name="langfuse")) is not None:
            return f"{host}/trace/{trace_id}"
        await asyncio.sleep(3)  # wait 3s before retrying for trace id

    return None
