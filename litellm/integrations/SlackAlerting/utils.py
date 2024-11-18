"""
Utils used for slack alerting
"""

import asyncio
from typing import Dict, List, Optional, Union

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import AlertType
from litellm.secret_managers.main import get_secret


def process_slack_alerting_variables(
    alert_to_webhook_url: Optional[Dict[AlertType, Union[List[str], str]]]
) -> Optional[Dict[AlertType, Union[List[str], str]]]:
    """
    process alert_to_webhook_url
    - check if any urls are set as os.environ/SLACK_WEBHOOK_URL_1 read env var and set the correct value
    """
    if alert_to_webhook_url is None:
        return None

    for alert_type, webhook_urls in alert_to_webhook_url.items():
        if isinstance(webhook_urls, list):
            _webhook_values: List[str] = []
            for webhook_url in webhook_urls:
                if "os.environ/" in webhook_url:
                    _env_value = get_secret(secret_name=webhook_url)
                    if not isinstance(_env_value, str):
                        raise ValueError(
                            f"Invalid webhook url value for: {webhook_url}. Got type={type(_env_value)}"
                        )
                    _webhook_values.append(_env_value)
                else:
                    _webhook_values.append(webhook_url)

            alert_to_webhook_url[alert_type] = _webhook_values
        else:
            _webhook_value_str: str = webhook_urls
            if "os.environ/" in webhook_urls:
                _env_value = get_secret(secret_name=webhook_urls)
                if not isinstance(_env_value, str):
                    raise ValueError(
                        f"Invalid webhook url value for: {webhook_urls}. Got type={type(_env_value)}"
                    )
                _webhook_value_str = _env_value
            else:
                _webhook_value_str = webhook_urls

            alert_to_webhook_url[alert_type] = _webhook_value_str

    return alert_to_webhook_url


async def _add_langfuse_trace_id_to_alert(
    request_data: Optional[dict] = None,
) -> Optional[str]:
    """
    Returns langfuse trace url

    - check:
    -> existing_trace_id
    -> trace_id
    -> litellm_call_id
    """
    # do nothing for now
    if (
        request_data is not None
        and request_data.get("litellm_logging_obj", None) is not None
    ):
        trace_id: Optional[str] = None
        litellm_logging_obj: Logging = request_data["litellm_logging_obj"]

        for _ in range(3):
            trace_id = litellm_logging_obj._get_trace_id(service_name="langfuse")
            if trace_id is not None:
                break
            await asyncio.sleep(3)  # wait 3s before retrying for trace id

        _langfuse_object = litellm_logging_obj._get_callback_object(
            service_name="langfuse"
        )
        if _langfuse_object is not None:
            base_url = _langfuse_object.Langfuse.base_url
            return f"{base_url}/trace/{trace_id}"
    return None
