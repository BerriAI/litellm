"""
Utils used for slack alerting
"""

from typing import Dict, List, Optional, Union

import litellm
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
