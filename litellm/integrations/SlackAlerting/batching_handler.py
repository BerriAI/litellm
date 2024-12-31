"""
Handles Batching + sending Httpx Post requests to slack 

Slack alerts are sent every 10s or when events are greater than X events 

see custom_batch_logger.py for more details / defaults 
"""

from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from .slack_alerting import SlackAlerting as _SlackAlerting

    SlackAlertingType = _SlackAlerting
else:
    SlackAlertingType = Any


def squash_payloads(queue):

    squashed = {}
    if len(queue) == 0:
        return squashed
    if len(queue) == 1:
        return {"key": {"item": queue[0], "count": 1}}

    for item in queue:
        url = item["url"]
        alert_type = item["alert_type"]
        _key = (url, alert_type)

        if _key in squashed:
            squashed[_key]["count"] += 1
            # Merge the payloads

        else:
            squashed[_key] = {"item": item, "count": 1}

    return squashed


async def send_to_webhook(slackAlertingInstance: SlackAlertingType, item, count):
    import json

    try:
        payload = item["payload"]
        if count > 1:
            payload["text"] = f"[Num Alerts: {count}]\n\n{payload['text']}"

        response = await slackAlertingInstance.async_http_handler.post(
            url=item["url"],
            headers=item["headers"],
            data=json.dumps(payload),
        )
        if response.status_code != 200:
            verbose_proxy_logger.debug(
                f"Error sending slack alert to url={item['url']}. Error={response.text}"
            )
    except Exception as e:
        verbose_proxy_logger.debug(f"Error sending slack alert: {str(e)}")
