"""
Handles Batching + sending Httpx Post requests to slack

Slack alerts are sent every DEFAULT_FLUSH_INTERVAL_SECONDS or when events are greater than X events

see custom_batch_logger.py for more details / defaults
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.types.integrations.slack_alerting import AlertQueueItem, AlertType

from .ms_teams import MS_TEAMS_ALERTING_DESTINATION, build_ms_teams_payload

if TYPE_CHECKING:
    from .slack_alerting import SlackAlerting as _SlackAlerting

    SlackAlertingType = _SlackAlerting
else:
    SlackAlertingType = Any


@dataclass(frozen=True, slots=True)
class SquashedAlert:
    item: AlertQueueItem
    count: int


def _squash_key(item: AlertQueueItem) -> tuple[str, AlertType | str, str]:
    return (item["url"], item["alert_type"], item["payload"]["text"])


def squash_payloads(queue: Sequence[AlertQueueItem]) -> tuple[SquashedAlert, ...]:
    counts: Final = Counter(_squash_key(item) for item in queue)
    first_item_by_key: Final = {_squash_key(item): item for item in reversed(queue)}
    return tuple(SquashedAlert(item=first_item_by_key[key], count=count) for key, count in counts.items())


def _print_alerting_payload_warning(payload: dict, slackAlertingInstance: SlackAlertingType):
    """
    Print the payload to the console when
    slackAlertingInstance.alerting_args.log_to_console is True

    Relevant issue: https://github.com/BerriAI/litellm/issues/7372
    """
    if slackAlertingInstance.alerting_args.log_to_console is True:
        verbose_proxy_logger.warning(payload)


async def send_to_webhook(slackAlertingInstance: SlackAlertingType, item: AlertQueueItem, count: int) -> None:
    """
    Send a single slack alert to the webhook
    """
    import json

    text: Final = item["payload"]["text"]
    payload: Final = {"text": text if count == 1 else f"[Num Alerts: {count}]\n\n{text}"}
    try:
        request_body: Final = (
            build_ms_teams_payload(payload["text"]) if item.get("format") == MS_TEAMS_ALERTING_DESTINATION else payload
        )
        response: Final = await slackAlertingInstance.async_http_handler.post(
            url=item["url"],
            headers=item["headers"],
            data=json.dumps(request_body),
        )
        if response.status_code != 200:
            verbose_proxy_logger.debug("Error sending alert to url=%s. Error=%s", item["url"], response.text)
    except Exception as e:
        verbose_proxy_logger.debug("Error sending alert: %s", e)
    finally:
        _print_alerting_payload_warning(payload, slackAlertingInstance=slackAlertingInstance)
