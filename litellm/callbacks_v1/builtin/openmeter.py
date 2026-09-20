"""OpenMeter on the v1 contract: one CloudEvent per successful call.

Legacy twin: `litellm.integrations.openmeter.OpenMeterLogger`.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.callbacks_v1 import JSONValue
from litellm.callbacks_v1.builtin.port import Batching, CallRecord, Delivery, SinkPort

_USAGE_KEYS: Final = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True, slots=True)
class Config:
    api_key: str
    endpoint: str = "https://openmeter.cloud"
    event_type: str = "litellm_tokens"

    @staticmethod
    def from_env(env: Mapping[str, str]) -> "Config":
        api_key: Final = env.get("OPENMETER_API_KEY")
        if api_key is None:
            raise ValueError("Missing keys=['OPENMETER_API_KEY'] in environment.")
        return Config(
            api_key=api_key,
            endpoint=env.get("OPENMETER_API_ENDPOINT", "https://openmeter.cloud"),
            event_type=env.get("OPENMETER_EVENT_TYPE", "litellm_tokens"),
        )

    @property
    def url(self) -> str:
        return f"{self.endpoint.rstrip('/')}/api/v1/events"


class Usage(TypedDict):
    prompt_tokens: ReadOnly[JSONValue]
    completion_tokens: ReadOnly[JSONValue]
    total_tokens: ReadOnly[JSONValue]


class MeterData(TypedDict):
    model: ReadOnly[str | None]
    cost: ReadOnly[float | None]
    prompt_tokens: ReadOnly[NotRequired[JSONValue]]
    completion_tokens: ReadOnly[NotRequired[JSONValue]]
    total_tokens: ReadOnly[NotRequired[JSONValue]]


class CloudEvent(TypedDict):
    specversion: ReadOnly[str]
    type: ReadOnly[str]
    id: ReadOnly[str]
    time: ReadOnly[str]
    subject: ReadOnly[str]
    source: ReadOnly[str]
    data: ReadOnly[MeterData]


def _usage(response: JSONValue) -> Usage | None:
    """Token counts when the response carries them in the OpenAI shape, as the legacy logger reads them."""
    usage: Final = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping) or not any(key in usage for key in _USAGE_KEYS):
        return None
    counted: Final[Usage] = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens"),
    }
    return counted


def _metered(model: str | None, usage: Usage) -> MeterData:
    data: Final[MeterData] = {"model": model, "cost": None, **usage}
    return data


@dataclass(frozen=True, slots=True)
class OpenMeter(SinkPort[CloudEvent]):
    """One CloudEvent per successful call, one request per event."""

    config: Config

    @property
    def batching(self) -> Batching:
        """One event per request, as the legacy logger posts them."""
        return Batching()

    def payload(self, call: CallRecord, now: datetime) -> CloudEvent | None:
        """The CloudEvent for a finished call, or `None` when OpenMeter has nothing to meter."""
        if call.terminal["type"] != "call.succeeded":
            return None
        subject: Final = call.started["metadata"].get("user_api_key_user_id") if call.started is not None else None
        if subject is None:
            raise ValueError("OpenMeter: user is required")
        response: Final = call.terminal["response"]
        response_id: Final = response.get("id") if isinstance(response, Mapping) else None
        model: Final = call.request["model"] if call.request is not None else None
        usage: Final = _usage(response)
        unmetered: Final[MeterData] = {"model": model, "cost": None}
        data: Final = unmetered if usage is None else _metered(model, usage)
        event: Final[CloudEvent] = {
            "specversion": "1.0",
            "type": self.config.event_type,
            "id": response_id if isinstance(response_id, str) else call.call_id,
            "time": now.isoformat(),
            "subject": str(subject),
            "source": "litellm-proxy",
            "data": data,
        }
        return event

    def deliveries(self, batch: tuple[CloudEvent, ...]) -> tuple[Delivery, ...]:
        """One request per CloudEvent, as the legacy logger posts them."""
        headers: Final = (
            ("Content-Type", "application/cloudevents+json"),
            ("Authorization", f"Bearer {self.config.api_key}"),
        )
        return tuple(Delivery(self.config.url, headers, json.dumps(event).encode()) for event in batch)
