"""Generic API logger on the v1 contract: batches of call records to one HTTP endpoint.

Legacy twin: `litellm.integrations.generic_api.generic_api_callback.GenericAPILogger`.
The legacy logger ships the whole `StandardLoggingPayload`; this port ships the fields of
it that v1 facts can fill, under the same names; `manifest.py` lists the rest as gaps.

Not ported yet: `callback_name` presets from `generic_api_compatible_callbacks.json`,
retries, and the pre-StandardLoggingPayload body behind `litellm.generic_api_use_v1`.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.callbacks_v1 import JSONValue
from litellm.callbacks_v1.builtin.port import Batching, CallRecord, Delivery, SinkPort

LogFormat: TypeAlias = Literal["json_array", "ndjson", "single"]
ApiEventType: TypeAlias = Literal["llm_api_success", "llm_api_failure"]


@dataclass(frozen=True, slots=True)
class Config:
    endpoint: str
    headers: tuple[tuple[str, str], ...] = (("Content-Type", "application/json"),)
    event_types: frozenset[ApiEventType] = frozenset({"llm_api_success", "llm_api_failure"})
    log_format: LogFormat = "json_array"
    batch_size: int = 512
    flush_interval: float = 5.0

    @staticmethod
    def from_env(env: Mapping[str, str]) -> "Config":
        endpoint: Final = env.get("GENERIC_LOGGER_ENDPOINT")
        if endpoint is None:
            raise ValueError("endpoint not set for GenericAPILogger, GENERIC_LOGGER_ENDPOINT not found")
        pairs: Final = (item.split("=", 1) for item in env.get("GENERIC_LOGGER_HEADERS", "").split(",") if "=" in item)
        return Config(
            endpoint=endpoint,
            headers=(("Content-Type", "application/json"), *((key.strip(), value.strip()) for key, value in pairs)),
        )


class ErrorInformation(TypedDict):
    error_class: ReadOnly[str]
    error_message: ReadOnly[str]
    error_code: ReadOnly[str]


class Record(TypedDict):
    """The `StandardLoggingPayload` fields v1 can fill, under their legacy names."""

    id: ReadOnly[str]
    litellm_call_id: ReadOnly[str]
    call_type: ReadOnly[str]
    stream: ReadOnly[bool]
    status: ReadOnly[Literal["success", "failure"]]
    startTime: ReadOnly[float]
    endTime: ReadOnly[float]
    response_time: ReadOnly[float]
    model: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    model_parameters: ReadOnly[JSONValue]
    response: ReadOnly[NotRequired[JSONValue]]
    error_str: ReadOnly[NotRequired[str]]
    error_information: ReadOnly[NotRequired[ErrorInformation]]


@dataclass(frozen=True, slots=True)
class GenericApi(SinkPort[Record]):
    """One record per finished call, batched into one endpoint."""

    config: Config

    @property
    def batching(self) -> Batching:
        return Batching(self.config.batch_size, self.config.flush_interval)

    def payload(self, call: CallRecord, _now: datetime) -> Record | None:
        """The record for a finished call, or `None` when `config.event_types` leaves its outcome out.
        A failed call has `error_*` where a successful one has `response`. The time is the call's own,
        so the clock the host passes is unused."""
        terminal: Final = call.terminal
        wanted: Final = "llm_api_success" if terminal["type"] == "call.succeeded" else "llm_api_failure"
        if wanted not in self.config.event_types:
            return None
        request: Final = call.request
        response: Final = terminal["response"] if terminal["type"] == "call.succeeded" else None
        response_id: Final = response.get("id") if isinstance(response, Mapping) else None
        common: Final[Record] = {
            "id": response_id if isinstance(response_id, str) else call.call_id,
            "litellm_call_id": call.call_id,
            "call_type": call.call_type,
            "stream": terminal["streamed"],
            "status": "success" if terminal["type"] == "call.succeeded" else "failure",
            "startTime": terminal["timing"]["start_time"],
            "endTime": terminal["timing"]["end_time"],
            "response_time": terminal["timing"]["duration"],
            "model": request["model"] if request is not None else None,
            "custom_llm_provider": request["custom_llm_provider"] if request is not None else None,
            "model_parameters": request["optional_params"] if request is not None else None,
        }
        if terminal["type"] == "call.succeeded":
            succeeded: Final[Record] = {**common, "response": terminal["response"]}
            return succeeded
        error: Final = terminal["error"]
        failed: Final[Record] = {
            **common,
            "error_str": error["message"],
            "error_information": {
                "error_class": error["class"].rsplit(".", 1)[-1],
                "error_message": error["message"],
                "error_code": str(error["status_code"]) if error["status_code"] is not None else "",
            },
        }
        return failed

    def deliveries(self, batch: tuple[Record, ...]) -> tuple[Delivery, ...]:
        if self.config.log_format == "single":
            return tuple(Delivery(self.config.endpoint, self.config.headers, json.dumps(r).encode()) for r in batch)
        body: Final = (
            json.dumps(batch)
            if self.config.log_format == "json_array"
            else "\n".join(json.dumps(record) for record in batch)
        )
        return (Delivery(self.config.endpoint, self.config.headers, body.encode()),)
