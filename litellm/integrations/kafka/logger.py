import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.kafka.config import resolve_params
from litellm.integrations.kafka.producer import KafkaExportStats, KafkaSink, ProducerFactory, load_producer_factory
from litellm.litellm_core_utils.redact_messages import (
    redacted_standard_logging_payload,
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy helper takes untyped kwargs
)
from litellm.types.integrations.kafka import KafkaInitParams

_MAPPING: Final = TypeAdapter(Mapping[str, object])


class KafkaLogger(CustomLogger):
    def __init__(
        self,
        params: KafkaInitParams | None = None,
        producer_factory: ProducerFactory | None = None,
        *,
        start_polling: bool = True,
    ) -> None:
        configured: Final = params if params is not None else litellm.kafka_params
        resolved: Final = resolve_params(configured)
        super().__init__(turn_off_message_logging=bool(resolved.turn_off_message_logging))  # pyright: ignore[reportUnknownMemberType]  # legacy base accepts untyped extra kwargs
        self.sink: Final = KafkaSink(
            resolved,
            producer_factory if producer_factory is not None else load_producer_factory(),
            self._callback_failed,
            start_polling=start_polling,
        )

    def _callback_failed(self) -> None:
        self.handle_callback_failure(callback_name="KafkaLogger")

    @property
    def statistics(self) -> KafkaExportStats:
        return self.sink.statistics

    def _log(self, kwargs: Mapping[str, object]) -> None:
        try:
            if kwargs.get("standard_logging_object") is None:
                return
            details: Final = _MAPPING.validate_python(
                self.redact_standard_logging_payload_from_model_call_details(dict(kwargs))  # pyright: ignore[reportUnknownMemberType]  # TypeAdapter validates the legacy helper result below
            )
            payload: Final = _MAPPING.validate_python(details.get("standard_logging_object"))
            redacted: Final = (
                redacted_standard_logging_payload(payload) if should_redact_message_logging(dict(kwargs)) else payload
            )
            self.sink.publish(redacted)
        except Exception:  # noqa: BLE001  # malformed log data must not change a model response
            self.sink.drop("invalid_payload")

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._log(kwargs)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._log(kwargs)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await asyncio.to_thread(self._log, kwargs)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await asyncio.to_thread(self._log, kwargs)

    async def flush_queue(self) -> None:
        _ = await asyncio.to_thread(self.sink.flush)

    def close(self, timeout: float | None = None) -> None:
        self.sink.close(timeout)
