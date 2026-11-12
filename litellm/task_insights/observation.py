import math
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from hashlib import sha256
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG, TaskObservation, UnobservedDelivery
from litellm.proxy.spend_tracking.spend_tracking_utils import user_turn_texts

MAX_REQUEST_TEXT_CHARS: Final = 2000
MAX_CONTEXT_TEXT_CHARS: Final = 1000

_FIELDS: Final = TypeAdapter(dict[str, object])
_VALUES: Final = TypeAdapter(tuple[object, ...])
_TIMESTAMP: Final = TypeAdapter(datetime)


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _FIELDS.validate_json(value) if isinstance(value, str) else _FIELDS.validate_python(value)
    except ValidationError:
        return {}


def _tags(value: object) -> frozenset[str]:
    try:
        parsed: Final = _VALUES.validate_json(value) if isinstance(value, str) else _VALUES.validate_python(value)
    except ValidationError:
        return frozenset()
    return frozenset(tag for tag in parsed if isinstance(tag, str))


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _start_time(value: object, fallback: datetime) -> datetime:
    try:
        return _TIMESTAMP.validate_python(value)
    except ValidationError:
        return fallback


def text_hash(request_text: str, context_text: str | None) -> str:
    return sha256(f"{request_text}\x1f{context_text or ''}".encode()).hexdigest()


def _clean(text: str) -> str:
    return text.replace("\x00", "").strip()


def _request_texts(delivery: UnobservedDelivery) -> tuple[str | None, str | None]:
    """The latest user text says what this request asks, and the first one says what the session is about. Agent
    histories replay every earlier turn, so only these two bounded excerpts are kept."""
    texts: Final = tuple(
        cleaned
        for cleaned in map(_clean, (*user_turn_texts(delivery.turns), *user_turn_texts(delivery.input_text)))
        if cleaned
    )
    if not texts:
        return None, None
    latest: Final = texts[-1][:MAX_REQUEST_TEXT_CHARS]
    first: Final = texts[0][:MAX_CONTEXT_TEXT_CHARS]
    return latest, first if len(texts) > 1 and first != latest else None


def _attributed_config(delivery: UnobservedDelivery, payload_model: str | None) -> str | None:
    """The request log names one deployment, the final attempt's. A delivery billed on another model, such as an
    earlier retry, keeps an unknown deployment rather than inheriting it."""
    if not delivery.model_config_id or payload_model is None or payload_model != delivery.logged_model:
        return None
    return delivery.model_config_id


def observe(delivery: UnobservedDelivery, observed_at: datetime) -> TaskObservation:
    """Every applied delivery yields exactly one observation, so a cleanup that waits for it never waits forever.

    A charge is kept whatever the outcome, since failed and retried attempts can be billed. Latency is kept for
    successful attempts only, so a timeout never reads as serving speed.
    """
    payload: Final = delivery.accounting_payload
    metadata: Final = _mapping(payload.get("metadata"))
    status: Final = _text(payload.get("status")) or _text(metadata.get("status"))
    model: Final = _text(payload.get("model"))
    excluded: Final = bool(metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)) or TASK_INSIGHTS_ANALYSIS_TAG in _tags(
        payload.get("request_tags")
    )
    request_text, context_text = (None, None) if excluded else _request_texts(delivery)
    latency: Final = _number(payload.get("request_duration_ms"))
    return TaskObservation(
        delivery_id=delivery.delivery_id,
        request_id=delivery.request_id,
        is_batch=delivery.is_batch,
        start_time=_start_time(payload.get("startTime"), observed_at),
        team_id=_text(payload.get("team_id")),
        model_group=_text(payload.get("model_group")),
        model=model,
        model_config_id=_attributed_config(delivery, model),
        session_id=delivery.session_id or None,
        spend=_number(payload.get("spend")),
        latency_ms=latency if status != "failure" and latency is not None and latency >= 0 else None,
        status=status,
        request_text=request_text,
        context_text=context_text,
        text_hash=text_hash(request_text, context_text) if request_text is not None else None,
        excluded=excluded,
    )


def task_insights_enabled() -> bool:
    return os.environ.get("LITELLM_TASK_INSIGHTS_ENABLED", "").strip().lower() in {"1", "true", "yes"}


class ObservationStore(Protocol):
    async def unobserved_deliveries(
        self, limit: int, request_id_prefix: str | None = None
    ) -> tuple[UnobservedDelivery, ...]: ...

    async def insert_observations(self, observations: tuple[TaskObservation, ...]) -> int: ...


async def observe_applied_deliveries(
    store: ObservationStore,
    *,
    now: Callable[[], datetime],
    batch_size: int,
    max_batches: int,
    request_id_prefix: str | None = None,
) -> int:
    """Project applied deliveries in bounded batches. Reads only the ledger and request log, never calls a model."""
    observed: int = 0  # rebind-ok: running total across bounded batches
    for _ in range(max_batches):
        deliveries = await store.unobserved_deliveries(batch_size, request_id_prefix)
        if not deliveries:
            break
        at = now()
        observed += await store.insert_observations(tuple(observe(delivery, at) for delivery in deliveries))
        if len(deliveries) < batch_size:
            break
    return observed
