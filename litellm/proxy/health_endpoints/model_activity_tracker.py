import time
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _extract_nested_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    val: Final = parent.get(key)
    if not isinstance(val, Mapping):
        return None
    try:
        return _MAPPING_ADAPTER.validate_python(val)
    except Exception:
        return None


def _extract_str_field(mapping: Mapping[str, object] | None, key: str) -> str | None:
    if mapping is None:
        return None
    val: Final = mapping.get(key)
    return str(val) if val is not None else None


class ModelActivityTracker:
    def __init__(self) -> None:
        self._last_activity_by_id: Final[dict[str, float]] = {}  # mutable-ok: internal activity timestamp store
        self._last_activity_by_name: Final[dict[str, float]] = {}  # mutable-ok: internal activity timestamp store

    def record_activity(
        self,
        *,
        model_id: str | None = None,
        model_name: str | None = None,
        litellm_model: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        now: Final = timestamp if timestamp is not None else time.time()
        if model_id:
            self._last_activity_by_id[str(model_id)] = now
        if model_name:
            self._last_activity_by_name[str(model_name)] = now
        if litellm_model:
            self._last_activity_by_name[str(litellm_model)] = now

    def get_last_activity(
        self,
        *,
        model_id: str | None = None,
        model_name: str | None = None,
        litellm_model: str | None = None,
    ) -> float | None:
        timestamps: Final = tuple(
            ts
            for ts in (
                self._last_activity_by_id.get(str(model_id)) if model_id else None,
                self._last_activity_by_name.get(str(model_name)) if model_name else None,
                self._last_activity_by_name.get(str(litellm_model)) if litellm_model else None,
            )
            if ts is not None
        )
        return max(timestamps) if timestamps else None

    def should_skip_health_probe(
        self,
        model: Mapping[str, object],
        health_check_interval: float | None,
        *,
        current_time: float | None = None,
    ) -> bool:
        if health_check_interval is None or health_check_interval <= 0:
            return False

        window: Final = min(health_check_interval * 0.2, 60.0)
        if window <= 0:
            return False

        model_info_map: Final = _extract_nested_mapping(model, "model_info")
        model_id: Final = _extract_str_field(model_info_map, "id")
        model_name: Final = _extract_str_field(model, "model_name")
        litellm_params_map: Final = _extract_nested_mapping(model, "litellm_params")
        litellm_model: Final = _extract_str_field(litellm_params_map, "model")

        last_activity: Final = self.get_last_activity(
            model_id=model_id,
            model_name=model_name,
            litellm_model=litellm_model,
        )
        if last_activity is None:
            return False

        now: Final = current_time if current_time is not None else time.time()
        elapsed: Final = now - last_activity
        return 0.0 <= elapsed <= window

    def reset(self) -> None:
        self._last_activity_by_id.clear()
        self._last_activity_by_name.clear()


model_activity_tracker: Final = ModelActivityTracker()


def record_model_activity(
    *,
    model_id: str | None = None,
    model_name: str | None = None,
    litellm_model: str | None = None,
    timestamp: float | None = None,
) -> None:
    model_activity_tracker.record_activity(
        model_id=model_id,
        model_name=model_name,
        litellm_model=litellm_model,
        timestamp=timestamp,
    )


def get_model_last_activity(
    *,
    model_id: str | None = None,
    model_name: str | None = None,
    litellm_model: str | None = None,
) -> float | None:
    return model_activity_tracker.get_last_activity(
        model_id=model_id,
        model_name=model_name,
        litellm_model=litellm_model,
    )


def should_skip_model_health_probe(
    model: Mapping[str, object],
    health_check_interval: float | None,
    *,
    current_time: float | None = None,
) -> bool:
    return model_activity_tracker.should_skip_health_probe(
        model,
        health_check_interval,
        current_time=current_time,
    )


def reset_model_activity_tracker() -> None:
    model_activity_tracker.reset()
