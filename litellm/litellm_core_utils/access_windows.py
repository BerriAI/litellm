from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Generic, TypeVar
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, ValidationError

from litellm.types.router import ModelAccessWindow

_WINDOWS_ADAPTER: Final = TypeAdapter(tuple[ModelAccessWindow, ...])

_DeploymentT = TypeVar("_DeploymentT", bound=Mapping[str, object])


def parse_access_windows(model_info: Mapping[str, object]) -> tuple[ModelAccessWindow, ...]:
    raw: Final = model_info.get("access_windows")
    if raw is None:
        return ()
    return _WINDOWS_ADAPTER.validate_python(raw)


def access_windows_config_error(model_info: Mapping[str, object], *, model_name: str) -> str | None:
    if model_info.get("access_windows") is None:
        return None
    try:
        parse_access_windows(model_info)
    except ValidationError as exc:
        first: Final = exc.errors()[0]
        loc: Final = ".".join(str(part) for part in first["loc"])
        return f"model '{model_name}': invalid model_info.access_windows: {loc}: {first['msg']}"
    return None


def is_window_active(window: ModelAccessWindow, now: datetime) -> bool:
    aware: Final = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    local: Final = aware.astimezone(ZoneInfo(window.timezone)).time()
    if window.start < window.end:
        return window.start <= local < window.end
    return local >= window.start or local < window.end


@dataclass(frozen=True, slots=True)
class ReservationFilterResult(Generic[_DeploymentT]):
    deployments: tuple[_DeploymentT, ...]
    blocking_window: ModelAccessWindow | None


def _reservation_blocking_window(
    deployment: Mapping[str, object], request_team_id: str | None, now: datetime
) -> ModelAccessWindow | None:
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return None
    active: Final = tuple(window for window in parse_access_windows(model_info) if is_window_active(window, now))
    if not active:
        return None
    if request_team_id is not None and any(request_team_id in window.team_ids for window in active):
        return None
    return active[0]


def filter_reserved_deployments(
    healthy_deployments: Sequence[_DeploymentT],
    request_team_id: str | None,
    now: datetime,
) -> ReservationFilterResult[_DeploymentT]:
    checks: Final = tuple(
        (deployment, _reservation_blocking_window(deployment, request_team_id, now))
        for deployment in healthy_deployments
    )
    return ReservationFilterResult(
        deployments=tuple(deployment for deployment, blocking in checks if blocking is None),
        blocking_window=next((blocking for _, blocking in checks if blocking is not None), None),
    )
