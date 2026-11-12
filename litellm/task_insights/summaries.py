import math
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Final, NamedTuple, TypeAlias

from litellm.models.task_insights import (
    UNCLASSIFIED_TASK_ID,
    UNKNOWN_TASK_ID,
    MatrixCell,
    ObservedPoint,
    TaskCapability,
    TaskDefinition,
    TaskVolume,
    WindowTotals,
)
from litellm.types.proxy.task_insights import (
    ModelConfigSummary,
    TaskCompatibility,
    TaskInsightsTotals,
    TaskKind,
    TaskMapPoint,
    TaskMatrixEntry,
    TaskRef,
    TaskSummary,
)


class ModelMetadata(NamedTuple):
    capabilities: Mapping[TaskCapability, bool | None]
    max_input_tokens: int | None
    mode: str | None = None


CapabilityProbe: TypeAlias = Callable[[str], ModelMetadata]

_NO_METADATA: Final = ModelMetadata(capabilities={}, max_input_tokens=None)
_TASK_SERVING_MODES: Final = frozenset({None, "chat", "responses", "completion"})
_JITTER_RADIUS: Final = 4.0
_PSEUDO_TASKS: Final = (
    (UNKNOWN_TASK_ID, "Unknown", "Requests that fit none of the frozen task definitions"),
    (UNCLASSIFIED_TASK_ID, "Not classified yet", "Requests without stored prompt text or not yet bucketed"),
)


def _kind(task_id: str) -> TaskKind:
    if task_id == UNKNOWN_TASK_ID:
        return "unknown"
    return "unclassified" if task_id == UNCLASSIFIED_TASK_ID else "task"


def _empty_volume(task_id: str) -> TaskVolume:
    return TaskVolume(
        task_id=task_id, request_count=0, delivery_count=0, session_count=0, spend=None, cost_observations=0
    )


def task_summaries(definitions: tuple[TaskDefinition, ...], volumes: tuple[TaskVolume, ...]) -> tuple[TaskSummary, ...]:
    """Frozen tasks ranked by request volume, followed by the unknown and not-classified buckets when they hold
    requests."""
    by_task: Final = {volume.task_id: volume for volume in volumes}
    total: Final = sum(volume.request_count for volume in volumes)
    tasks: Final = tuple(
        _summary(task.task_id, task.name, task.description, by_task.get(task.task_id), total, task)
        for task in definitions
        if task.status == "frozen"
    )
    pseudo: Final = tuple(
        _summary(task_id, name, description, by_task[task_id], total, None)
        for task_id, name, description in _PSEUDO_TASKS
        if task_id in by_task
    )
    return (*sorted(tasks, key=lambda item: (-item.request_count, item.name)), *pseudo)


def task_summary(task_id: str, definitions: tuple[TaskDefinition, ...], volumes: tuple[TaskVolume, ...]) -> TaskSummary:
    """One task or bucket, present even when the window holds no requests for it."""
    listed: Final = next((item for item in task_summaries(definitions, volumes) if item.task_id == task_id), None)
    if listed is not None:
        return listed
    pseudo: Final = next((item for item in _PSEUDO_TASKS if item[0] == task_id), (task_id, task_id, ""))
    return _summary(task_id, pseudo[1], pseudo[2], None, 0, None)


def _summary(
    task_id: str,
    name: str,
    description: str,
    volume: TaskVolume | None,
    total: int,
    definition: TaskDefinition | None,
) -> TaskSummary:
    counted: Final = volume or _empty_volume(task_id)
    return TaskSummary(
        task_id=task_id,
        name=name,
        description=description,
        kind=_kind(task_id),
        request_count=counted.request_count,
        delivery_count=counted.delivery_count,
        session_count=counted.session_count,
        share=counted.request_count / total if total else 0.0,
        spend=counted.spend,
        cost_observations=counted.cost_observations,
        avg_cost_usd=counted.spend / counted.cost_observations
        if counted.spend is not None and counted.cost_observations
        else None,
        map_x=definition.map_x if definition is not None else None,
        map_y=definition.map_y if definition is not None else None,
    )


def _known_sum(values: tuple[float | None, ...]) -> float | None:
    known: Final = tuple(value for value in values if value is not None)
    return sum(known) if known else None


def totals(volumes: tuple[TaskVolume, ...], window: WindowTotals) -> TaskInsightsTotals:
    def requests(task_id: str) -> int:
        return sum(volume.request_count for volume in volumes if volume.task_id == task_id)

    return TaskInsightsTotals(
        request_count=window.request_count,
        delivery_count=window.delivery_count,
        session_count=window.session_count,
        spend=_known_sum(tuple(volume.spend for volume in volumes)),
        cost_observations=sum(volume.cost_observations for volume in volumes),
        unattributed_count=sum(volume.unattributed_count for volume in volumes),
        classified_count=sum(volume.request_count for volume in volumes if _kind(volume.task_id) == "task"),
        unknown_count=requests(UNKNOWN_TASK_ID),
        unclassified_count=requests(UNCLASSIFIED_TASK_ID),
        analysis_request_count=window.analysis_request_count,
        analysis_spend=window.analysis_spend,
    )


def _jitter(request_id: str) -> tuple[float, float]:
    digest: Final = sha256(request_id.encode()).digest()
    angle: Final = digest[0] / 255 * 2 * math.pi
    radius: Final = math.sqrt(digest[1] / 255) * _JITTER_RADIUS
    return radius * math.cos(angle), radius * math.sin(angle)


def _placed(point: ObservedPoint, centroids: Mapping[str, tuple[float, float]]) -> TaskMapPoint | None:
    if point.x is not None and point.y is not None:
        return TaskMapPoint(
            request_id=point.request_id, task_id=point.task_id, x=point.x, y=point.y, placement="embedding"
        )
    centroid: Final = centroids.get(point.task_id)
    if centroid is None:
        return None
    dx, dy = _jitter(point.request_id)
    return TaskMapPoint(
        request_id=point.request_id,
        task_id=point.task_id,
        x=centroid[0] + dx,
        y=centroid[1] + dy,
        placement="categorical",
    )


def map_points(points: tuple[ObservedPoint, ...], definitions: tuple[TaskDefinition, ...]) -> tuple[TaskMapPoint, ...]:
    """A request bucketed after discovery has no embedding, so it is drawn at a stable offset around its task, keeping
    the layout fixed while counts grow."""
    centroids: Final = {task.task_id: (task.map_x, task.map_y) for task in definitions if task.status == "frozen"}
    return tuple(placed for placed in (_placed(point, centroids) for point in points) if placed is not None)


def matrix_entries(cells: tuple[MatrixCell, ...]) -> tuple[TaskMatrixEntry, ...]:
    return tuple(TaskMatrixEntry.model_validate(cell.model_dump()) for cell in cells)


def task_refs(definitions: tuple[TaskDefinition, ...], volumes: tuple[TaskVolume, ...]) -> tuple[TaskRef, ...]:
    return tuple(
        TaskRef(
            task_id=summary.task_id,
            name=summary.name,
            kind=summary.kind,
            required_capabilities=next(
                (task.required_capabilities for task in definitions if task.task_id == summary.task_id), ()
            ),
        )
        for summary in task_summaries(definitions, volumes)
    )


def model_configs(
    configured: tuple[ModelConfigSummary, ...], cells: tuple[MatrixCell, ...], probe: CapabilityProbe
) -> tuple[ModelConfigSummary, ...]:
    """Every configured deployment that can serve a task, busiest first, so one without traffic still shows as an
    empty column, then the deployments the traffic names that are no longer configured. Embedding and other non-chat
    deployments only appear once they serve traffic."""
    known: Final = frozenset(config.model_config_id for config in configured)
    gone: Final = tuple(
        ModelConfigSummary(model_config_id=config_id, model_name=None, provider_model=None, configured=False)
        for config_id in sorted(frozenset(cell.model_config_id for cell in cells) - known)
    )

    def volume(config: ModelConfigSummary) -> int:
        return sum(cell.request_count for cell in cells if cell.model_config_id == config.model_config_id)

    candidates: Final = tuple(
        config for config in configured if volume(config) or _metadata(config, probe).mode in _TASK_SERVING_MODES
    )
    ranked: Final = sorted(
        candidates, key=lambda config: (-volume(config), config.model_name or "", config.model_config_id)
    )
    return tuple(
        config.model_copy(update={"max_input_tokens": _metadata(config, probe).max_input_tokens})
        for config in (*ranked, *gone)
    )


def _metadata(config: ModelConfigSummary, probe: CapabilityProbe) -> ModelMetadata:
    return probe(config.provider_model) if config.provider_model else _NO_METADATA


def model_capabilities(provider_model: str) -> ModelMetadata:
    import litellm

    try:
        info: Final = litellm.get_model_info(provider_model)
    except Exception:  # noqa: BLE001  # a model missing from the cost map has no metadata to judge it by
        return _NO_METADATA
    max_input: Final = info.get("max_input_tokens")
    return ModelMetadata(
        capabilities={
            "function_calling": info.get("supports_function_calling"),
            "vision": info.get("supports_vision"),
            "reasoning": info.get("supports_reasoning"),
            "structured_output": info.get("supports_response_schema"),
        },
        max_input_tokens=max_input if isinstance(max_input, int) else None,
        mode=info.get("mode"),
    )


def compatibility(
    definitions: tuple[TaskDefinition, ...],
    configs: tuple[ModelConfigSummary, ...],
    probe: CapabilityProbe,
) -> tuple[TaskCompatibility, ...]:
    """Derived from current model metadata on every read, so a model or task change never leaves a stale flag. It
    only covers what metadata can say, a router still has to check each request's actual constraints."""
    capabilities: Final = {config.model_config_id: _metadata(config, probe).capabilities for config in configs}
    frozen: Final = tuple(task for task in definitions if task.status == "frozen")
    return tuple(
        _compatibility(config.model_config_id, task, capabilities[config.model_config_id])
        for config in configs
        for task in frozen  # comprehension-ok: every deployment against every task is the matrix itself
    )


def _compatibility(
    model_config_id: str, task: TaskDefinition, supported: Mapping[TaskCapability, bool | None]
) -> TaskCompatibility:
    missing: Final = tuple(cap for cap in task.required_capabilities if supported.get(cap) is False)
    unknown: Final = any(supported.get(cap) is None for cap in task.required_capabilities)
    status: Final = "incompatible" if missing else ("unknown" if unknown else "compatible")
    return TaskCompatibility(model_config_id=model_config_id, task_id=task.task_id, status=status, missing=missing)
