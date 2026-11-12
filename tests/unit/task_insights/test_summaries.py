from datetime import datetime, timezone

import pytest

from litellm.models.task_insights import (
    UNCLASSIFIED_TASK_ID,
    UNKNOWN_TASK_ID,
    MatrixCell,
    ObservedPoint,
    TaskDefinition,
    TaskVolume,
    WindowTotals,
)
from litellm.task_insights.summaries import (
    ModelMetadata,
    compatibility,
    map_points,
    model_configs,
    task_summaries,
    task_summary,
    totals,
)
from litellm.types.proxy.task_insights import ModelConfigSummary

_AT = datetime(2026, 11, 11, tzinfo=timezone.utc)


def _task(task_id: str, *, status: str = "frozen", capabilities=()) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        name=task_id.title(),
        description="",
        inclusion_criteria="",
        exclusion_criteria="",
        required_capabilities=capabilities,
        status=status,
        source="discovered",
        map_x=50,
        map_y=40,
        created_at=_AT,
    )


def _volume(task_id: str, requests: int, *, spend: float | None = None, cost: int = 0, sessions: int = 1) -> TaskVolume:
    return TaskVolume(
        task_id=task_id,
        request_count=requests,
        delivery_count=requests,
        session_count=sessions,
        spend=spend,
        cost_observations=cost,
    )


def _config(config_id: str, provider_model: str | None = "anthropic/model") -> ModelConfigSummary:
    return ModelConfigSummary(
        model_config_id=config_id, model_name=config_id, provider_model=provider_model, configured=True
    )


def _cell(config_id: str, requests: int) -> MatrixCell:
    return MatrixCell(
        model_config_id=config_id,
        task_id="a",
        request_count=requests,
        avg_cost_usd=None,
        cost_observations=0,
        p95_latency_ms=None,
        latency_observations=0,
    )


def test_frozen_tasks_rank_by_volume_and_the_buckets_follow():
    definitions = (_task("a"), _task("b"), _task("draft", status="draft"))
    volumes = (_volume("a", 2), _volume("b", 6, spend=3.0, cost=3), _volume(UNKNOWN_TASK_ID, 2))

    summaries = task_summaries(definitions, volumes)

    assert [(item.task_id, item.kind, item.share) for item in summaries] == [
        ("b", "task", 0.6),
        ("a", "task", 0.2),
        (UNKNOWN_TASK_ID, "unknown", 0.2),
    ]
    assert summaries[0].avg_cost_usd == 1.0
    assert summaries[1].avg_cost_usd is None


def test_a_task_without_traffic_still_summarizes_and_unknown_ids_get_names():
    assert task_summary("a", (_task("a"),), ()).request_count == 0
    assert task_summary(UNCLASSIFIED_TASK_ID, (), ()).name == "Not classified yet"


def test_totals_take_sessions_from_the_window_rather_than_summing_tasks():
    volumes = (_volume("a", 3, sessions=2), _volume("b", 2, sessions=2), _volume(UNCLASSIFIED_TASK_ID, 1))
    window = WindowTotals(
        request_count=6, delivery_count=8, session_count=2, analysis_request_count=4, analysis_spend=0.1
    )

    result = totals(volumes, window)

    assert (result.session_count, result.delivery_count, result.classified_count, result.unclassified_count) == (
        2,
        8,
        5,
        1,
    )
    assert result.spend is None


def test_known_spend_sums_only_what_was_recorded():
    window = WindowTotals(
        request_count=2, delivery_count=2, session_count=1, analysis_request_count=0, analysis_spend=None
    )

    assert totals((_volume("a", 1, spend=0.5, cost=1), _volume("b", 1)), window).spend == 0.5


def test_points_from_discovery_keep_their_position_and_later_requests_orbit_their_task():
    points = (
        ObservedPoint(request_id="seen", task_id="a", x=10, y=20),
        ObservedPoint(request_id="new", task_id="a", x=None, y=None),
        ObservedPoint(request_id="lost", task_id=UNKNOWN_TASK_ID, x=None, y=None),
    )

    placed = map_points(points, (_task("a"),))
    again = map_points(points, (_task("a"),))

    assert [(point.request_id, point.placement) for point in placed] == [("seen", "embedding"), ("new", "categorical")]
    assert (placed[0].x, placed[0].y) == (10, 20)
    assert placed[1] == again[1]
    assert abs(placed[1].x - 50) <= 4 and abs(placed[1].y - 40) <= 4


def _probe(metadata: dict[str, ModelMetadata]):
    return lambda provider_model: metadata.get(provider_model, ModelMetadata(capabilities={}, max_input_tokens=None))


def test_configured_chat_models_without_traffic_stay_as_columns_and_embedding_models_do_not():
    probe = _probe({"chat": ModelMetadata({}, 200_000, "chat"), "embed": ModelMetadata({}, 8_000, "embedding")})
    configured = (_config("idle", "chat"), _config("busy", "chat"), _config("embedder", "embed"))

    configs = model_configs(configured, (_cell("busy", 4), _cell("removed", 1)), probe)

    assert [(config.model_config_id, config.configured) for config in configs] == [
        ("busy", True),
        ("idle", True),
        ("removed", False),
    ]
    assert configs[0].max_input_tokens == 200_000


@pytest.mark.parametrize(
    ("supported", "status", "missing"),
    [
        ({"function_calling": True}, "compatible", ()),
        ({"function_calling": False}, "incompatible", ("function_calling",)),
        ({}, "unknown", ()),
    ],
)
def test_compatibility_comes_from_metadata_and_the_task_requirements(supported, status, missing):
    probe = _probe({"m": ModelMetadata(supported, None, "chat")})

    (result,) = compatibility((_task("a", capabilities=("function_calling",)),), (_config("c", "m"),), probe)

    assert (result.status, result.missing) == (status, missing)


def test_a_task_without_requirements_is_compatible_with_any_model():
    (result,) = compatibility((_task("a"),), (_config("c", None),), _probe({}))

    assert result.status == "compatible"
