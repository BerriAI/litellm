from collections.abc import Mapping
from typing import Final, Literal

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import Admit, KwargsFields, hook_subscribers, run_subscribers
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_helpers.native_store import PrometheusCompanion
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, Logging
from litellm.types.integrations.custom_logger_v2 import Outcome

Event = Literal["success", "failure"]


def enable_native_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)


def companion_of(logger: PrometheusLogger) -> PrometheusCompanion:
    companion: Final = logger.v2_companion
    assert isinstance(companion, PrometheusCompanion)
    return companion


async def _deliver(
    event: Event, logger: PrometheusLogger, kwargs: dict[str, object], response: object, start: object, end: object
) -> None:
    if event == "success":
        await logger.async_log_success_event(kwargs, response, start, end)
    else:
        await logger.async_log_failure_event(kwargs, response, start, end)


async def _run_line(
    event: Event,
    logger: PrometheusLogger,
    kwargs: dict[str, object],
    response: object,
    start_time: object,
    end_time: object,
    monkeypatch: pytest.MonkeyPatch | None,
) -> None:
    import litellm

    start: Final = kwargs.get("start_time") if start_time is None else start_time
    end: Final = kwargs.get("end_time") if end_time is None else end_time
    line: Final = (logger, companion_of(logger))
    patcher: Final = monkeypatch or pytest.MonkeyPatch()
    try:
        patcher.setattr(litellm, "_async_success_callback", list(line) if event == "success" else [])
        patcher.setattr(litellm, "_async_failure_callback", list(line) if event == "failure" else [])
        owner: Final = Logging(
            model=str(kwargs.get("model", "")),
            messages=[],
            stream=kwargs.get("stream") is True,
            call_type="acompletion",
            start_time=start,
            litellm_call_id=str(kwargs.get("litellm_call_id", "native")),
            function_id="native",
        )
        owner.model_call_details = kwargs
        args: Final = HookArgs(
            request=kwargs,
            result=response,
            start_time=start,
            end_time=end,
            outcome=Outcome.success if event == "success" else Outcome.failure,
            logging_obj=owner,
        )

        async def deliver(callback: object, current: object) -> None:
            if callback is logger and isinstance(current, dict):
                await _deliver(event, logger, current, response, start, end)

        _ = await run_subscribers(
            hook_subscribers(Hook.attempt_ended, line, admit=Admit.every_entry),
            with_sources(KwargsFields(kwargs, LOGGING_KEYS), DERIVED_ATTEMPT_FIELDS, args),
            deliver,
            call=owner.v2_call_facts(end),
        )
    finally:
        if monkeypatch is None:
            patcher.undo()


async def native_success(
    logger: PrometheusLogger,
    kwargs: dict[str, object],
    response_obj: object,
    start_time: object = None,
    end_time: object = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    await _run_line("success", logger, kwargs, response_obj, start_time, end_time, monkeypatch)


async def native_failure(
    logger: PrometheusLogger,
    kwargs: dict[str, object],
    response_obj: object,
    start_time: object = None,
    end_time: object = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    await _run_line("failure", logger, kwargs, response_obj, start_time, end_time, monkeypatch)


def _label_text(value: object) -> str:
    return str(value)


def series(name: str) -> list[tuple[dict[str, str], float]]:
    found: Final[list[tuple[dict[str, str], float]]] = []
    for family in REGISTRY.collect():
        if family.name != name and family.name + "_total" != name:
            continue
        main: Final = {"counter": family.name + "_total", "gauge": family.name, "histogram": family.name + "_sum"}
        for sample in family.samples:
            if sample.name == main.get(family.type, family.name):
                found.append((dict(sample.labels), sample.value))
    return found


def observations(name: str, labels: Mapping[str, object]) -> list[tuple[float, float]]:
    wanted: Final = {key: _label_text(value) for key, value in labels.items()}
    counts: Final[list[tuple[float, float]]] = []
    for family in REGISTRY.collect():
        if family.name != name:
            continue
        totals: Final = {
            tuple(sorted(sample.labels.items())): sample.value
            for sample in family.samples
            if sample.name == name + "_sum"
        }
        for sample in family.samples:
            if sample.name == name + "_count" and dict(sample.labels) == wanted:
                counts.append((sample.value, totals.get(tuple(sorted(sample.labels.items())), 0.0)))
    return counts


def value_of(name: str, labels: Mapping[str, object]) -> float | None:
    wanted: Final = {key: _label_text(value) for key, value in labels.items()}
    matches: Final = [value for found, value in series(name) if found == wanted]
    assert len(matches) <= 1
    return matches[0] if matches else None


def labelsets(name: str) -> list[dict[str, str]]:
    return [labels for labels, _ in series(name)]


def standard_payload(metadata: Mapping[str, object] | None = None, **fields: object) -> dict[str, object]:
    payload: Final[dict[str, object]] = {
        "id": "native-payload",
        "call_type": "completion",
        "status": "success",
        "response_cost": 0.0,
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20,
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 1.5,
        "model": "gpt-4",
        "model_id": "model_1",
        "model_group": "gpt-4",
        "api_base": "http://api.base",
        "custom_llm_provider": "openai",
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
        "stream": False,
        "response": {"id": "chatcmpl-1"},
        "model_parameters": {},
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
        "metadata": {
            "user_api_key_hash": None,
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_user_id": None,
            "user_api_key_user_email": None,
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "user_api_key_end_user_id": None,
            "requester_metadata": None,
            "usage_object": None,
        }
        | dict(metadata or {}),
    }
    payload.update(fields)
    return payload
