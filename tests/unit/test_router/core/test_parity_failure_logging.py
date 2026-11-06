"""
Parity of the Rust core with ``router_python_core.py`` on what a failed request leaves behind: the
failure dispatched to the request's logging object (``_async_get_available_deployment``'s ``except``)
and the walk stamped on the caller's own metadata dict, which a proxy's failure SpendLogs row reads
(``async_function_with_fallbacks``, ``async_function_with_retries``). Also the decision headers of a
routed response, typed ``model_info`` read back, ``routing_strategy_args`` as a plain dict and the
``cooldown_list`` of a no-deployments error.
"""

import asyncio
from collections.abc import Mapping
from datetime import datetime, time, timezone
from typing import Final

import pytest
from pydantic_core import to_jsonable_python

import litellm
from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _get_spend_logs_metadata,  # pyright: ignore[reportPrivateUsage]  # what a failure row keeps
)
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import AllowedFailsPolicy, RouterRateLimitError
from tests.unit.test_router.core.fallbacks_helpers import (
    MESSAGES,
    MakeCore,
    complete,
    complete_sync,
    headers,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit, server_error

GROUP_OF_TWO: Final = [deployment("a", "a1"), deployment("a", "a2")]
UTC_2030: Final = datetime(2030, 1, 1, tzinfo=timezone.utc)


class _RecordingLoggingObject:
    """Stands in for a request's ``litellm_logging_obj``: records the failure dispatches."""

    def __init__(self) -> None:
        self.dispatched: list[dict[str, object]] = []  # mutable-ok: the dispatches, in order
        self.messages: list[object] = []  # mutable-ok: each error's message when the dispatch ran

    async def dispatch_failure_handlers(self, **kwargs: object) -> None:  # kwargs-ok: the dispatch arguments
        self.dispatched.append(kwargs)
        self.messages.append(getattr(kwargs.get("exception"), "message", None))


class _RecordingLogging(Logging):
    """A real logging object whose failure handlers only record that they ran."""

    def __init__(self, call_type: str) -> None:
        super().__init__(
            model="e",
            messages=[],
            stream=False,
            call_type=call_type,
            start_time=datetime.now(tz=timezone.utc),
            litellm_call_id="c",
            function_id="f",
        )
        self.update_environment_variables(model="e", user=None, optional_params={}, litellm_params={})
        self.failures: list[tuple[str, object]] = []  # mutable-ok: the handlers that ran, in order

    def failure_handler(self, exception: Exception, traceback_exception: str, *_: object, **__: object) -> None:
        self.failures.append(("sync", exception))

    async def async_failure_handler(
        self, exception: Exception, traceback_exception: str, *_: object, **__: object
    ) -> None:
        self.failures.append(("async", exception))


async def _let_tasks_run() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def _failed(
    core: RouterCore, model: str, *, expected_error: type[Exception], **request: object
) -> Exception:  # kwargs-ok: request kwargs
    with pytest.raises(expected_error) as raised:
        await complete(core, model, **request)
    await _let_tasks_run()
    return raised.value


# --- router-raised errors -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_model_is_dispatched_once_with_the_error_raised(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    logging_obj: Final = _RecordingLoggingObject()

    error: Final = await _failed(
        core, "nope", expected_error=litellm.BadRequestError, litellm_logging_obj=logging_obj, metadata={}
    )

    assert isinstance(error, litellm.BadRequestError)
    (dispatched,) = logging_obj.dispatched
    assert dispatched["exception"] is error
    assert dispatched["prefer_async_handlers"] is True
    assert isinstance(dispatched["traceback_exception"], str)


@pytest.mark.asyncio
async def test_a_refused_pick_is_logged_with_the_note_when_the_retry_read_does_not_yield(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The retry handler reads the cooldowns even for a name with no group. Without Redis that read never
    hands the loop to the failure log, so the no-fallback note is on the error before it is logged."""
    show_router_debug(host)
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0, fallbacks=[{"a": ["a"]}])
    logging_obj: Final = _RecordingLoggingObject()

    error: Final = await _failed(core, "nope", expected_error=litellm.BadRequestError, litellm_logging_obj=logging_obj)

    note: Final = "no fallback model group was found"
    assert note in str(error)
    (logged,) = logging_obj.messages
    assert note in str(logged)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "handlers"),
    [("aembedding", ["sync", "async"]), ("embedding", ["sync"])],
    ids=["async", "sync"],
)
async def test_an_oversized_embedding_is_stamped_logged_and_counted_as_its_deployments_failure(
    make_core: MakeCore, host: ScriptedHost, endpoint: str, handlers: list[str]
) -> None:
    core: Final = make_core(
        model_list=[deployment("e", "e1", model_info={"max_embedding_batch_size": 1})], num_retries=0
    )
    logging_obj: Final = _RecordingLogging(endpoint)

    with pytest.raises(litellm.BadRequestError) as raised:
        await core.call(endpoint, {"model": "e", "input": ["a", "b"], "litellm_logging_obj": logging_obj})

    error: Final = raised.value
    assert getattr(error, "failed_deployment_id", None) == "e1"
    assert getattr(error, "retry_skip_deployment_id", None) == "e1"
    assert [(kind, exception is error) for kind, exception in logging_obj.failures] == [(k, True) for k in handlers]
    assert logging_obj.model_call_details["litellm_params"][PROXY_REJECTED_BEFORE_ROUTING_KEY] is True
    assert host.calls == []
    assert dict(core.fail_calls) == {"openai/e1": 1}
    assert dict(core.total_calls) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["aembedding", "embedding"])
async def test_an_admitted_embedding_drops_the_rejection_an_earlier_attempt_left(
    make_core: MakeCore, endpoint: str
) -> None:
    core: Final = make_core(
        model_list=[deployment("e", "e1", model_info={"max_embedding_batch_size": 2})], num_retries=0
    )
    logging_obj: Final = _RecordingLogging(endpoint)
    logging_obj.model_call_details["litellm_params"][PROXY_REJECTED_BEFORE_ROUTING_KEY] = True

    await core.call(endpoint, {"model": "e", "input": ["a", "b"], "litellm_logging_obj": logging_obj})

    assert PROXY_REJECTED_BEFORE_ROUTING_KEY not in logging_obj.model_call_details["litellm_params"]


@pytest.mark.asyncio
async def test_an_unknown_model_stamps_the_walk_on_the_callers_metadata(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    metadata: Final[dict[str, object]] = {"user_api_key_alias": "k"}  # mutable-ok: the caller's own dict

    _ = await _failed(core, "nope", expected_error=litellm.BadRequestError, metadata=metadata)

    assert metadata["attempted_retries"] == 0
    assert metadata["max_retries"] == 0
    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "nope"
    assert metadata["model_group"] == "nope"
    assert metadata["model_group_size"] == 0
    assert metadata["user_api_key_alias"] == "k"


@pytest.mark.asyncio
async def test_a_refused_pick_leaves_every_field_of_the_failure_row(make_core: MakeCore) -> None:
    """A real logging object, read the way ``get_logging_payload`` reads it: the walk from the
    request's metadata, the rest from the payload the dispatched failure built."""
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict
    logging_obj: Final = Logging(
        model="nope",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),  # noqa: DTZ005  # SDK logging uses naive local timestamps
        litellm_call_id="call",
        function_id="fn",
        applied_guardrails=["g1"],
    )
    logging_obj.update_environment_variables(
        model="nope", user="", optional_params={}, litellm_params={"metadata": metadata}
    )

    _ = await _failed(
        core, "nope", expected_error=litellm.BadRequestError, litellm_logging_obj=logging_obj, metadata=metadata
    )

    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    assert payload["status"] == "failure"
    row: Final = _get_spend_logs_metadata(
        metadata,
        applied_guardrails=payload["metadata"].get("applied_guardrails"),
        usage_object=payload["metadata"].get("usage_object"),
        model_map_information=payload["model_map_information"],
    )
    assert row["attempted_retries"] == 0
    assert row["max_retries"] == 0
    assert row["attempted_fallbacks"] == 0
    assert row["original_model_group"] == "nope"
    assert row["applied_guardrails"] == ["g1"]
    assert row["usage_object"] == payload["metadata"]["usage_object"]
    assert row["model_map_information"] == payload["model_map_information"]


@pytest.mark.asyncio
async def test_all_cooled_deployments_are_dispatched_with_their_cooldown_details(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("a", "a1", api_base="https://a1.example.com"),
            deployment("a", "a2", api_base="https://a2.example.com"),
        ],
        num_retries=0,
        cooldown_time=60,
    )
    for deployment_id in ("a1", "a2"):
        core.health().record_failure(deployment_id, rate_limit(), 429)
    logging_obj: Final = _RecordingLoggingObject()
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    error: Final = await _failed(
        core, "a", expected_error=RouterRateLimitError, litellm_logging_obj=logging_obj, metadata=metadata
    )

    assert isinstance(error, RouterRateLimitError)
    assert error.cooldown_list == ["a1", "a2"]
    assert error.cooldown_deployments == (
        {"model_id": "a1", "api_base": "https://a1.example.com"},
        {"model_id": "a2", "api_base": "https://a2.example.com"},
    )
    (dispatched,) = logging_obj.dispatched
    assert dispatched["exception"] is error
    assert metadata["model_group_size"] == 2


@pytest.mark.asyncio
async def test_a_refused_pick_with_retries_settles_the_retry_budget(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=3)
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    error: Final = await _failed(core, "nope", expected_error=litellm.BadRequestError, metadata=metadata)

    assert isinstance(error, litellm.BadRequestError)
    # A 400 is not retried: `should_retry_this_error` raises before the budget is written again.
    assert metadata["attempted_retries"] == 0
    assert metadata["max_retries"] == 3


@pytest.mark.asyncio
async def test_a_strategy_with_no_async_selector_does_not_dispatch(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0, routing_strategy="usage-based-routing")
    logging_obj: Final = _RecordingLoggingObject()

    _ = await _failed(
        core, "nope", expected_error=litellm.BadRequestError, litellm_logging_obj=logging_obj, metadata={}
    )

    assert logging_obj.dispatched == []


def test_the_sync_path_stamps_the_walk_and_does_not_dispatch(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    logging_obj: Final = _RecordingLoggingObject()
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    with pytest.raises(litellm.BadRequestError):
        complete_sync(core, "nope", litellm_logging_obj=logging_obj, metadata=metadata)

    assert logging_obj.dispatched == []
    assert metadata["attempted_retries"] == 0
    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "nope"


@pytest.mark.asyncio
async def test_a_refused_image_pick_with_no_logging_object_builds_one_to_dispatch(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: Final[list[BaseException]] = []  # mutable-ok: the dispatches, in order

    async def record(
        self: Logging, exception: Exception, **_arguments: object
    ) -> None:  # kwargs-ok: dispatch arguments
        dispatched.append(exception)

    monkeypatch.setattr(Logging, "dispatch_failure_handlers", record)
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)

    with pytest.raises(litellm.BadRequestError) as raised:
        await core.aimage_generation(prompt="a cat", model="nope")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    await _let_tasks_run()

    assert dispatched == [raised.value]


@pytest.mark.asyncio
async def test_a_refused_pre_call_check_is_dispatched_on_chat_only(make_core: MakeCore, host: ScriptedHost) -> None:
    class Gate(CustomLogger):
        async def async_pre_call_check(self, deployment: dict, parent_otel_span: object) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride]
            raise rate_limit(message="gate says no")

    host.lists["callbacks"].append(Gate())
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    logging_obj: Final = _RecordingLoggingObject()

    error: Final = await _failed(
        core, "a", expected_error=litellm.RateLimitError, litellm_logging_obj=logging_obj, metadata={}
    )

    assert isinstance(error, litellm.RateLimitError)
    (dispatched,) = logging_obj.dispatched
    assert dispatched["exception"] is error

    embedding_logging: Final = _RecordingLoggingObject()
    with pytest.raises(litellm.RateLimitError):
        await core.aembedding(model="a", input="hi", litellm_logging_obj=embedding_logging)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    await _let_tasks_run()
    assert embedding_logging.dispatched == []


# --- provider failures ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_failure_stamps_each_field_and_the_router_dispatches_nothing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", server_error())
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=1)
    logging_obj: Final = _RecordingLoggingObject()
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    error: Final = await _failed(
        core, "a", expected_error=litellm.InternalServerError, litellm_logging_obj=logging_obj, metadata=metadata
    )

    assert isinstance(error, litellm.InternalServerError)
    assert metadata["attempted_retries"] == 1
    assert metadata["max_retries"] == 1
    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "a"
    assert metadata["model_group_size"] == 2
    assert metadata["request_retry_count"] == 2
    breadcrumbs: Final = metadata["previous_models"]
    assert isinstance(breadcrumbs, tuple)
    assert len(breadcrumbs) == 2
    # The SDK's own wrapper logs a provider failure, on the same logging object.
    assert logging_obj.dispatched == []


@pytest.mark.asyncio
async def test_a_deployments_retry_budget_replaces_the_routers_in_the_stamp(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", server_error())
    core: Final = make_core(
        model_list=[deployment("a", "a1", num_retries=0), deployment("a", "a2", num_retries=0)], num_retries=2
    )
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    _ = await _failed(core, "a", expected_error=litellm.InternalServerError, metadata=metadata)

    assert metadata["max_retries"] == 0
    assert metadata["attempted_retries"] == 0


@pytest.mark.asyncio
async def test_a_fallback_hop_works_on_its_own_copy_of_the_metadata(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("*", server_error())
    core: Final = make_core(
        model_list=[deployment("a", "a1"), deployment("b", "b1")], num_retries=0, fallbacks=[{"a": ["b"]}]
    )
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    _ = await _failed(core, "a", expected_error=litellm.InternalServerError, metadata=metadata)

    assert metadata["attempted_fallbacks"] == 0
    assert metadata["attempted_retries"] == 0
    assert metadata["max_retries"] == 0
    assert metadata["original_model_group"] == "a"
    assert metadata["model_group"] == "a"
    hop: Final = host.calls_to("acompletion")[1].kwargs["metadata"]
    assert isinstance(hop, Mapping)
    assert hop["attempted_fallbacks"] == 1
    assert hop["original_model_group"] == "a"
    assert hop["model_group"] == "b"


@pytest.mark.asyncio
async def test_the_fallbacks_mock_raises_before_the_retry_loop_stamps_anything(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=2)
    logging_obj: Final = _RecordingLoggingObject()
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    _ = await _failed(
        core,
        "a",
        expected_error=litellm.InternalServerError,
        mock_testing_fallbacks=True,
        litellm_logging_obj=logging_obj,
        metadata=metadata,
    )

    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "a"
    assert "attempted_retries" not in metadata
    assert "max_retries" not in metadata
    assert logging_obj.dispatched == []


@pytest.mark.asyncio
async def test_the_rate_limit_mock_raises_inside_the_retry_loop(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    logging_obj: Final = _RecordingLoggingObject()
    metadata: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    _ = await _failed(
        core,
        "a",
        expected_error=litellm.RateLimitError,
        mock_testing_rate_limit_error=True,
        litellm_logging_obj=logging_obj,
        metadata=metadata,
    )

    assert metadata["attempted_retries"] == 0
    assert metadata["max_retries"] == 0
    assert metadata["model_group_size"] == 2
    assert logging_obj.dispatched == []


@pytest.mark.asyncio
async def test_a_generic_endpoint_stamps_its_bucket_and_clears_the_siblings_walk(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("*", server_error())
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    sibling: Final[dict[str, object]] = {  # mutable-ok: the caller's own dict
        "attempted_fallbacks": 7,
        "original_model_group": "stale",
        "kept": True,
    }
    bucket: Final[dict[str, object]] = {}  # mutable-ok: the caller's own dict

    with pytest.raises(litellm.InternalServerError):
        await core.aresponses(model="a", input="hi", metadata=sibling, litellm_metadata=bucket)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint

    assert "attempted_fallbacks" not in sibling
    assert "original_model_group" not in sibling
    assert sibling["kept"] is True
    assert bucket["attempted_fallbacks"] == 0
    assert bucket["original_model_group"] == "a"
    assert bucket["attempted_retries"] == 0
    assert bucket["max_retries"] == 0


# --- R7: the decision headers ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stamped_quality_decision_reaches_the_response_headers(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), num_retries=0)
    decision: Final = {"routed_model": "t1", "quality_tier": 2, "routed_via": "default", "matched_keyword": None}

    response: Final = await complete(core, "a", metadata={"quality_router_decision": decision})

    found: Final = headers(response)
    assert found["x-litellm-quality-router-model"] == "t1"
    assert found["x-litellm-quality-router-tier"] == "2"
    assert found["x-litellm-quality-router-via"] == "default"
    assert "x-litellm-quality-router-keyword" not in found


@pytest.mark.asyncio
async def test_a_quality_router_stamps_its_decision_headers(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment(
                "t1",
                "t1-21",
                model="anthropic/claude-3-5-sonnet-20240620",
                model_info={"litellm_routing_preferences": {"quality_tier": 1}},
            ),
            deployment("qr", "qr0", model="auto_router/quality_router", quality_router_default_model="t1"),
        ],
        num_retries=0,
    )

    response: Final = await complete(core, "qr")

    assert headers(response)["x-litellm-quality-router-model"] == "t1"


# --- P9, N9: typed model_info ---------------------------------------------------------------------


def _typed_model_list() -> list[dict[str, object]]:  # mutable-ok: a model_list
    return [
        deployment(
            "p",
            "p1",
            model="azure/gpt-4o",
            api_base="https://example.openai.azure.com",
            model_info={
                "ptu_shares": {"a": 2},
                "ptu_count": 2,
                "cost_per_ptu_per_hour": 2,
                "ptu_effective_from": "2026-01-01T00:00:00Z",
            },
        ),
        deployment(
            "h",
            "w3",
            model_info={
                "access_windows": [
                    {"start": "03:00:00", "end": "03:00:01", "timezone": "Asia/Tokyo", "team_ids": ["t1"]}
                ]
            },
        ),
        deployment("d", "db1", model_info={"created_at": UTC_2030, "updated_at": UTC_2030, "db_model": True}),
    ]


def _model_info(core: RouterCore, deployment_id: str) -> Mapping[str, object]:
    for entry in core.registry().all():
        info = entry["model_info"]
        assert isinstance(info, Mapping)
        if info["id"] == deployment_id:
            return info
    raise AssertionError(f"no deployment {deployment_id}")


def test_model_info_reads_back_a_datetime(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_typed_model_list())
    effective_from: Final = _model_info(core, "p1")["ptu_effective_from"]
    assert isinstance(effective_from, datetime)
    assert effective_from == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_model_info_reads_back_access_windows_typed(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_typed_model_list())
    windows: Final = _model_info(core, "w3")["access_windows"]
    assert isinstance(windows, tuple)
    (window,) = windows
    assert window["start"] == time(3)
    assert window["end"] == time(3, 0, 1)
    assert window["team_ids"] == ("t1",)


def test_db_timestamps_read_back_as_the_datetimes_stamped(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_typed_model_list())
    info: Final = _model_info(core, "db1")
    assert info["created_at"] == UTC_2030
    assert info["updated_at"] == UTC_2030
    created_at: Final = info["created_at"]
    assert isinstance(created_at, datetime)
    assert created_at.isoformat() == "2030-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_a_picked_deployment_carries_its_typed_model_info(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_typed_model_list())
    picked: Final = await core.select("d")
    assert isinstance(picked, Mapping)
    model_info: Final = picked["model_info"]
    assert model_info["created_at"] == UTC_2030


# --- PV1: routing_strategy_args -------------------------------------------------------------------


def test_routing_strategy_args_serialize_as_a_plain_dict(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO), routing_strategy_args={"ttl": 10})
    args: Final = core.routing_strategy_args  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the deprecated attribute
    assert isinstance(args, dict)
    assert to_jsonable_python(args) == {"ttl": 10}


def test_unset_routing_strategy_args_serialize_too(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_OF_TWO))
    assert to_jsonable_python(core.routing_strategy_args) == {}  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the deprecated attribute


# --- U13: the cooldown list -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_routing_group_with_nothing_left_lists_the_cooling_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/rc1", rate_limit())
    core: Final = make_core(
        model_list=[deployment("g1", "rc1"), deployment("g2", "rc2")],
        num_retries=0,
        routing_groups=[
            {
                "group_name": "pg",
                "models": ["g1", "g2"],
                "routing_strategy": "priority",
                "model_priorities": {"g1": 1, "g2": 2},
            }
        ],
    )
    _ = await complete(core, "pg")
    core.apply(ChangeSet(delete=("rc2",)))

    error: Final = await _failed(core, "pg", expected_error=RouterRateLimitError)

    assert isinstance(error, RouterRateLimitError)
    assert error.cooldown_list == ["rc1"]


@pytest.mark.asyncio
async def test_the_cooldown_list_names_every_cooling_deployment_of_the_router(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a1"), deployment("b", "b1")],
        num_retries=0,
        cooldown_time=60,
        allowed_fails_policy=AllowedFailsPolicy(
            RateLimitErrorAllowedFails=0
        ),  # a lone deployment stays warm without it
    )
    for deployment_id in ("a1", "b1"):
        core.health().record_failure(deployment_id, rate_limit(), 429)

    error: Final = await _failed(core, "a", expected_error=RouterRateLimitError)

    assert isinstance(error, RouterRateLimitError)
    assert error.cooldown_list == ["a1", "b1"]
    assert error.all_deployments_in_cooldown is True
    assert error.cooldown_deployments == ({"model_id": "a1", "api_base": None},)
