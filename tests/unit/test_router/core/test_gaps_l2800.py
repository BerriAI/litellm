"""
Gap tests for ``router_python_core.py`` lines 2800-5600: optional pre-call check re-registration,
silent experiments, per-deployment ``num_retries`` stamped on errors, and the request kwargs
``_update_kwargs_with_deployment`` builds (Kubernetes pod routing, baggage stamps, stream usage
injection, malformed request tools, per-user GitHub Copilot credentials).

Each test asserts what the Python core does today; oddities are named in the test's docstring.
"""

import asyncio
import threading
import time
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.constants import BAGGAGE_MAPPED_TAGS_METADATA_KEY, KUBERNETES_POD_ROUTING_KEY
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.litellm_params import STREAM_USAGE_INJECTED_FIELD
from litellm.types.router import BaggageMappedTagsStamp
from tests.unit.test_router.core.construction_precall_helpers import CallbackHost, host_callback_names
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit

MakeCore = Callable[..., RouterCore]
MESSAGES: Final = [{"role": "user", "content": "hi"}]


@pytest.fixture
def host() -> ScriptedHost:
    return CallbackHost()


def _metadata(kwargs: Mapping[str, object], name: str = "metadata") -> Mapping[str, object]:
    bucket: Final = kwargs[name]
    assert isinstance(bucket, Mapping)
    return bucket


async def _settle() -> None:
    for _ in range(30):
        await asyncio.sleep(0)


# --- optional pre-call checks added again at runtime (add_optional_pre_call_checks) ------------


def test_runtime_affinity_checks_merge_into_the_existing_affinity_callback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], optional_pre_call_checks=["deployment_affinity"])
    (existing,) = core.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    core.apply(ChangeSet(settings={"optional_pre_call_checks": ["deployment_affinity", "session_affinity"]}))
    (merged,) = core.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    assert merged is existing
    assert (
        merged.enable_user_key_affinity,
        merged.enable_responses_api_affinity,
        merged.enable_session_id_affinity,
    ) == (
        True,
        False,
        True,
    )
    assert host_callback_names(host)["callbacks"] == ["DeploymentAffinityCheck"]


def test_runtime_affinity_flags_are_only_ever_turned_on(make_core: MakeCore) -> None:
    """Oddity: a runtime list without ``deployment_affinity`` leaves the flag it set on, since the
    existing callback's flags are OR-ed with the new list's."""
    core: Final = make_core(model_list=[deployment("g", "a")], optional_pre_call_checks=["deployment_affinity"])
    core.apply(ChangeSet(settings={"optional_pre_call_checks": ["responses_api_deployment_check"]}))
    (merged,) = core.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    assert (
        merged.enable_user_key_affinity,
        merged.enable_responses_api_affinity,
        merged.enable_session_id_affinity,
    ) == (
        True,
        True,
        False,
    )


def test_runtime_affinity_merge_takes_the_model_group_affinity_config(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")],
        optional_pre_call_checks=["deployment_affinity"],
        model_group_affinity_config={"g": ["session_affinity"]},
    )
    core.apply(ChangeSet(settings={"optional_pre_call_checks": ["session_affinity"]}))
    (merged,) = [cb for cb in core.optional_callbacks if type(cb).__name__ == "DeploymentAffinityCheck"]  # pyright: ignore[reportAttributeAccessIssue]
    assert merged.model_group_affinity_config == {"g": ["session_affinity"]}
    assert merged.enable_session_id_affinity is True


def test_runtime_encrypted_content_affinity_reuses_the_existing_callback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], optional_pre_call_checks=["encrypted_content_affinity"])
    (existing,) = core.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    core.apply(ChangeSet(settings={"optional_pre_call_checks": ["encrypted_content_affinity"]}))
    (kept,) = core.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    assert kept is existing
    assert kept.enable_global_affinity is True
    assert host_callback_names(host)["callbacks"] == ["EncryptedContentAffinityCheck"]


def test_runtime_encrypted_content_affinity_turns_global_affinity_on_for_a_group_scoped_callback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], model_group_affinity_config={"g": ["encrypted_content_affinity"]}
    )
    (existing,) = [cb for cb in core.optional_callbacks if type(cb).__name__ == "EncryptedContentAffinityCheck"]  # pyright: ignore[reportAttributeAccessIssue]
    assert existing.enable_global_affinity is False
    core.add_optional_pre_call_checks(["encrypted_content_affinity"])  # pyright: ignore[reportAttributeAccessIssue]
    assert existing.enable_global_affinity is True
    assert host_callback_names(host)["callbacks"] == ["EncryptedContentAffinityCheck", "DeploymentAffinityCheck"]


async def test_router_budget_limiting_is_not_added_twice_beside_a_provider_budget_config(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")],
        provider_budget_config={"openai": {"budget_limit": 1, "time_period": "1d"}},
        optional_pre_call_checks=["router_budget_limiting"],
    )
    limiters: Final = [cb for cb in core.optional_callbacks if isinstance(cb, RouterBudgetLimiting)]  # pyright: ignore[reportAttributeAccessIssue]
    assert len(limiters) == 1
    assert core.registry().plugins("budget_limiter") is limiters[0]
    core.retire()


# --- silent experiments -----------------------------------------------------------------------


def _wait_for_calls(host: ScriptedHost, count: int) -> None:
    deadline: Final = time.monotonic() + 5
    while len(host.calls) < count and time.monotonic() < deadline:
        time.sleep(0.01)


def test_sync_completion_mirrors_to_the_silent_model_on_a_background_thread(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    core.completion(model="g", messages=MESSAGES, metadata={"trace": "t"})  # pyright: ignore[reportAttributeAccessIssue]
    _wait_for_calls(host, 2)
    # the thread starts before the primary call, so the two calls can land in either order
    assert sorted((call.op, call.deployment_id) for call in host.calls) == [("acompletion", "s1"), ("completion", "d1")]
    primary: Final = next(call for call in host.calls if call.deployment_id == "d1")
    shadow: Final = _metadata(next(call for call in host.calls if call.deployment_id == "s1").kwargs)
    assert (shadow["is_silent_experiment"], shadow["model_group"], shadow["trace"]) == (True, "shadow", "t")
    assert "is_silent_experiment" not in _metadata(primary.kwargs)


async def test_a_request_already_marked_silent_does_not_start_another_experiment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    await core.acompletion(model="g", messages=MESSAGES, metadata={"is_silent_experiment": True})  # pyright: ignore[reportAttributeAccessIssue]
    await _settle()
    assert host.deployments_called() == ["d1"]


def test_a_sync_request_already_marked_silent_does_not_start_another_experiment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    core.completion(model="g", messages=MESSAGES, metadata={"is_silent_experiment": True})  # pyright: ignore[reportAttributeAccessIssue]
    time.sleep(0.2)
    assert host.deployments_called() == ["d1"]


async def test_silent_experiment_marks_its_own_metadata_when_the_request_metadata_cannot_be_deep_copied(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    lock: Final = threading.Lock()
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    await core.acompletion(model="g", messages=MESSAGES, metadata={"lock": lock})  # pyright: ignore[reportAttributeAccessIssue]
    await _settle()
    primary: Final = next(call for call in host.calls if call.deployment_id == "d1")
    shadow: Final = next(call for call in host.calls if call.deployment_id == "s1")
    assert "is_silent_experiment" not in _metadata(primary.kwargs)
    assert _metadata(primary.kwargs)["model_group"] == "g"
    assert (_metadata(shadow.kwargs)["is_silent_experiment"], _metadata(shadow.kwargs)["lock"]) == (True, lock)


async def test_a_streaming_request_mirrors_a_streaming_silent_experiment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts

    host.answer("openai/d1", chat_stream("A", model="d1"))
    host.answer("openai/s1", chat_stream("S", model="s1"))
    core: Final = make_core(model_list=[deployment("g", "d1", silent_model="shadow"), deployment("shadow", "s1")])
    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    assert await collect_texts(stream) == "A"
    await _settle()
    shadow: Final = next(call for call in host.calls if call.deployment_id == "s1")
    assert shadow.kwargs["stream"] is True


# --- per-deployment num_retries stamped on the error ------------------------------------------


async def test_a_num_retries_already_on_the_error_wins_over_the_deployment_num_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    error: Final = rate_limit()
    error.num_retries = 0  # pyright: ignore[reportAttributeAccessIssue]
    host.answer("openai/d1", error)
    core: Final = make_core(model_list=[deployment("g", "d1", num_retries=2)])
    with pytest.raises(litellm.RateLimitError):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert len(host.calls) == 1


async def test_a_string_deployment_num_retries_is_used_as_an_int(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/d1", rate_limit())
    core: Final = make_core(model_list=[deployment("g", "d1", num_retries="1")])
    with pytest.raises(litellm.RateLimitError) as raised:
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert len(host.calls) == 2
    assert getattr(raised.value, "num_retries") == 1


async def test_a_non_numeric_deployment_num_retries_is_ignored(make_core: MakeCore, host: ScriptedHost) -> None:
    """The router's own default (2 retries) applies when the deployment value cannot be read as an int."""
    host.answer("openai/d1", rate_limit())
    core: Final = make_core(model_list=[deployment("g", "d1", num_retries="abc")])
    with pytest.raises(litellm.RateLimitError):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert len(host.calls) == 3


# --- request kwargs built for the deployment --------------------------------------------------


async def test_responses_replace_kubernetes_pod_routing_in_the_request_metadata(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """On the Responses path the routing goes to ``litellm_metadata``; a pod routing key the caller put
    in ``metadata`` is overwritten with the serving deployment's (none here)."""
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.aresponses(  # pyright: ignore[reportAttributeAccessIssue]
        model="g", input="hi", metadata={KUBERNETES_POD_ROUTING_KEY: {"pod": "stale"}, "keep": 1}
    )
    sent: Final = host.calls[0].kwargs
    assert _metadata(sent) == {KUBERNETES_POD_ROUTING_KEY: None, "keep": 1}
    assert _metadata(sent, "litellm_metadata")[KUBERNETES_POD_ROUTING_KEY] is None


async def test_a_baggage_stamp_records_the_serving_deployment_tags(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", tags=["t1", "t2"])])
    stamp: Final = BaggageMappedTagsStamp(tags=("b1",))
    await core.acompletion(  # pyright: ignore[reportAttributeAccessIssue]
        model="g", messages=MESSAGES, metadata={BAGGAGE_MAPPED_TAGS_METADATA_KEY: stamp, "tags": ["b1"]}
    )
    metadata: Final = _metadata(host.calls[0].kwargs)
    assert metadata[BAGGAGE_MAPPED_TAGS_METADATA_KEY] == BaggageMappedTagsStamp(
        tags=("b1",), serving_deployment_tags=("t1", "t2")
    )
    assert metadata["tags"] == ["t1", "t2", "b1"]


@pytest.mark.parametrize(
    ("request_extra", "injected"),
    [
        ({"stream": True}, True),
        ({"stream": True, "stream_options": {"include_usage": False}}, False),
        ({}, False),
    ],
)
async def test_a_deployment_stream_options_include_usage_marks_the_streaming_request_as_injected(
    make_core: MakeCore, host: ScriptedHost, request_extra: dict[str, object], injected: bool
) -> None:
    from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts

    host.answer("openai/d1", chat_stream("A"))
    core: Final = make_core(model_list=[deployment("g", "d1", stream_options={"include_usage": True})])
    response: Final = await core.acompletion(model="g", messages=MESSAGES, **request_extra)  # pyright: ignore[reportAttributeAccessIssue]
    if request_extra.get("stream"):
        await collect_texts(response)
    assert host.calls[0].kwargs.get(STREAM_USAGE_INJECTED_FIELD, False) is injected


async def test_malformed_request_tools_are_dropped_for_the_deployment_tools_when_drop_params_is_set(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    tool: Final = {"type": "function", "function": {"name": "dep_tool", "parameters": {}}}
    core: Final = make_core(model_list=[deployment("g", "d1", tools=[tool], drop_params=True)])
    await core.acompletion(model="g", messages=MESSAGES, tools="not-a-list", tool_choice="auto")  # pyright: ignore[reportAttributeAccessIssue]
    sent: Final = host.calls[0].kwargs
    assert sent["tools"] == [tool]
    assert "tool_choice" not in sent


async def test_a_per_user_github_copilot_deployment_rejects_another_credential_name(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[
            deployment(
                "g",
                "d1",
                model="github_copilot/gpt-4o",
                litellm_credential_name="copilot-cred",
                github_copilot_auth_type="per_user_oauth",
            )
        ],
        num_retries=0,
    )
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.acompletion(model="g", messages=MESSAGES, litellm_credential_name="someone-else")  # pyright: ignore[reportAttributeAccessIssue]
    assert "litellm_credential_name cannot be overridden on a deployment that uses per-user GitHub OAuth" in str(
        raised.value
    )
    assert host.calls == []


# --- timeouts ---------------------------------------------------------------------------------


async def test_the_process_request_timeout_outranks_the_router_timeout_once_a_router_timeout_is_set(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This per-attempt timeout is read from the host's process settings (not ``litellm.request_timeout``
    itself), and only when the router has a ``timeout`` of its own."""
    host.set_dynamic(request_timeout=33.0)
    monkeypatch.setattr(litellm, "request_timeout", 77.0)  # pyright: ignore[reportUnknownMemberType]  # pytest and the cores' untyped members
    core: Final = make_core(model_list=[deployment("g", "d1")], timeout=42)
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[0].kwargs["timeout"] == 33.0


async def test_the_process_request_timeout_is_the_stream_timeout_without_a_stream_timeout(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts

    host.set_dynamic(request_timeout=33.0)
    monkeypatch.setattr(litellm, "request_timeout", 77.0)  # pyright: ignore[reportUnknownMemberType]  # pytest and the cores' untyped members
    host.answer("openai/d1", chat_stream("A"))
    core: Final = make_core(model_list=[deployment("g", "d1")], timeout=42)
    await collect_texts(await core.acompletion(model="g", messages=MESSAGES, stream=True))  # pyright: ignore[reportAttributeAccessIssue]
    assert (host.calls[0].kwargs["stream_timeout"], host.calls[0].kwargs["timeout"]) == (33.0, 33.0)


async def test_default_litellm_params_timeout_applies_when_no_router_or_process_timeout_is_set(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(request_timeout=None)
    core: Final = make_core(model_list=[deployment("g", "d1")], default_litellm_params={"timeout": 9})
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[0].kwargs["timeout"] == 9


async def test_default_litellm_params_stream_timeout_is_the_last_stream_timeout_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts

    host.answer("openai/d1", chat_stream("A"))
    core: Final = make_core(model_list=[deployment("g", "d1")], default_litellm_params={"stream_timeout": 4})
    await collect_texts(await core.acompletion(model="g", messages=MESSAGES, stream=True))  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[0].kwargs["stream_timeout"] == 4


# --- sync completion runs the routing pre-call checks for a model group only -------------------


def test_sync_completion_runs_routing_pre_call_checks_for_a_group_but_not_a_deployment_id(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    from litellm.integrations.custom_logger import CustomLogger

    checked: Final[list[object]] = []

    class Recorder(CustomLogger):
        def pre_call_check(self, deployment: dict) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride]
            checked.append(deployment["model_info"]["id"])
            return None

    host.lists["callbacks"].append(Recorder())
    core: Final = make_core(model_list=[deployment("g", "d1")])
    core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    core.completion(model="d1", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert checked == ["d1"]
    assert host.deployments_called() == ["d1", "d1"]


def test_a_sync_routing_pre_call_check_error_stops_the_call(make_core: MakeCore, host: ScriptedHost) -> None:
    from litellm.integrations.custom_logger import CustomLogger

    class OverLimit(CustomLogger):
        def pre_call_check(self, deployment: dict) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride]
            raise litellm.RateLimitError(message="over the rpm limit", llm_provider="openai", model="d1")

    host.lists["callbacks"].append(OverLimit())
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    with pytest.raises(litellm.RateLimitError, match="over the rpm limit"):
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []


# --- per-model call counters ------------------------------------------------------------------


async def test_acompletion_counts_total_success_and_failed_calls_per_deployment_model(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    host.answer("openai/d1", rate_limit())
    with pytest.raises(litellm.RateLimitError):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    counts: Final = (
        core.total_calls["openai/d1"],  # pyright: ignore[reportAttributeAccessIssue]
        core.success_calls["openai/d1"],  # pyright: ignore[reportAttributeAccessIssue]
        core.fail_calls["openai/d1"],  # pyright: ignore[reportAttributeAccessIssue]
    )
    assert counts == (2, 1, 1)


def test_completion_counts_total_success_and_failed_calls_per_deployment_model(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The sync path counts calls as ``acompletion`` does."""
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    host.answer("openai/d1", rate_limit())
    with pytest.raises(litellm.RateLimitError):  # pyright: ignore[reportUnknownMemberType]  # pytest and the cores' untyped members
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    counts: Final = (  # pyright: ignore[reportUnknownVariableType]  # pytest and the cores' untyped members
        core.total_calls["openai/d1"],  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        core.success_calls["openai/d1"],  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        core.fail_calls["openai/d1"],  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )
    assert counts == (2, 1, 1)
