"""
Gap characterization for ``router_python_core.py`` lines 5600-8400: batch completions, the scheduler,
prompt-management groups, the image/audio/rerank/text/adapter endpoints, generic calls and their
pass-through endpoint rewrite, guardrail selection, embeddings, files and batches.

Each test pins a behavior the earlier characterization files leave unasserted; oddities are pinned as
Python does them today and called out in the test's docstring.
"""

import datetime
import inspect
import io
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router_core import RouterCore
from litellm.types.utils import CredentialItem, LiteLLMBatch
from tests.unit.test_router.core.endpoints_helpers import EndpointHost, patch_direct_endpoints
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    auth_error,
    deployment,
    rate_limit,
    server_error,
)

MESSAGES: Final = [{"role": "user", "content": "hi"}]
MakeCore = Callable[..., RouterCore]
TWO_DEPLOYMENTS: Final = [deployment("g", "d1"), deployment("g", "d2")]


@pytest.fixture
def host() -> EndpointHost:
    return EndpointHost()


def _batch() -> LiteLLMBatch:
    return LiteLLMBatch(
        id="b1",
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="f",
        object="batch",
        status="completed",
    )


def _hidden(response: object) -> Mapping[str, object]:
    return getattr(response, "_hidden_params")


def _headers(response: object) -> Mapping[str, object]:
    found: Final = _hidden(response)["additional_headers"]
    assert isinstance(found, Mapping)
    return found


# The routed endpoints that answer through ``async_function_with_fallbacks`` with their own attempt helper.
ROUTED_ENDPOINTS: Final[dict[str, dict[str, object]]] = {
    "aimage_generation": {"prompt": "p"},
    "atranscription": {"file": b"audio"},
    "aspeech": {"input": "x", "voice": "alloy"},
    "arerank": {"query": "q", "documents": ["a"]},
    "aadapter_completion": {"adapter_id": "ad", "messages": MESSAGES},
    "atext_completion": {"prompt": "p"},
    "aembedding": {"input": ["a"]},
    "acreate_batch": {"input_file_id": "f", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    "acancel_batch": {"batch_id": "b1"},
}

SYNC_COUNTED: Final[dict[str, dict[str, object]]] = {
    "completion": {"messages": MESSAGES},
    "embedding": {"input": ["a"]},
}

# --- batch completions (5565-5790) ----------------------------------------------------------


async def test_abatch_completion_with_message_lists_keeps_a_failed_request_in_its_own_row(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("openai/d1", rate_limit("d1", "a failed"))
    core: Final = make_core(model_list=[deployment("a", "d1"), deployment("b", "d2")], num_retries=0)
    rows: Final = await core.abatch_completion(models=["a", "b"], messages=[MESSAGES, MESSAGES])
    assert [[type(result).__name__ for result in row] for row in rows] == [
        ["RateLimitError", "ModelResponse"],
        ["RateLimitError", "ModelResponse"],
    ]


async def test_abatch_completion_with_mixed_message_shapes_returns_none_without_calling(
    make_core: MakeCore, host: EndpointHost
):
    """Neither every element a dict nor every element a list: no branch matches and None comes back."""
    core: Final = make_core(model_list=[deployment("a", "d1")])
    assert await core.abatch_completion(models=["a"], messages=[MESSAGES, {"role": "user", "content": "x"}]) is None
    assert host.calls == []


async def test_fastest_response_forwards_the_stream_flag(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("a", "d1")])
    await core.abatch_completion_fastest_response(model="a", messages=MESSAGES, stream=True)
    assert host.calls[0].kwargs["stream"] is True


# --- scheduler (5792-5912) ------------------------------------------------------------------


async def test_schedule_acompletion_forwards_the_stream_flag(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.schedule_acompletion(model="g", messages=MESSAGES, priority=1, stream=True)
    assert host.calls[0].kwargs["stream"] is True


async def test_a_prioritized_request_for_a_deployment_id_is_queued_and_served_by_it(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(3):
        response = await core.acompletion(model="d1", messages=MESSAGES, priority=2)
        assert _headers(response)["x-litellm-request-prioritization-used"] is True
    assert host.deployments_called() == ["d1", "d1", "d1"]


async def test_a_bad_priority_for_a_deployment_id_follows_that_deployments_drop_params(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", drop_params=True), deployment("g", "d2")])
    response: Final = await core.acompletion(model="d1", messages=MESSAGES, priority="high")
    assert "x-litellm-request-prioritization-used" not in _headers(response)
    assert "priority" not in host.calls[0].kwargs
    with pytest.raises(litellm.BadRequestError, match="priority must be an integer, got str 'high'"):
        await core.acompletion(model="d2", messages=MESSAGES, priority="high")
    assert host.deployments_called() == ["d1"]


async def test_a_bad_priority_is_dropped_when_any_deployment_of_the_group_drops_params(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", drop_params=True), deployment("g", "d2")])
    response: Final = await core.acompletion(model="g", messages=MESSAGES, priority="high")
    assert "x-litellm-request-prioritization-used" not in _headers(response)
    assert len(host.calls) == 1


# --- prompt-management groups (5893-6034) ---------------------------------------------------


class _PromptLogging(LiteLLMLogging):
    """A logging object whose prompt template renders to ``rendered_model`` with a system message."""

    rendered_model: str = "openai/rendered"
    seen: Mapping[str, object] = {}

    def get_chat_completion_prompt(self, **kwargs: object):  # kwargs-ok: Logging's own keyword signature
        self.seen = kwargs
        return self.rendered_model, [{"role": "system", "content": "rendered"}], {"temperature": 0.7}


def _prompt_logging(rendered_model: str = "openai/rendered") -> _PromptLogging:
    logging_obj: Final = _PromptLogging(
        model="pm",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="c1",
        function_id="f1",
    )
    logging_obj.rendered_model = rendered_model
    return logging_obj


async def test_a_prompt_management_group_renders_the_deployment_prompt_and_calls_the_rendered_model(
    make_core: MakeCore, host: EndpointHost
):
    """The deployment's prompt_id/variables/label reach the template; the prompt-management config keys
    (bitbucket_config, ...) are stripped from what is sent; the rendered model, messages and optional
    params are sent directly (no routing hidden params) since the rendered model is no group."""
    core: Final = make_core(
        model_list=[
            deployment(
                "pm",
                "p1",
                model="langfuse/gpt-4o",
                prompt_id="dep-pid",
                prompt_variables={"a": 1},
                prompt_label="lbl",
                bitbucket_config={"x": 1},
                timeout=7,
            )
        ]
    )
    logging_obj: Final = _prompt_logging()
    response: Final = await core.acompletion(model="pm", messages=MESSAGES, litellm_logging_obj=logging_obj)
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id, call.model_group) == ("acompletion", "openai/rendered", "p1", "pm")
    sent: Final = call.kwargs
    assert (sent["prompt_id"], sent["prompt_variables"], sent["prompt_label"]) == ("dep-pid", {"a": 1}, "lbl")
    assert "bitbucket_config" not in sent
    assert (sent["temperature"], sent["timeout"], sent["messages"]) == (
        0.7,
        7.0,
        [{"role": "system", "content": "rendered"}],
    )
    assert sent["litellm_logging_obj"] is logging_obj
    assert (logging_obj.seen["model"], logging_obj.seen["messages"], logging_obj.seen["prompt_id"]) == (
        "langfuse/gpt-4o",
        MESSAGES,
        "dep-pid",
    )
    assert logging_obj.seen["injected_for_every_deployment"] is True
    assert _hidden(response) == {}


async def test_a_prompt_rendered_to_another_group_routes_into_it_with_the_request_prompt_id_and_timeout(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[
            deployment("pm", "p1", model="langfuse/gpt-4o", prompt_id="dep-pid", timeout=9),
            deployment("rendered-g", "r1"),
        ]
    )
    response: Final = await core.acompletion(
        model="pm",
        messages=MESSAGES,
        litellm_logging_obj=_prompt_logging("rendered-g"),
        prompt_id="req-pid",
        timeout=3,
    )
    call: Final = host.calls[0]
    assert (call.model, call.model_group, call.kwargs["prompt_id"], call.kwargs["timeout"]) == (
        "openai/r1",
        "pm",
        "req-pid",
        3,
    )
    assert _headers(response)["x-litellm-model-group"] == "rendered-g"


async def test_a_single_deployment_without_a_provider_prefix_is_not_a_prompt_management_group(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", model="gpt-4o")])
    response: Final = await core.acompletion(model="g", messages=MESSAGES)
    assert (host.calls[0].model, host.calls[0].deployment_id) == ("gpt-4o", "d1")
    assert _hidden(response)["model_id"] == "d1"


# --- routed endpoints: retries and counters (6036-6603, 7646-7789, 8026-8315) -------------


@pytest.mark.parametrize("endpoint", sorted(ROUTED_ENDPOINTS))
async def test_a_routed_endpoint_retries_a_401_on_the_other_deployment(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    for _ in range(4):
        host.calls.clear()
        host.answer("*", auth_error(), None)
        core = make_core(model_list=TWO_DEPLOYMENTS, num_retries=1, disable_cooldowns=True)
        await core.call(endpoint, {"model": "g", **ROUTED_ENDPOINTS[endpoint]})
        assert sorted(host.deployments_called()) == ["d1", "d2"]


@pytest.mark.parametrize("endpoint", sorted(ROUTED_ENDPOINTS))
async def test_a_routed_endpoint_ignores_the_deployment_num_retries(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    """Unlike acompletion and the generic endpoints, these attempt helpers never stamp the deployment's
    num_retries on the error, so the router's num_retries=0 governs."""
    host.answer("*", server_error())
    core: Final = make_core(model_list=[deployment("g", "d1", num_retries=2)], num_retries=0)
    with pytest.raises(litellm.InternalServerError):
        await core.call(endpoint, {"model": "g", **ROUTED_ENDPOINTS[endpoint]})
    assert len(host.calls) == 1
    assert host.raised[-1].num_retries is None  # pyright: ignore[reportAttributeAccessIssue]  # stamped by the retry loop


async def test_a_generic_endpoint_retries_with_the_deployment_num_retries(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch, host: EndpointHost
):
    patch_direct_endpoints(monkeypatch, host)
    host.answer("*", server_error())
    core: Final = make_core(model_list=[deployment("g", "d1", num_retries=2)], num_retries=0)
    with pytest.raises(litellm.InternalServerError) as raised:
        await core.aresponses(model="g", input="hi")
    assert len(host.calls) == 3
    assert raised.value.num_retries == 2  # pyright: ignore[reportAttributeAccessIssue]  # stamped from the deployment


async def test_sync_image_generation_is_retried_with_the_router_num_retries(make_core: MakeCore, host: EndpointHost):
    host.answer("*", rate_limit())
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=2)
    with pytest.raises(litellm.RateLimitError):
        core.image_generation(model="g", prompt="p")
    assert len(host.calls_to("image_generation")) == 3


@pytest.mark.parametrize("endpoint", ["aimage_generation", "image_generation"])
async def test_image_generation_for_an_unknown_group_raises_without_calling_the_host(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    with pytest.raises(litellm.BadRequestError, match="There are no healthy deployments for this model"):
        await core.call(endpoint, {"model": "nope", "prompt": "p"})
    assert host.calls == []


@pytest.mark.parametrize(
    ("endpoint", "total_key", "fail_key"),
    [
        ("aimage_generation", "openai/d1", "openai/d1"),
        ("atranscription", "g", "g"),
        ("aspeech", "g", "g"),
        ("arerank", "openai/d1", "openai/d1"),
        ("aadapter_completion", "openai/d1", "g"),
        ("atext_completion", "openai/d1", "g"),
        ("aembedding", "openai/d1", "openai/d1"),
        ("acreate_batch", "openai/d1", "g"),
        ("acancel_batch", "openai/d1", "g"),
        ("image_generation", "openai/d1", "openai/d1"),
        ("text_completion", "openai/d1", "g"),
        ("completion", "openai/d1", "openai/d1"),
        ("embedding", "openai/d1", "openai/d1"),
    ],
)
async def test_call_counters_are_keyed_per_endpoint_and_the_same_for_sync_and_async(
    make_core: MakeCore, host: EndpointHost, endpoint: str, total_key: str, fail_key: str
):
    """total_calls/fail_calls are keyed by the deployment model for some endpoints and by the model group
    for others (fail_calls even differs from total_calls for some); a sync endpoint counts as its async
    twin does."""
    host.answer("*", rate_limit())
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=1)
    request: Final = SYNC_COUNTED.get(endpoint) or ROUTED_ENDPOINTS.get(endpoint, {"prompt": "p"})
    with pytest.raises(litellm.RateLimitError):
        await core.call(endpoint, {"model": "g", **request})
    assert dict(core.total_calls) == {total_key: 2}  # pyright: ignore[reportAttributeAccessIssue]  # a public counter
    assert dict(core.fail_calls) == {fail_key: 2}  # pyright: ignore[reportAttributeAccessIssue]  # a public counter
    assert dict(core.success_calls) == {}  # pyright: ignore[reportAttributeAccessIssue]  # a public counter


async def test_success_counters_count_the_deployment_model(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.aspeech(model="g", input="x", voice="alloy")
    await core.aembedding(model="g", input=["a"])
    assert dict(core.total_calls) == {"g": 1, "openai/d1": 1}  # pyright: ignore[reportAttributeAccessIssue]  # a public counter
    assert dict(core.success_calls) == {"g": 1, "openai/d1": 1}  # pyright: ignore[reportAttributeAccessIssue]  # a public counter


# --- guardrails (6692-6721) -----------------------------------------------------------------


async def test_aguardrail_picks_among_same_named_guardrails_by_weight(make_core: MakeCore, host: ScriptedHost):
    picked: Final[list[object]] = []

    async def guard(**kwargs: object) -> str:  # kwargs-ok: the guardrail's keyword payload
        selected: Final = kwargs["selected_guardrail"]
        assert isinstance(selected, Mapping)
        picked.append(selected["litellm_params"]["guardrail"])
        return "verdict"

    guardrails: Final = [
        {"guardrail_name": "gr", "litellm_params": {"guardrail": "never", "weight": 0}},
        {"guardrail_name": "gr", "litellm_params": {"guardrail": "always", "weight": 1}},
    ]
    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=guardrails)
    for _ in range(10):
        await core.aguardrail(guardrail_name="gr", original_function=guard)
    assert set(picked) == {"always"}


async def test_aguardrail_without_weights_spreads_over_same_named_guardrails(make_core: MakeCore, host: ScriptedHost):
    picked: Final[set[object]] = set()

    async def guard(**kwargs: object) -> str:  # kwargs-ok: the guardrail's keyword payload
        selected: Final = kwargs["selected_guardrail"]
        assert isinstance(selected, Mapping)
        picked.add(selected["litellm_params"]["guardrail"])
        return "verdict"

    guardrails: Final = [
        {"guardrail_name": "gr", "litellm_params": {"guardrail": "a"}},
        {"guardrail_name": "gr", "litellm_params": {"guardrail": "b"}},
    ]
    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=guardrails)
    for _ in range(60):
        await core.aguardrail(guardrail_name="gr", original_function=guard)
    assert picked == {"a", "b"}


# --- generic calls: pass-through endpoint rewrite (6754-6781, 7561-7627) --------------------


async def test_async_passthrough_rewrites_only_whole_path_segments_naming_the_group(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch, host: EndpointHost
):
    patch_direct_endpoints(monkeypatch, host)
    core: Final = make_core(model_list=[deployment("g", "d1", model="bedrock/us.anthropic.claude")])
    await core.allm_passthrough_route(model="g", method="POST", endpoint="/model/g/invoke/gg/g:stream")
    assert host.calls[0].kwargs["endpoint"] == "/model/us.anthropic.claude/invoke/gg/us.anthropic.claude:stream"


@pytest.mark.parametrize("endpoint", ["aresponses", "responses"])  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]  # pytest and the cores' untyped members
async def test_generic_call_rewrites_only_whole_path_segments_naming_the_group(
    make_core: MakeCore,
    monkeypatch: pytest.MonkeyPatch,
    host: EndpointHost,
    endpoint: str,  # pyright: ignore[reportUnknownMemberType, reportUnknownParameterType]  # pytest and the cores' untyped members
):
    """Sync and async both swap the group name for the provider-stripped deployment model one whole path
    segment at a time, so a segment that merely contains the group name ("gg") is left alone."""
    patch_direct_endpoints(monkeypatch, host)
    core: Final = make_core(model_list=[deployment("g", "d1")])
    sent: Final = getattr(core, endpoint)(model="g", input="hi", endpoint="/gg/g")  # pyright: ignore[reportAny]  # pytest and the cores' untyped members
    if inspect.isawaitable(sent):  # pyright: ignore[reportAny]  # pytest and the cores' untyped members
        await sent
    call: Final = host.calls[0]
    assert (call.model, call.kwargs["endpoint"]) == ("openai/d1", "/gg/d1")


# --- batches (8112-8362) --------------------------------------------------------------------


async def test_aretrieve_batch_sends_the_deployment_provider_over_the_request_one(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("*", _batch())
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("h", "d2", custom_llm_provider="azure")])
    await core.aretrieve_batch(model="g", batch_id="b1", custom_llm_provider="azure")
    await core.aretrieve_batch(model="h", batch_id="b1")
    assert [call.kwargs["custom_llm_provider"] for call in host.calls] == ["openai", "azure"]


async def test_alist_batches_takes_the_first_set_first_id_and_the_last_answer_last_id(
    make_core: MakeCore, host: EndpointHost
):
    """last_id is overwritten by every answer in deployment order, even with None."""
    host.answer("openai/d1", SimpleNamespace(first_id=None, last_id="b", data=[1], has_more=False))
    host.answer("openai/d2", SimpleNamespace(first_id="c", last_id=None, data=[2], has_more=False))
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    assert await core.alist_batches(model="g") == {
        "object": "list",
        "data": [1, 2],
        "first_id": "c",
        "last_id": None,
        "has_more": False,
    }


async def test_acancel_batch_resolves_the_deployment_credential_but_acreate_batch_does_not(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch, host: EndpointHost
):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="cred1",
                credential_values={"api_key": "sk-cred", "api_base": "https://cred.example"},
                credential_info={},
            )
        ],
    )
    host.answer("*", _batch())
    core: Final = make_core(model_list=[deployment("g", "d1", litellm_credential_name="cred1")])
    await core.acancel_batch(model="g", batch_id="b1")
    await core.acreate_batch(model="g", **ROUTED_ENDPOINTS["acreate_batch"])
    cancel, create = host.calls[0].kwargs, host.calls[1].kwargs
    assert (cancel["api_key"], cancel["api_base"], "litellm_credential_name" in cancel) == (
        "sk-cred",
        "https://cred.example",
        False,
    )
    assert (create["api_key"], create.get("api_base"), create["litellm_credential_name"]) == ("sk-test", None, "cred1")


# --- files (7814-7934) ----------------------------------------------------------------------


async def test_acreate_file_passthrough_keeps_the_batch_file_model(make_core: MakeCore, host: EndpointHost):
    upload: Final = io.BytesIO(b'{"custom_id": "1", "body": {"model": "g", "messages": []}}\n')
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acreate_file(model="g", file=upload, purpose="batch", passthrough=True)
    sent: Final = host.calls[0].kwargs["file"]
    assert isinstance(sent, io.BytesIO)
    assert b'"model": "g"' in sent.getvalue()
