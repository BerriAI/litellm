"""
Characterization of the fan-out endpoints and request shapers: the files API (one file per
deployment), batches (create, retrieve across deployments, cancel, list), batch completions,
fastest-response completion, the scheduler (priority) and prompt-management model groups.
"""

import re
import datetime
import io
import json
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router_core import RouterCore
from litellm.types.utils import LiteLLMBatch, ModelResponse
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.scripted_host import ScriptedHost, bad_request, deployment

MESSAGES: Final = [{"role": "user", "content": "hi"}]
MakeCore = Callable[..., RouterCore]
TWO_DEPLOYMENTS: Final = [deployment("g", "d1"), deployment("g", "d2")]


@pytest.fixture
def host() -> EndpointHost:
    return EndpointHost()


def _bucket(kwargs: Mapping[str, object], name: str) -> Mapping[str, object]:
    bucket: Final = kwargs[name]
    assert isinstance(bucket, Mapping)
    return bucket


def _batch(batch_id: str = "b1") -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="f",
        object="batch",
        status="completed",
    )


def _jsonl() -> io.BytesIO:
    return io.BytesIO(b'{"custom_id": "1", "body": {"model": "g", "messages": []}}\n')


# --- files ----------------------------------------------------------------------------------


async def test_acreate_file_creates_the_file_on_every_deployment_of_the_group(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.acreate_file(model="g", file=_jsonl(), purpose="batch")
    assert sorted(host.deployments_called("acreate_file")) == ["d1", "d2"]


async def test_acreate_file_returns_the_first_response_with_the_deployment_file_id_mapping(
    make_core: MakeCore, host: EndpointHost
):
    first: Final = ModelResponse(id="file-1")
    second: Final = ModelResponse(id="file-2")
    host.answer("openai/d1", first)
    host.answer("openai/d2", second)
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    response: Final = await core.acreate_file(model="g", file=_jsonl(), purpose="batch")
    assert response in (first, second)
    assert response._hidden_params["model_file_id_mapping"] == {"d1": "file-1", "d2": "file-2"}
    assert "model_id" not in response._hidden_params


async def test_acreate_file_rewrites_the_jsonl_model_to_each_deployment_model(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.acreate_file(model="g", file=_jsonl(), purpose="batch")
    models: Final = {}
    for call in host.calls:
        sent_file: Final = call.kwargs["file"]
        assert isinstance(sent_file, io.BytesIO)
        models[call.deployment_id] = json.loads(sent_file.getvalue().decode().strip())["body"]["model"]
    assert models == {"d1": "d1", "d2": "d2"}


async def test_acreate_file_keeps_a_non_batch_file_as_sent(make_core: MakeCore, host: EndpointHost):
    upload: Final = _jsonl()
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acreate_file(model="g", file=upload, purpose="assistants")
    assert host.calls[0].kwargs["file"] is not None
    sent: Final = host.calls[0].kwargs["file"]
    assert isinstance(sent, io.BytesIO)
    assert json.loads(sent.getvalue().decode().strip())["body"]["model"] == "g"


async def test_acreate_file_puts_the_model_group_in_metadata_and_the_deployment_in_litellm_metadata(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acreate_file(model="g", file=_jsonl(), purpose="batch")
    sent: Final = host.calls[0].kwargs
    assert _bucket(sent, "metadata")["model_group"] == "g"
    assert "deployment" not in _bucket(sent, "metadata")
    assert _bucket(sent, "litellm_metadata")["deployment"] == "openai/d1"
    assert "model_group" not in _bucket(sent, "litellm_metadata")
    assert sent["custom_llm_provider"] == "openai"


async def test_acreate_file_without_a_purpose_raises_before_calling_the_host(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    with pytest.raises(Exception, match="file and file_purpose are required for create_file"):
        await core.acreate_file(model="g", file=_jsonl())
    assert host.calls == []


async def test_acreate_file_sends_target_model_names_to_a_litellm_proxy_deployment(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", model="litellm_proxy/remote-model")])
    await core.acreate_file(model="g", file=_jsonl(), purpose="batch", extra_body={"keep": 1})
    assert dict(host.calls[0].kwargs["extra_body"]) == {"keep": 1, "target_model_names": "remote-model"}  # pyright: ignore[reportArgumentType]  # a mapping


async def test_acreate_file_sends_the_deployment_gcs_bucket_in_litellm_metadata(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", gcs_bucket_name="bucket-1")])
    await core.acreate_file(model="g", file=_jsonl(), purpose="batch")
    assert _bucket(host.calls[0].kwargs, "litellm_metadata")["gcs_bucket_name"] == "bucket-1"


# --- batches --------------------------------------------------------------------------------


async def test_acreate_batch_routes_with_litellm_metadata_only(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    response: Final = await core.acreate_batch(
        model="g", input_file_id="f", endpoint="/v1/chat/completions", completion_window="24h"
    )
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id, call.kwargs["custom_llm_provider"]) == (
        "acreate_batch",
        "openai/d1",
        "d1",
        "openai",
    )
    assert "metadata" not in call.kwargs
    assert _bucket(call.kwargs, "litellm_metadata")["model_group"] == "g"
    assert response._hidden_params["model_id"] == "d1"


async def test_acreate_batch_is_retried(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"), _batch())
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=1)
    await core.acreate_batch(model="g", input_file_id="f", endpoint="/v1/chat/completions", completion_window="24h")
    assert len(host.calls_to("acreate_batch")) == 2


async def test_aretrieve_batch_asks_every_deployment_and_returns_the_batch_found(
    make_core: MakeCore, host: EndpointHost
):
    found: Final = _batch()
    host.answer("openai/d1", bad_request("d1", "no such batch"))
    host.answer("openai/d2", found)
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    assert await core.aretrieve_batch(model="g", batch_id="b1") is found
    assert sorted(host.deployments_called("aretrieve_batch")) == ["d1", "d2"]
    for call in host.calls:
        assert (call.kwargs["batch_id"], call.kwargs["custom_llm_provider"], call.model_group) == ("b1", "openai", "g")
        assert "metadata" not in call.kwargs


async def test_aretrieve_batch_raises_the_first_error_when_no_deployment_has_the_batch(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("openai/d1", bad_request("d1", "first missing"))
    host.answer("openai/d2", bad_request("d2", "second missing"))
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    with pytest.raises(litellm.BadRequestError, match="missing"):
        await core.aretrieve_batch(model="g", batch_id="b1")
    assert len(host.calls) == 2


async def test_aretrieve_batch_ignores_a_non_batch_answer_and_raises_a_generic_error(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    with pytest.raises(Exception, match="Unable to find batch in any model"):
        await core.aretrieve_batch(model="g", batch_id="b1")


async def test_aretrieve_batch_without_a_model_asks_every_deployment_with_its_own_group(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("*", bad_request(message="missing"))
    core: Final = make_core(model_list=[*TWO_DEPLOYMENTS, deployment("h", "d3")])
    with pytest.raises(litellm.BadRequestError):
        await core.aretrieve_batch(batch_id="b1")
    assert sorted((call.deployment_id, call.model_group) for call in host.calls) == [
        ("d1", "g"),
        ("d2", "g"),
        ("d3", "h"),
    ]


async def test_aretrieve_batch_is_not_retried(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"))
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=3)
    with pytest.raises(litellm.InternalServerError):
        await core.aretrieve_batch(model="g", batch_id="b1")
    assert len(host.calls) == 1


async def test_acancel_batch_routes_with_litellm_metadata(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", _batch())
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    response: Final = await core.acancel_batch(model="g", batch_id="b1")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["batch_id"], call.kwargs["api_base"]) == (
        "acancel_batch",
        "d1",
        "b1",
        "https://d1.example",
    )
    assert "metadata" not in call.kwargs
    assert response._hidden_params["model_id"] == "d1"


async def test_alist_batches_merges_the_lists_of_every_deployment(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", SimpleNamespace(first_id="a", last_id="b", data=[1, 2], has_more=False))
    host.answer("openai/d2", SimpleNamespace(first_id="c", last_id="d", data=[3], has_more=True))
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    merged: Final = await core.alist_batches(model="g", limit=2)
    assert sorted(merged["data"]) == [1, 2, 3]
    assert (merged["object"], merged["has_more"]) == ("list", True)
    assert merged["first_id"] in ("a", "c")
    assert merged["last_id"] in ("b", "d")


async def test_alist_batches_sends_the_deployment_params_and_request_without_routing_metadata(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("*", SimpleNamespace(first_id=None, last_id=None, data=[], has_more=False))
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    await core.alist_batches(model="g", limit=2, api_base="https://request.example")
    call: Final = host.calls[0]
    assert (call.model, call.kwargs["limit"], call.kwargs["api_base"], call.deployment_id) == (
        "openai/d1",
        2,
        "https://request.example",
        None,
    )
    assert "metadata" not in call.kwargs and "litellm_metadata" not in call.kwargs


async def test_alist_batches_drops_failing_deployments(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", bad_request("d1"))
    host.answer("openai/d2", SimpleNamespace(first_id="c", last_id="d", data=[3], has_more=False))
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    merged: Final = await core.alist_batches(model="g")
    assert merged == {"object": "list", "data": [3], "first_id": "c", "last_id": "d", "has_more": False}


async def test_alist_batches_for_an_unknown_group_returns_an_empty_list(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    assert await core.alist_batches(model="nope") == {
        "object": "list",
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }
    assert host.calls == []


# --- batch completions ------------------------------------------------------------------------


async def test_abatch_completion_sends_one_request_to_each_model_and_returns_errors_in_place(
    make_core: MakeCore, host: EndpointHost
):
    error: Final = bad_request("d1", "a failed")
    host.answer("openai/d1", error)
    core: Final = make_core(model_list=[deployment("a", "d1"), deployment("b", "d2")], num_retries=0)
    results: Final = await core.abatch_completion(models=["a", "b"], messages=MESSAGES)
    assert isinstance(results[0], litellm.BadRequestError)
    assert isinstance(results[1], ModelResponse) and results[1]._hidden_params["model_id"] == "d2"


async def test_abatch_completion_with_message_lists_groups_results_per_request(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("a", "d1"), deployment("b", "d2")])
    second: Final = [{"role": "user", "content": "second"}]
    results: Final = await core.abatch_completion(models=["a", "b"], messages=[MESSAGES, second])
    assert [[response._hidden_params["model_id"] for response in row] for row in results] == [
        ["d1", "d2"],
        ["d1", "d2"],
    ]
    assert sorted(str(call.kwargs["messages"]) for call in host.calls) == sorted(
        [str(MESSAGES), str(MESSAGES), str(second), str(second)]
    )


async def test_abatch_completion_forwards_extra_kwargs_to_every_request(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("a", "d1"), deployment("b", "d2")])
    await core.abatch_completion(models=["a", "b"], messages=MESSAGES, max_tokens=15)
    assert [call.kwargs["max_tokens"] for call in host.calls] == [15, 15]


async def test_abatch_completion_one_model_multiple_requests_returns_one_result_per_request(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("openai/d1", ModelResponse(id="r1"), bad_request("d1"), ModelResponse(id="r3"))
    core: Final = make_core(model_list=[deployment("a", "d1")], num_retries=0)
    results: Final = await core.abatch_completion_one_model_multiple_requests(model="a", messages=[MESSAGES] * 3)
    assert len(results) == 3
    assert sum(isinstance(result, litellm.BadRequestError) for result in results) == 1
    assert len(host.calls) == 3


async def test_fastest_response_returns_the_first_success_marked_in_hidden_params(
    make_core: MakeCore, host: EndpointHost
):
    host.answer("openai/d1", bad_request("d1"))
    core: Final = make_core(model_list=[deployment("a", "d1"), deployment("b", "d2")], num_retries=0)
    response: Final = await core.abatch_completion_fastest_response(model="a, b", messages=MESSAGES)
    assert response._hidden_params["model_id"] == "d2"
    assert response._hidden_params["fastest_response_batch_completion"] is True


async def test_fastest_response_raises_when_every_model_fails(make_core: MakeCore, host: EndpointHost):
    host.answer("*", bad_request())
    core: Final = make_core(model_list=[deployment("a", "d1"), deployment("b", "d2")], num_retries=0)
    with pytest.raises(Exception, match="All tasks failed"):
        await core.abatch_completion_fastest_response(model="a,b", messages=MESSAGES)


async def test_fastest_response_with_an_unknown_model_counts_it_as_failed(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("a", "d1")], num_retries=0)
    response: Final = await core.abatch_completion_fastest_response(model="nope, a", messages=MESSAGES)
    assert response._hidden_params["model_id"] == "d1"


# --- scheduler / priority -------------------------------------------------------------------


async def test_schedule_acompletion_marks_the_response_as_prioritized_and_does_not_send_priority(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    response: Final = await core.schedule_acompletion(model="g", messages=MESSAGES, priority=0)
    assert response._hidden_params["additional_headers"]["x-litellm-request-prioritization-used"] is True
    assert "priority" not in host.calls[0].kwargs


async def test_schedule_acompletion_without_a_priority_or_default_raises(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    with pytest.raises(
        litellm.BadRequestError,
        match=re.escape("priority is required to queue a request, send one or set router_settings.default_priority"),
    ):
        await core.schedule_acompletion(model="g", messages=MESSAGES)
    assert host.calls == []


async def test_schedule_acompletion_without_a_priority_uses_the_default_priority(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_priority=5)
    response: Final = await core.schedule_acompletion(model="g", messages=MESSAGES)
    assert response._hidden_params["additional_headers"]["x-litellm-request-prioritization-used"] is True


@pytest.mark.parametrize("endpoint", ["schedule_acompletion", "acompletion"])
async def test_a_non_integer_priority_raises_bad_request(make_core: MakeCore, host: EndpointHost, endpoint: str):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    with pytest.raises(litellm.BadRequestError, match="priority must be an integer, got str 'high'"):
        await core.call(endpoint, {"model": "g", "messages": MESSAGES, "priority": "high"})
    assert host.calls == []


async def test_a_non_integer_priority_with_drop_params_is_served_unscheduled(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    response: Final = await core.acompletion(model="g", messages=MESSAGES, priority="high", drop_params=True)
    assert "x-litellm-request-prioritization-used" not in response._hidden_params["additional_headers"]
    assert "priority" not in host.calls[0].kwargs


async def test_a_non_integer_priority_with_drop_params_queues_at_the_default_priority(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_priority=5)
    response: Final = await core.acompletion(model="g", messages=MESSAGES, priority="high", drop_params=True)
    assert response._hidden_params["additional_headers"]["x-litellm-request-prioritization-used"] is True


@pytest.mark.parametrize(
    ("endpoint", "request_body"),
    [("acompletion", {"messages": MESSAGES}), ("atext_completion", {"prompt": "p"})],
)
async def test_a_request_priority_schedules_the_call(
    make_core: MakeCore, host: EndpointHost, endpoint: str, request_body: dict[str, object]
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    response: Final = await core.call(endpoint, {"model": "g", "priority": 3, **request_body})
    assert response._hidden_params["additional_headers"]["x-litellm-request-prioritization-used"] is True  # pyright: ignore[reportAttributeAccessIssue]  # the scripted response
    assert "priority" not in host.calls[0].kwargs


async def test_default_priority_schedules_every_acompletion(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], default_priority=5)
    response: Final = await core.acompletion(model="g", messages=MESSAGES)
    assert response._hidden_params["additional_headers"]["x-litellm-request-prioritization-used"] is True


async def test_a_scheduled_call_error_carries_its_priority(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", bad_request("d1"))
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.acompletion(model="g", messages=MESSAGES, priority=4)
    assert raised.value.priority == 4  # pyright: ignore[reportAttributeAccessIssue]  # stamped by the scheduler


@pytest.mark.parametrize(("header", "expected"), [("7", {"priority": 7}), (None, {}), ("", {})])
async def test_priority_header_update_parses_the_priority_header(
    make_core: MakeCore, host: EndpointHost, header: str | None, expected: dict[str, int]
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    assert dict(core.priority_header_update("g", {}, header)) == expected


# --- prompt management models -----------------------------------------------------------------


def _logging_obj() -> LiteLLMLogging:
    return LiteLLMLogging(
        model="pm",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="c1",
        function_id="f1",
    )


async def test_prompt_management_group_without_a_prompt_id_raises_value_error(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("pm", "p1", model="langfuse/gpt-4o")])
    with pytest.raises(ValueError, match=re.escape("Prompt ID is not set or not a string. Got=None")):
        await core.acompletion(model="pm", messages=MESSAGES, litellm_call_id="c1")
    assert host.calls == []


async def test_prompt_management_group_with_non_dict_prompt_variables_raises_value_error(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[deployment("pm", "p1", model="langfuse/gpt-4o", prompt_id="p", prompt_variables="x")]
    )
    with pytest.raises(ValueError, match=re.escape("Prompt variables is set but not a dictionary. Got=x")):
        await core.acompletion(model="pm", messages=MESSAGES, litellm_call_id="c1")


async def test_prompt_management_group_without_a_call_id_or_logging_object_raises_key_error(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("pm", "p1", model="langfuse/gpt-4o")])
    with pytest.raises(KeyError, match="litellm_call_id"):
        await core.acompletion(model="pm", messages=MESSAGES)


async def test_litellm_agent_group_without_a_logging_object_raises_key_error(make_core: MakeCore, host: EndpointHost):
    """function_setup consumes original_function, so the direct-call path cannot pop it."""
    core: Final = make_core(model_list=[deployment("pm", "p1", model="litellm_agent/openai/gpt-4o")])
    with pytest.raises(KeyError, match="original_function"):
        await core.acompletion(model="pm", messages=MESSAGES, litellm_call_id="c1")
    assert host.calls == []


async def test_litellm_agent_group_calls_the_stripped_model_directly_with_the_deployment_params(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[deployment("pm", "p1", model="litellm_agent/openai/gpt-4o", timeout=9, temperature=0.2)]
    )
    response: Final = await core.acompletion(model="pm", messages=MESSAGES, litellm_logging_obj=_logging_obj())
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id, call.model_group) == ("acompletion", "openai/gpt-4o", "p1", "pm")
    assert (call.kwargs["timeout"], call.kwargs["temperature"], "original_function" in call.kwargs) == (9.0, 0.2, False)
    assert response._hidden_params == {}


async def test_litellm_agent_group_naming_another_group_routes_into_it_with_the_agent_timeout(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(
        model_list=[deployment("pm", "p1", model="litellm_agent/g", timeout=9), deployment("g", "d1")]
    )
    response: Final = await core.acompletion(model="pm", messages=MESSAGES, litellm_logging_obj=_logging_obj())
    call: Final = host.calls[0]
    assert (call.model, call.model_group, call.kwargs["timeout"]) == ("openai/d1", "pm", 9.0)
    # The agent deployment's api_key travels as a request credential, so the target gets a request-scoped id.
    assert call.deployment_id != "d1"
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "g"


async def test_a_group_with_two_prompt_management_deployments_routes_normally(make_core: MakeCore, host: ScriptedHost):
    core: Final = make_core(
        model_list=[deployment("pm", "p1", model="langfuse/gpt-4o"), deployment("pm", "p2", model="langfuse/gpt-4o")]
    )
    await core.acompletion(model="pm", messages=MESSAGES)
    call: Final = host.calls[0]
    assert (call.op, call.model, call.model_group) == ("acompletion", "langfuse/gpt-4o", "pm")
