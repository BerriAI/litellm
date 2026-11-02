"""
Characterization of the generic endpoints (Responses API and its get/delete/cancel/compact/input
items, Google GenAI, OCR, search, decisions, video, containers, skills, interactions, managed
agents, realtime, fine-tuning, files, image edit, LLM pass-through, vector stores, assistants,
guardrails): which host op each reaches, the kwargs the core builds, and how an id in the request
pins the deployment.

Some of these endpoints are served by host ops named by a dotted path under ``litellm``; the
``host`` fixture replaces those litellm functions with recording stand-ins (see
``endpoints_helpers.py``), so every call shows up in ``host.calls``.
"""

import re
import inspect
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_core import RouterCore
from litellm.types.videos.utils import encode_video_id_with_provider
from tests.unit.test_router.core.endpoints_helpers import EndpointHost, patch_direct_endpoints
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment

MESSAGES: Final = [{"role": "user", "content": "hi"}]
MakeCore = Callable[..., RouterCore]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> EndpointHost:
    endpoint_host: Final = EndpointHost()
    patch_direct_endpoints(monkeypatch, endpoint_host)
    return endpoint_host


async def _call(core: RouterCore, endpoint: str, request: Mapping[str, object]) -> object:
    result: Final = getattr(core, endpoint)(**request)
    return await result if inspect.isawaitable(result) else result


def _bucket(kwargs: Mapping[str, object], name: str) -> Mapping[str, object]:
    bucket: Final = kwargs[name]
    assert isinstance(bucket, Mapping)
    return bucket


# endpoint -> (recorded op, request without model)
ROUTED_ASYNC: Final[dict[str, tuple[str, dict[str, object]]]] = {
    "aresponses": ("aresponses", {"input": "hi"}),
    "agenerate_content": ("agenerate_content", {"contents": []}),
    "agenerate_content_stream": ("agenerate_content_stream", {"contents": []}),
    "afile_delete": ("afile_delete", {"file_id": "f1"}),
    "afile_content": ("afile_content", {"file_id": "f1"}),
    "afile_list": ("afile_list", {}),
    "aget_responses": ("aget_responses", {"response_id": "resp_plain"}),
    "acancel_responses": ("acancel_responses", {"response_id": "resp_plain"}),
    "acompact_responses": ("acompact_responses", {"response_id": "resp_plain"}),
    "adelete_responses": ("adelete_responses", {"response_id": "resp_plain"}),
    "alist_input_items": ("alist_input_items", {"response_id": "resp_plain"}),
    "acreate_realtime_client_secret": ("acreate_realtime_client_secret", {}),
    "arealtime_calls": ("arealtime_calls", {}),
    "acreate_realtime_transcription_session": ("acreate_realtime_transcription_session", {}),
    "acreate_fine_tuning_job": ("acreate_fine_tuning_job", {"training_file": "f"}),
    "acancel_fine_tuning_job": ("acancel_fine_tuning_job", {"fine_tuning_job_id": "j"}),
    "alist_fine_tuning_jobs": ("alist_fine_tuning_jobs", {}),
    "aretrieve_fine_tuning_job": ("aretrieve_fine_tuning_job", {"fine_tuning_job_id": "j"}),
    "aimage_edit": ("aimage_edit", {"image": b"x", "prompt": "p"}),
    "allm_passthrough_route": ("allm_passthrough_route", {"method": "POST", "endpoint": "/v1/x"}),
    "aocr": ("aocr", {"document": {}}),
    "adecisions": ("adecisions", {}),
    "asystemone": ("asystemone", {}),
    "aextraction": ("aextraction", {}),
    "avideo_generation": ("avideo_generation", {"prompt": "p"}),
    "avideo_list": ("avideo_list", {}),
    "avideo_status": ("avideo_status", {"video_id": "v"}),
    "avideo_content": ("avideo_content", {"video_id": "v"}),
    "avideo_remix": ("avideo_remix", {"video_id": "v", "prompt": "p"}),
    "avideo_create_character": ("avideo_create_character", {}),
    "avideo_get_character": ("avideo_get_character", {}),
    "avideo_edit": ("avideo_edit", {"video_id": "v"}),
    "avideo_extension": ("avideo_extension", {"video_id": "v"}),
    "acreate_container": ("acreate_container", {"name": "c"}),
    "alist_containers": ("alist_containers", {}),
    "aretrieve_container": ("aretrieve_container", {"container_id": "cntr_native"}),
    "adelete_container": ("adelete_container", {"container_id": "cntr_native"}),
    "aupload_container_file": ("aupload_container_file", {"container_id": "cntr_native", "file": b"x"}),
    "alist_container_files": ("alist_container_files", {"container_id": "cntr_native"}),
    "aretrieve_container_file": ("aretrieve_container_file", {"container_id": "cntr_native", "file_id": "f"}),
    "adelete_container_file": ("adelete_container_file", {"container_id": "cntr_native", "file_id": "f"}),
    "aretrieve_container_file_content": (
        "aretrieve_container_file_content",
        {"container_id": "cntr_native", "file_id": "f"},
    ),
    "acreate_skill": ("acreate_skill", {}),
    "alist_skills": ("alist_skills", {}),
    "aget_skill": ("aget_skill", {"skill_id": "s"}),
    "adelete_skill": ("adelete_skill", {"skill_id": "s"}),
    "acreate_interaction": ("interactions.acreate", {"input": "x"}),
    "avector_store_search": ("vs.asearch", {"vector_store_id": "v", "query": "q"}),
    "avector_store_retrieve": ("vs.aretrieve", {"vector_store_id": "v"}),
    "avector_store_list": ("vs.alist", {}),
    "avector_store_update": ("vs.aupdate", {"vector_store_id": "v"}),
    "avector_store_delete": ("vs.adelete", {"vector_store_id": "v"}),
    "avector_store_file_create": ("vsf.acreate", {"vector_store_id": "v", "file_id": "f"}),
    "avector_store_file_list": ("vsf.alist", {"vector_store_id": "v"}),
    "avector_store_file_retrieve": ("vsf.aretrieve", {"vector_store_id": "v", "file_id": "f"}),
    "avector_store_file_content": ("vsf.aretrieve_content", {"vector_store_id": "v", "file_id": "f"}),
    "avector_store_file_update": ("vsf.aupdate", {"vector_store_id": "v", "file_id": "f"}),
    "avector_store_file_delete": ("vsf.adelete", {"vector_store_id": "v", "file_id": "f"}),
}

ROUTED_SYNC: Final[dict[str, tuple[str, dict[str, object]]]] = {
    "responses": ("responses", {"input": "hi"}),
    "generate_content": ("generate_content", {"contents": []}),
    "generate_content_stream": ("generate_content_stream", {"contents": []}),
    "ocr": ("ocr", {"document": {}}),
    "decisions": ("decisions", {}),
    "systemone": ("systemone", {}),
    "extraction": ("extraction", {}),
    "video_generation": ("video_generation", {"prompt": "p"}),
    "video_list": ("video_list", {}),
    "video_status": ("video_status", {"video_id": "v"}),
    "video_content": ("video_content", {"video_id": "v"}),
    "video_remix": ("video_remix", {"video_id": "v", "prompt": "p"}),
    "create_container": ("create_container", {"name": "c"}),
    "list_containers": ("list_containers", {}),
    "retrieve_container": ("retrieve_container", {"container_id": "c"}),
    "delete_container": ("delete_container", {"container_id": "c"}),
    "vector_store_search": ("vs.search", {"vector_store_id": "v", "query": "q"}),
    "vector_store_create": ("vs.create", {"name": "v"}),
    "vector_store_retrieve": ("vs.retrieve", {"vector_store_id": "v"}),
    "vector_store_list": ("vs.list", {}),
    "vector_store_update": ("vs.update", {"vector_store_id": "v"}),
    "vector_store_delete": ("vs.delete", {"vector_store_id": "v"}),
}


@pytest.mark.parametrize("endpoint", sorted(ROUTED_ASYNC))
async def test_async_generic_endpoint_routes_to_the_group_deployment_with_litellm_metadata(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    op, request = ROUTED_ASYNC[endpoint]
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    response: Final = await _call(core, endpoint, {"model": "g", **request})
    assert [call.op for call in host.calls] == [op]
    call: Final = host.calls[0]
    assert (call.model, call.deployment_id, call.kwargs["api_base"], call.kwargs["custom_llm_provider"]) == (
        "openai/d1",
        "d1",
        "https://d1.example",
        "openai",
    )
    metadata: Final = _bucket(call.kwargs, "litellm_metadata")
    assert (metadata["model_group"], metadata["deployment"]) == ("g", "openai/d1")
    assert "metadata" not in call.kwargs
    assert response._hidden_params["model_id"] == "d1"  # pyright: ignore[reportAttributeAccessIssue]  # the scripted response
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "g"  # pyright: ignore[reportAttributeAccessIssue]  # the scripted response


@pytest.mark.parametrize("endpoint", sorted(ROUTED_SYNC))
async def test_sync_generic_endpoint_routes_to_the_group_deployment_and_returns_the_response(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    op, request = ROUTED_SYNC[endpoint]
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    response: Final = getattr(core, endpoint)(model="g", **request)
    assert not inspect.isawaitable(response)
    assert [call.op for call in host.calls] == [op]
    call: Final = host.calls[0]
    assert (call.model, call.deployment_id, call.kwargs["api_base"], call.kwargs["custom_llm_provider"]) == (
        "openai/d1",
        "d1",
        "https://d1.example",
        "openai",
    )
    assert _bucket(call.kwargs, "litellm_metadata")["model_group"] == "g"
    assert response is not None


@pytest.mark.parametrize("endpoint", ["ocr", "generate_content", "video_generation"])
async def test_sync_generic_endpoints_set_no_routing_hidden_params(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    _, request = ROUTED_SYNC[endpoint]
    core: Final = make_core(model_list=[deployment("g", "d1")])
    response: Final = getattr(core, endpoint)(model="g", **request)
    assert response._hidden_params == {}


@pytest.mark.parametrize("endpoint", ["ocr", "generate_content"])
async def test_sync_generic_endpoints_are_not_retried(make_core: MakeCore, host: EndpointHost, endpoint: str):
    host.answer("openai/d1", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"))
    _, request = ROUTED_SYNC[endpoint]
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=3)
    with pytest.raises(litellm.InternalServerError):
        getattr(core, endpoint)(model="g", **request)
    assert len(host.calls) == 1


async def test_async_generic_endpoint_is_retried_with_the_router_num_retries(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"))
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=2)
    with pytest.raises(litellm.InternalServerError):
        await core.aocr(model="g", document={})
    assert len(host.calls) == 3


@pytest.mark.parametrize(
    ("endpoint", "op"),
    [
        ("aadapter_generate_content", "aadapter_generate_content"),
        ("video_create_character", "video_create_character"),
        ("video_get_character", "video_get_character"),
        ("video_edit", "video_edit"),
        ("video_extension", "video_extension"),
        ("upload_container_file", "upload_container_file"),
        ("list_container_files", "list_container_files"),
        ("retrieve_container_file", "retrieve_container_file"),
        ("delete_container_file", "delete_container_file"),
        ("retrieve_container_file_content", "retrieve_container_file_content"),
        ("get_interaction", "interactions.get"),
        ("delete_interaction", "interactions.delete"),
        ("cancel_interaction", "interactions.cancel"),
    ],
)
async def test_endpoints_that_had_no_dispatch_branch_route_like_their_siblings(
    make_core: MakeCore, host: EndpointHost, endpoint: str, op: str
):
    """The async one routes as the other async generic endpoints do; the sync ones answer
    synchronously, routed as the other sync generic endpoints are."""
    core: Final = make_core(model_list=[deployment("g", "d1")])
    result: Final = getattr(core, endpoint)(model="g", container_id="c", video_id="v", interaction_id="i")
    assert inspect.isawaitable(result) is endpoint.startswith("a")
    response: Final = await result if inspect.isawaitable(result) else result
    assert response is not None
    assert [(call.op, call.deployment_id) for call in host.calls] == [(op, "d1")]


async def test_sync_create_interaction_answers_synchronously_with_the_routed_response(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    response: Final[object] = getattr(core, "create_interaction")(model="g", input="x")
    assert not inspect.isawaitable(response)
    assert response is not None
    assert [(call.op, call.deployment_id) for call in host.calls] == [("interactions.create", "d1")]


async def test_sync_create_interaction_with_an_agent_calls_the_handler_directly_with_gemini(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("ag", "d1")])
    getattr(core, "create_interaction")(agent="ag", model="ag", input="x")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["model"], call.kwargs["custom_llm_provider"]) == (
        "interactions.create",
        None,
        None,
        "gemini",
    )


# --- request params, provider and timeouts ------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["aocr", "ocr"])
async def test_generic_request_params_win_over_deployment_params(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=[deployment("g", "d1", temperature=0.1)])
    await _call(core, endpoint, {"model": "g", "document": {}, "temperature": 0.9})
    assert host.calls[0].kwargs["temperature"] == 0.9


async def test_generic_call_always_sends_the_deployment_model(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.aocr(model="g", document={})
    assert host.calls[0].model == "openai/d1"


async def test_generic_call_sends_the_declared_custom_llm_provider(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", model="my-model", custom_llm_provider="hosted_vllm")])
    await core.aocr(model="g", document={})
    assert (host.calls[0].model, host.calls[0].kwargs["custom_llm_provider"]) == ("my-model", "hosted_vllm")


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [("aocr", 600.0), ("aresponses", 600.0), ("ocr", 600.0), ("generate_content", 600.0), ("responses", 600.0)],
)
async def test_generic_default_timeout_is_the_passthrough_default_for_sync_and_async(
    make_core: MakeCore, host: EndpointHost, endpoint: str, expected: float
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    request: Final = {**(ROUTED_ASYNC.get(endpoint) or ROUTED_SYNC[endpoint])[1], "model": "g"}
    await _call(core, endpoint, request)
    assert host.calls[0].kwargs["timeout"] == expected


@pytest.mark.parametrize("endpoint", ["aocr", "ocr"])
async def test_generic_timeout_takes_the_router_timeout(make_core: MakeCore, host: EndpointHost, endpoint: str):
    core: Final = make_core(model_list=[deployment("g", "d1")], timeout=42)
    await _call(core, endpoint, {"model": "g", "document": {}})
    assert host.calls[0].kwargs["timeout"] == 42


@pytest.mark.parametrize("endpoint", ["aocr", "ocr"])  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]  # pytest and the cores' untyped members
async def test_generic_timeout_without_a_router_timeout_takes_the_host_request_timeout(
    make_core: MakeCore,
    host: EndpointHost,
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,  # pyright: ignore[reportUnknownMemberType, reportUnknownParameterType]  # pytest and the cores' untyped members
):
    host.set_dynamic(request_timeout=33.0)
    monkeypatch.setattr(litellm, "request_timeout", 77.0)  # pyright: ignore[reportUnknownMemberType]  # pytest and the cores' untyped members
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await _call(core, endpoint, {"model": "g", "document": {}})
    assert host.calls[0].kwargs["timeout"] == 33.0


@pytest.mark.parametrize(
    ("endpoint", "request_body", "expected"),
    [
        # A stream-timeout-aware handler (by its endpoint, whatever the host function is named)
        # gets the request timeout and the stream timeout apart.
        ("aresponses", {"input": "hi"}, (11.0, 3.0)),
        ("agenerate_content", {"contents": []}, (3.0, 3.0)),
    ],
)
async def test_streaming_generic_call_sends_the_deployment_stream_timeout(
    make_core: MakeCore,
    host: EndpointHost,
    endpoint: str,
    request_body: dict[str, object],
    expected: tuple[float, float],
):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11, stream_timeout=3)])
    await _call(core, endpoint, {"model": "g", "stream": True, **request_body})
    sent: Final = host.calls[0].kwargs
    assert (sent["timeout"], sent["stream_timeout"]) == expected


async def test_non_streaming_aresponses_sends_the_deployment_timeout(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11, stream_timeout=3)])
    await core.aresponses(model="g", input="hi")
    assert host.calls[0].kwargs["timeout"] == 11.0


async def test_realtime_client_secret_session_model_follows_the_routed_deployment(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.acreate_realtime_client_secret(model="g", session={"model": "g", "type": "realtime"})
    assert host.calls[0].kwargs["session"] == {"model": "openai/d1", "type": "realtime"}


async def test_generic_endpoint_with_an_unknown_group_raises_without_calling_the_host(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    with pytest.raises(litellm.BadRequestError, match="model=nope"):
        await core.agenerate_content(model="nope", contents=[])
    assert host.calls == []


# --- Responses API ids pin the deployment -----------------------------------------------------

TWO_DEPLOYMENTS: Final = [deployment("g", "d1"), deployment("g", "d2")]
ENCODED_RESPONSE_ID: Final = ResponsesAPIRequestUtils._build_responses_api_response_id("openai", "d2", "resp_abc")  # pyright: ignore[reportPrivateUsage]  # litellm's encoder


@pytest.mark.parametrize(
    "endpoint", ["aget_responses", "acancel_responses", "acompact_responses", "adelete_responses", "alist_input_items"]
)
async def test_a_litellm_response_id_pins_the_deployment_it_encodes(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(5):
        await _call(core, endpoint, {"response_id": ENCODED_RESPONSE_ID})
    assert host.deployments_called() == ["d2"] * 5


async def test_a_litellm_response_id_overrides_the_request_model(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(5):
        await core.adelete_responses(response_id=ENCODED_RESPONSE_ID, model="g")
    assert host.deployments_called() == ["d2"] * 5
    assert _bucket(host.calls[0].kwargs, "litellm_metadata")["model_group"] == "d2"


async def test_the_response_id_reaches_the_host_still_encoded(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.aget_responses(response_id=ENCODED_RESPONSE_ID)
    assert host.calls[0].kwargs["response_id"] == ENCODED_RESPONSE_ID


async def test_a_plain_response_id_without_a_model_calls_the_handler_directly(make_core: MakeCore, host: EndpointHost):
    """As litellm's own function does without a router: no deployment, the given provider."""
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await _call(core, "aget_responses", {"response_id": "resp_plain", "custom_llm_provider": "azure"})
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["response_id"], call.kwargs["custom_llm_provider"]) == (
        "aget_responses",
        None,
        "resp_plain",
        "azure",
    )
    assert "model" not in call.kwargs
    assert "litellm_metadata" not in call.kwargs


# --- container ids ------------------------------------------------------------------------

ENCODED_CONTAINER_ID: Final = ResponsesAPIRequestUtils.build_container_id("azure", "d2", "cntr_upstream")


async def test_a_litellm_container_id_pins_its_deployment_and_is_sent_decoded(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(5):
        await core.aretrieve_container(container_id=ENCODED_CONTAINER_ID)
    assert host.deployments_called() == ["d2"] * 5
    assert {call.kwargs["container_id"] for call in host.calls} == {"cntr_upstream"}


async def test_a_routed_container_call_sends_the_deployment_provider_not_the_decoded_one(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.aretrieve_container(container_id=ENCODED_CONTAINER_ID, custom_llm_provider="openai")
    assert host.calls[0].kwargs["custom_llm_provider"] == "openai"


async def test_a_native_container_id_routes_by_the_forwarded_model_id(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(5):
        await core.aretrieve_container(container_id="cntr_native", model_id=" d1 ")
    assert host.deployments_called() == ["d1"] * 5
    assert host.calls[0].kwargs["container_id"] == "cntr_native"


async def test_a_native_container_id_without_a_model_calls_the_handler_directly(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.aretrieve_container(container_id="cntr_native", custom_llm_provider="azure")
    call: Final = host.calls[0]
    assert (call.deployment_id, call.kwargs["container_id"], call.kwargs["custom_llm_provider"]) == (
        None,
        "cntr_native",
        "azure",
    )
    assert "litellm_metadata" not in call.kwargs


async def test_create_container_with_an_unknown_model_passes_through_with_that_model(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.acreate_container(name="c", model="not-a-group")
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id) == ("acreate_container", "not-a-group", None)


async def test_create_container_without_a_model_calls_the_handler_directly(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.acreate_container(name="c")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, "model" in call.kwargs) == ("acreate_container", None, False)


# --- video ids -----------------------------------------------------------------------------

ENCODED_VIDEO_ID: Final = encode_video_id_with_provider("video_up", "openai", model_id="d2", deployment_id="d2")


@pytest.mark.parametrize(
    "endpoint", ["avideo_status", "avideo_content", "avideo_remix", "avideo_edit", "avideo_extension"]
)
async def test_a_litellm_video_id_pins_the_deployment_that_created_it(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(6):
        await _call(core, endpoint, {"model": "g", "video_id": ENCODED_VIDEO_ID, "prompt": "p"})
    assert host.deployments_called() == ["d2"] * 6


async def test_a_video_id_on_a_video_generation_call_does_not_pin(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(30):
        await core.avideo_generation(model="g", prompt="p", video_id=ENCODED_VIDEO_ID)
    assert set(host.deployments_called()) == {"d1", "d2"}


async def test_sync_video_status_pins_the_deployment_of_the_video_id(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    for _ in range(6):
        core.video_status(model="g", video_id=ENCODED_VIDEO_ID)
    assert host.deployments_called() == ["d2"] * 6


# --- LLM pass-through ----------------------------------------------------------------------


async def test_llm_passthrough_with_an_unknown_model_calls_the_handler_with_that_model(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.allm_passthrough_route(model="not-a-group", method="POST", endpoint="/model/not-a-group/invoke")
    call: Final = host.calls[0]
    assert (call.model, call.deployment_id, call.kwargs["endpoint"]) == (
        "not-a-group",
        None,
        "/model/not-a-group/invoke",
    )


async def test_llm_passthrough_rewrites_the_group_name_in_the_endpoint_to_the_provider_model(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", model="bedrock/us.anthropic.claude")])
    await core.allm_passthrough_route(model="g", method="POST", endpoint="/model/g/invoke")
    call: Final = host.calls[0]
    assert (call.model, call.kwargs["endpoint"], call.kwargs["custom_llm_provider"]) == (
        "bedrock/us.anthropic.claude",
        "/model/us.anthropic.claude/invoke",
        "bedrock",
    )


async def test_llm_passthrough_forwards_the_request_client(make_core: MakeCore, host: EndpointHost):
    client: Final = object()
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.allm_passthrough_route(model="g", method="POST", endpoint="/v1/x", client=client)
    assert host.calls[0].kwargs["client"] is client


# --- interactions and managed agents ----------------------------------------------------------


async def test_interactions_get_without_a_model_calls_the_handler_with_the_given_provider(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.aget_interaction(interaction_id="i", custom_llm_provider="vertex_ai")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["custom_llm_provider"]) == ("interactions.aget", None, "vertex_ai")


@pytest.mark.parametrize("endpoint", ["aget_interaction", "adelete_interaction", "acancel_interaction"])
async def test_interactions_default_to_the_gemini_provider(make_core: MakeCore, host: EndpointHost, endpoint: str):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await _call(core, endpoint, {"interaction_id": "i"})
    assert host.calls[0].kwargs["custom_llm_provider"] == "gemini"


async def test_interaction_with_an_agent_named_as_the_model_is_not_routed(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("ag", "d1")])
    await core.acreate_interaction(agent="ag", model="ag", input="x")
    call: Final = host.calls[0]
    assert (call.deployment_id, call.kwargs["model"], call.kwargs["custom_llm_provider"]) == (None, None, "gemini")


@pytest.mark.parametrize(
    ("endpoint", "op"),
    [
        ("acreate_agent", "agents.acreate"),
        ("alist_agents", "agents.alist"),
        ("aget_agent", "agents.aget"),
        ("adelete_agent", "agents.adelete"),
        ("alist_agent_versions", "agents.alist_versions"),
        ("create_agent", "agents.create"),
        ("list_agents", "agents.list"),
        ("get_agent", "agents.get"),
        ("delete_agent", "agents.delete"),
        ("list_agent_versions", "agents.list_versions"),
    ],
)
async def test_managed_agent_endpoints_call_the_handler_directly_with_gemini_by_default(
    make_core: MakeCore, host: EndpointHost, endpoint: str, op: str
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await _call(core, endpoint, {"name": "a", "model": "g"})
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["custom_llm_provider"], call.kwargs["model"]) == (
        op,
        None,
        "gemini",
        "g",
    )


@pytest.mark.parametrize("endpoint", ["acreate_agent", "create_agent"])
async def test_managed_agent_endpoints_keep_the_given_provider(make_core: MakeCore, host: EndpointHost, endpoint: str):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await _call(core, endpoint, {"name": "a", "custom_llm_provider": "vertex_ai"})
    assert host.calls[0].kwargs["custom_llm_provider"] == "vertex_ai"


# --- assistants ----------------------------------------------------------------------------

ASSISTANT_ENDPOINTS: Final = (
    "acreate_assistants",
    "adelete_assistant",
    "aget_assistants",
    "acreate_thread",
    "aget_thread",
    "a_add_message",
    "aget_messages",
    "arun_thread",
)


@pytest.mark.parametrize("endpoint", ASSISTANT_ENDPOINTS)
async def test_assistants_endpoints_without_a_provider_or_config_raise(
    make_core: MakeCore, host: EndpointHost, endpoint: str
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    with pytest.raises(Exception, match=re.escape("'custom_llm_provider' must be set. Either via:")):
        await getattr(core, endpoint)()
    assert host.calls == []


@pytest.mark.parametrize("endpoint", ASSISTANT_ENDPOINTS)
async def test_assistants_endpoints_use_the_assistants_config(make_core: MakeCore, host: EndpointHost, endpoint: str):
    core: Final = make_core(
        model_list=[deployment("g", "d1")],
        assistants_config={
            "custom_llm_provider": "azure",
            "litellm_params": {"api_key": "k", "api_base": "https://az"},
        },
    )
    await getattr(core, endpoint)(thread_id="t")
    call: Final = host.calls[0]
    assert (call.op, call.kwargs["custom_llm_provider"], call.kwargs["api_key"], call.kwargs["api_base"]) == (
        endpoint,
        "azure",
        "k",
        "https://az",
    )
    assert (call.kwargs["client"], call.kwargs["thread_id"]) == (None, "t")


async def test_assistants_endpoint_with_an_explicit_provider_ignores_the_config(
    make_core: MakeCore, host: EndpointHost
):
    client: Final = object()
    core: Final = make_core(
        model_list=[deployment("g", "d1")],
        assistants_config={"custom_llm_provider": "azure", "litellm_params": {"api_key": "k"}},
    )
    await core.arun_thread(custom_llm_provider="openai", client=client, thread_id="t")
    call: Final = host.calls[0]
    assert (call.kwargs["custom_llm_provider"], call.kwargs["client"], "api_key" in call.kwargs) == (
        "openai",
        client,
        False,
    )


# --- vector stores ---------------------------------------------------------------------------


async def test_vector_store_search_without_a_model_gets_the_router_and_overrides_a_client_router(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.avector_store_search(vector_store_id="v", query="q", router="client-value")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["router"]) == ("vs.asearch", None, core)
    assert "_direct_vector_store_embedding_executor" in call.kwargs


async def test_sync_vector_store_search_without_a_model_gets_the_router(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    core.vector_store_search(vector_store_id="v", query="q")
    assert host.calls[0].kwargs["router"] is core


async def test_routed_vector_store_search_gets_the_embedding_executor_but_not_the_router(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.avector_store_search(model="g", vector_store_id="v", query="q")
    sent: Final = host.calls[0].kwargs
    assert ("router" in sent, "_direct_vector_store_embedding_executor" in sent) == (False, True)


async def test_vector_store_retrieve_without_a_model_calls_the_handler_directly(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    await core.avector_store_retrieve(vector_store_id="v", custom_llm_provider="openai")
    call: Final = host.calls[0]
    assert (call.deployment_id, call.kwargs["custom_llm_provider"], "router" in call.kwargs) == (None, "openai", False)


async def test_sync_vector_store_create_without_a_model_passes_the_provider(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=TWO_DEPLOYMENTS)
    core.vector_store_create(name="v", custom_llm_provider="openai")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id, call.kwargs["custom_llm_provider"]) == ("vs.create", None, "openai")


@pytest.mark.parametrize(
    ("endpoint", "op"),
    [
        ("vector_store_file_create", "vsf.create"),
        ("vector_store_file_list", "vsf.list"),
        ("vector_store_file_retrieve", "vsf.retrieve"),
        ("vector_store_file_content", "vsf.retrieve_content"),
        ("vector_store_file_update", "vsf.update"),
        ("vector_store_file_delete", "vsf.delete"),
    ],
)
async def test_sync_vector_store_file_endpoints_never_route_even_with_a_model(
    make_core: MakeCore, host: EndpointHost, endpoint: str, op: str
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    getattr(core, endpoint)(model="g", vector_store_id="v", file_id="f")
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id, call.kwargs["custom_llm_provider"], call.kwargs["client"]) == (
        op,
        "g",
        None,
        None,
        None,
    )


async def test_avector_store_create_with_a_model_routes_with_metadata_and_no_model_group(
    make_core: MakeCore, host: EndpointHost
):
    core: Final = make_core(model_list=[deployment("g", "d1", api_base="https://d1.example")])
    await core.avector_store_create(model="g", name="n")
    call: Final = host.calls[0]
    assert (call.op, call.model, call.deployment_id, call.kwargs["custom_llm_provider"]) == (
        "vs.acreate",
        "openai/d1",
        "d1",
        "openai",
    )
    metadata: Final = _bucket(call.kwargs, "metadata")
    assert (metadata["deployment"], "model_group" in metadata) == ("openai/d1", False)


async def test_avector_store_create_without_a_model_calls_the_handler_directly(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await core.avector_store_create(model=None, name="n")
    call: Final = host.calls[0]
    assert (call.op, call.deployment_id) == ("vs.acreate", None)


async def test_avector_store_create_is_not_retried(make_core: MakeCore, host: EndpointHost):
    host.answer("openai/d1", litellm.InternalServerError(message="boom", llm_provider="openai", model="d1"))
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=3)
    with pytest.raises(litellm.InternalServerError):
        await core.avector_store_create(model="g", name="n")
    assert len(host.calls) == 1


# --- search tools ---------------------------------------------------------------------------

SEARCH_TOOLS: Final = [{"search_tool_name": "s", "litellm_params": {"search_provider": "tavily", "api_key": "tk"}}]


async def test_asearch_calls_the_named_search_tool_with_its_litellm_params(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], search_tools=SEARCH_TOOLS)
    response: Final = await core.asearch(model="s", query="q")
    call: Final = host.calls[0]
    assert (call.op, call.kwargs["search_provider"], call.kwargs["api_key"], call.kwargs["query"]) == (
        "asearch",
        "tavily",
        "tk",
        "q",
    )
    assert _bucket(call.kwargs, "litellm_metadata")["model_group"] == "s"
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "s"  # pyright: ignore[reportAttributeAccessIssue]  # the scripted response


async def test_sync_search_calls_the_named_search_tool_like_asearch(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], search_tools=SEARCH_TOOLS)
    response: Final[object] = getattr(core, "search")(model="s", query="q")
    assert not inspect.isawaitable(response)
    call: Final = host.calls[0]
    assert (call.op, call.kwargs["search_provider"], call.kwargs["api_key"], call.kwargs["query"]) == (
        "search",
        "tavily",
        "tk",
        "q",
    )
    assert _bucket(call.kwargs, "litellm_metadata")["model_group"] == "s"
    assert "model" not in call.kwargs


async def test_sync_search_with_an_unknown_tool_raises_value_error(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], search_tools=SEARCH_TOOLS)
    with pytest.raises(ValueError, match=re.escape("Search tool 'g' not found in router.search_tools")):
        getattr(core, "search")(search_tool_name="g", query="q")
    assert host.calls == []


async def test_asearch_with_an_unknown_tool_raises_value_error(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], search_tools=SEARCH_TOOLS, num_retries=0)
    with pytest.raises(ValueError, match=re.escape("Search tool 'g' not found in router.search_tools")):
        await core.asearch(model="g", query="q")
    assert host.calls == []


# --- guardrails ---------------------------------------------------------------------------

GUARDRAILS: Final = [{"guardrail_name": "gr", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}}]


async def test_aguardrail_runs_the_function_with_the_selected_guardrail(make_core: MakeCore, host: ScriptedHost):
    seen: Final[list[Mapping[str, object]]] = []

    async def guard(**kwargs: object) -> str:  # kwargs-ok: the guardrail's keyword payload
        seen.append(kwargs)
        return "verdict"

    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=GUARDRAILS)
    assert await core.aguardrail(guardrail_name="gr", original_function=guard, data={"x": 1}) == "verdict"
    assert sorted(seen[0]) == ["data", "litellm_metadata", "litellm_trace_id", "selected_guardrail"]
    assert seen[0]["selected_guardrail"] == GUARDRAILS[0]
    assert _bucket(seen[0], "litellm_metadata")["model_group"] == "gr"
    assert host.calls == []


async def test_aguardrail_with_an_unknown_name_raises_value_error(make_core: MakeCore, host: ScriptedHost):
    async def guard(**_kwargs: object) -> str:  # kwargs-ok: the guardrail's keyword payload
        return "verdict"

    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=GUARDRAILS, num_retries=0)
    with pytest.raises(ValueError, match="No guardrail found with name: nope"):
        await core.aguardrail(guardrail_name="nope", original_function=guard)


async def test_aguardrail_does_not_retry_a_failing_guardrail_function(make_core: MakeCore, host: ScriptedHost):
    """A guardrail name is no model group (model_group_size 0), so the router num_retries do not apply."""
    attempts: Final[list[int]] = []

    async def guard(**_kwargs: object) -> str:  # kwargs-ok: the guardrail's keyword payload
        attempts.append(1)
        raise litellm.InternalServerError(message="boom", llm_provider="", model="gr")

    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=GUARDRAILS, num_retries=2)
    with pytest.raises(litellm.InternalServerError, match="boom"):
        await core.aguardrail(guardrail_name="gr", original_function=guard)
    assert len(attempts) == 1
