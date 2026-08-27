"""Tests for the OTel v2 sources of truth: span registry, semconv keys, config,
and the typed StandardLoggingPayload adapter. These need no OTel SDK."""

import logging
import re
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.integrations.otel import (
    BAGGAGE_PROMOTED_KEYS,
    DB,
    Error,
    GenAI,
    GenAIOperation,
    GenAIOutputType,
    HTTP,
    LiteLLM,
    OpenTelemetryV2Config,
    Server,
    is_otel_v2_enabled,
    promoted_baggage,
    resolve_operation,
    resolve_output_type,
    resolve_provider,
)
from litellm.integrations.otel.mappers.genai import GenAIMapper
from litellm.integrations.otel.model import spans as spans_mod
from litellm.integrations.otel.model.payloads import (
    LLMCallSpanData,
    RequestIdentity,
    _upstream_address_port,
)
from litellm.integrations.otel.model.spans import (
    SPAN_REGISTRY,
    LiteLLMSpanKind,
    SpanRole,
    child_roles,
    root_roles,
    validate_registry,
)


@pytest.fixture(autouse=True)
def _clear_otel_v2_flag_cache():
    is_otel_v2_enabled.cache_clear()
    yield
    is_otel_v2_enabled.cache_clear()


def _sample_payload(**overrides):
    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "stream": False,
        "model_parameters": {
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["STOP"],
            "seed": 42,
        },
        "response": {
            "id": "resp_1",
            "model": "gpt-4o-2024",
            "choices": [{"finish_reason": "stop"}],
        },
        "metadata": {
            "team_id": "t1",
            "team_alias": "team one",
            "user_api_key_hash": "hsh",
            "user_api_key_org_id": "org1",
        },
        "api_base": "https://api.openai.com:443/v1",
        "status": "success",
        "litellm_call_id": "call_1",
        "end_user": "u1",
        "response_cost": 0.002,
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


# --- span registry (source of truth #2) ------------------------------------- #


def test_registry_validates_and_is_complete():
    validate_registry()  # raises on inconsistency
    assert set(SPAN_REGISTRY) == set(SpanRole)


def test_registry_parent_integrity_no_orphans():
    for role, spec in SPAN_REGISTRY.items():
        assert spec.role is role
        if spec.parent is not None:
            assert spec.parent in SPAN_REGISTRY


def test_registry_hierarchy_shape():
    # MCP roles have no in-process parent: per the MCP semconv they root (or adopt
    # the client's propagated _meta context), so they sit alongside PROXY_REQUEST.
    assert set(root_roles()) == {
        SpanRole.PROXY_REQUEST,
        SpanRole.MCP_TOOL_CALL,
        SpanRole.MCP_LIST_TOOLS,
    }
    # Guardrails parent to the request span, not the LLM call: a pre-call
    # guardrail runs before the LLM call exists, so it's a sibling of it.
    assert set(child_roles(SpanRole.PROXY_REQUEST)) == {
        SpanRole.LLM_CALL,
        SpanRole.GUARDRAIL,
        SpanRole.DB_CALL,
        SpanRole.SERVICE,
    }
    assert SPAN_REGISTRY[SpanRole.LLM_CALL].kind is LiteLLMSpanKind.CLIENT
    # The proxy is an MCP client to the upstream tool server: CLIENT span. Listing
    # tools is the same client relationship, so it's a CLIENT span too.
    assert SPAN_REGISTRY[SpanRole.MCP_TOOL_CALL].kind is LiteLLMSpanKind.CLIENT
    assert SPAN_REGISTRY[SpanRole.MCP_LIST_TOOLS].kind is LiteLLMSpanKind.CLIENT
    # MCP spans don't nest under the transport: they link the PROXY_REQUEST span
    # instead of parenting to it (OTel GenAI MCP semconv).
    assert SPAN_REGISTRY[SpanRole.MCP_TOOL_CALL].parent is None
    assert SPAN_REGISTRY[SpanRole.MCP_LIST_TOOLS].parent is None
    assert SPAN_REGISTRY[SpanRole.MCP_TOOL_CALL].links is SpanRole.PROXY_REQUEST
    assert SPAN_REGISTRY[SpanRole.MCP_LIST_TOOLS].links is SpanRole.PROXY_REQUEST
    assert SPAN_REGISTRY[SpanRole.PROXY_REQUEST].kind is LiteLLMSpanKind.SERVER
    assert SPAN_REGISTRY[SpanRole.GUARDRAIL].parent is SpanRole.PROXY_REQUEST
    # An outbound datastore call is a CLIENT span; an internal service is INTERNAL.
    assert SPAN_REGISTRY[SpanRole.DB_CALL].kind is LiteLLMSpanKind.CLIENT
    assert SPAN_REGISTRY[SpanRole.SERVICE].kind is LiteLLMSpanKind.INTERNAL


def test_llm_call_span_name():
    data = LLMCallSpanData.from_standard_logging_payload(_sample_payload())
    assert spans_mod.llm_call_span_name(data) == "chat gpt-4o"


# --- semconv (source of truth #1) ------------------------------------------- #


def _all_constants(cls):
    return {
        getattr(cls, name)
        for name in vars(cls)
        if not name.startswith("__") and isinstance(getattr(cls, name), str)
    }


def test_attribute_keys_are_unique_across_namespaces():
    from litellm.integrations.otel import MCP, Client, JsonRpc, LiteLLMError, Network

    # prefixes are allowed to be substrings; exact keys must not collide.
    exact = set()
    for cls in (GenAI, Error, LiteLLMError, Server, HTTP, DB, MCP, JsonRpc, Network, Client):
        for key in _all_constants(cls):
            assert key not in exact, f"duplicate attribute key {key}"
            exact.add(key)


def test_mcp_attribute_vocabulary_is_complete():
    """Every span-attribute key the OTel GenAI MCP semconv defines has a constant.

    Pins the vocabulary so a dropped or renamed key fails here rather than
    silently emitting a non-conformant attribute name.
    """
    from litellm.integrations.otel import MCP, Client, JsonRpc, Network

    defined = set()
    for cls in (GenAI, Error, Server, MCP, JsonRpc, Network, Client):
        defined |= _all_constants(cls)
    required = {
        "mcp.method.name",
        "mcp.session.id",
        "mcp.protocol.version",
        "mcp.resource.uri",
        "jsonrpc.request.id",
        "jsonrpc.protocol.version",
        "rpc.system",
        "rpc.response.status_code",
        "gen_ai.operation.name",
        "gen_ai.tool.name",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.prompt.name",
        "error.type",
        "server.address",
        "server.port",
        "client.address",
        "client.port",
        "network.protocol.name",
        "network.protocol.version",
        "network.transport",
    }
    assert required <= defined, f"missing MCP semconv keys: {required - defined}"


def test_provider_resolution():
    assert resolve_provider("openai") == "openai"
    assert resolve_provider("bedrock") == "aws.bedrock"
    assert resolve_provider("vertex_ai") == "gcp.vertex_ai"
    # unknown providers pass through verbatim (semconv allows provider-specific)
    assert resolve_provider("my_custom_llm") == "my_custom_llm"
    assert resolve_provider(None) == ""


def test_operation_resolution():
    assert resolve_operation("acompletion") is GenAIOperation.CHAT
    assert resolve_operation("aembedding") is GenAIOperation.EMBEDDINGS
    assert resolve_operation("atext_completion") is GenAIOperation.TEXT_COMPLETION
    assert resolve_operation(None) is GenAIOperation.CHAT
    # An MCP tool call is an ``execute_tool`` operation, not a chat completion.
    assert resolve_operation("call_mcp_tool") is GenAIOperation.EXECUTE_TOOL


@pytest.mark.parametrize("call_type", ["vector_store_search", "avector_store_search"])
def test_vector_store_search_is_a_retrieval_operation(call_type):
    """A vector-store search is a retrieval, so its duration and cost must not
    land in the chat series that dashboards read latency off."""
    assert resolve_operation(call_type) is GenAIOperation.RETRIEVAL
    assert resolve_operation(call_type).value == "retrieval"


@pytest.mark.parametrize("call_type", ["query", "aquery"])
def test_rag_query_is_a_retrieval_operation(call_type):
    """``/rag/query`` reaches the same recorder as a vector-store search and is the
    same operation, so it must not be the one retrieval surface left reading as chat."""
    assert resolve_operation(call_type) is GenAIOperation.RETRIEVAL


@pytest.mark.parametrize(
    "call_type",
    [
        f"{prefix}vector_store_{verb}"
        for verb in ("create", "retrieve", "list", "update", "delete")
        for prefix in ("", "a")
    ],
)
def test_vector_store_management_is_not_chat(call_type):
    """The store lifecycle calls are not GenAI client operations and the convention
    names nothing for them, so they take a vendor value rather than defaulting into
    the chat series."""
    assert resolve_operation(call_type) is GenAIOperation.LITELLM_VECTOR_STORE_MANAGEMENT
    assert resolve_operation(call_type).value == "litellm.vector_store_management"


@pytest.mark.parametrize(
    "call_type",
    [
        f"{prefix}vector_store_file_{verb}"
        for verb in ("create", "list", "retrieve", "content", "update", "delete")
        for prefix in ("", "a")
    ],
)
def test_vector_store_file_management_is_not_chat(call_type):
    """The file operations are a distinct REST resource from the store lifecycle, so
    they get their own vendor value instead of sharing one bucket."""
    assert resolve_operation(call_type) is GenAIOperation.LITELLM_VECTOR_STORE_FILE_MANAGEMENT
    assert resolve_operation(call_type).value == "litellm.vector_store_file_management"


@pytest.mark.parametrize(
    "call_type",
    [
        f"{prefix}{operation}"
        for operation in ("get_responses", "delete_responses", "cancel_responses", "list_input_items")
        for prefix in ("", "a")
    ],
)
def test_responses_management_is_not_chat(call_type):
    """Fetching, deleting or cancelling a stored response runs no inference, so it must not
    read as a chat completion: the retrieved object replays the original call's tokens and
    would inflate the chat series on every read. Regression test for LIT-5602."""
    assert resolve_operation(call_type) is GenAIOperation.LITELLM_RESPONSES_MANAGEMENT
    assert resolve_operation(call_type).value == "litellm.responses_management"


def test_creating_a_response_is_still_chat():
    """Guards the test above: ``/v1/responses`` itself is a chat completion."""
    assert resolve_operation("aresponses") is GenAIOperation.CHAT


_NON_CHAT_ROUTES: Final = (
    ("image_generation", GenAIOperation.GENERATE_CONTENT, GenAIOutputType.IMAGE),
    ("speech", GenAIOperation.GENERATE_CONTENT, GenAIOutputType.SPEECH),
    ("transcription", GenAIOperation.GENERATE_CONTENT, GenAIOutputType.TEXT),
    ("ocr", GenAIOperation.GENERATE_CONTENT, GenAIOutputType.TEXT),
    ("moderation", GenAIOperation.LITELLM_MODERATION, None),
)


@pytest.mark.parametrize(
    ("call_type", "operation", "output_type"),
    [
        (f"{prefix}{call_type}", operation, output_type)
        for call_type, operation, output_type in _NON_CHAT_ROUTES
        for prefix in ("", "a")
    ],
)
def test_non_chat_inference_routes_follow_genai_semconv(call_type, operation, output_type):
    """Image generation, speech, transcription and OCR all produce content, so the
    convention names them ``generate_content`` and separates them by the requested
    output modality rather than by an invented operation. Moderation classifies
    instead of generating and the convention names nothing for it, so it keeps a
    vendor value. Either way the spans must not land in the chat series a dashboard
    reads."""
    assert resolve_operation(call_type) is operation
    assert resolve_output_type(call_type) is output_type


@pytest.mark.parametrize(
    ("call_type", "operation", "output_type"),
    [(f"a{call_type}", operation, output_type) for call_type, operation, output_type in _NON_CHAT_ROUTES],
)
def test_non_chat_route_spans_carry_semconv_name_and_modality(call_type, operation, output_type):
    """The emitted span, not just the mapping table: name is
    ``{gen_ai.operation.name} {gen_ai.request.model}``, the modality rides
    ``gen_ai.output.type``, and the route stays recoverable from
    ``litellm.call_type`` now that several routes share one operation."""
    data = LLMCallSpanData.from_standard_logging_payload(
        _sample_payload(call_type=call_type, model="some-model", custom_llm_provider="openai")
    )
    attrs = GenAIMapper().map(data)

    assert spans_mod.llm_call_span_name(data) == f"{operation.value} some-model"
    assert attrs[GenAI.OPERATION_NAME] == operation.value
    assert attrs[GenAI.PROVIDER_NAME] == "openai"
    assert attrs[GenAI.REQUEST_MODEL] == "some-model"
    assert attrs[LiteLLM.CALL_TYPE] == call_type
    assert attrs.get(GenAI.OUTPUT_TYPE) == (output_type.value if output_type else None)


def test_non_chat_route_error_span_keeps_error_attributes():
    """Modality mapping must not cost the failure signal: a failed non-chat call
    still carries the error type alongside the standardized operation."""
    data = LLMCallSpanData.from_standard_logging_payload(
        _sample_payload(
            call_type="aspeech",
            model="tts-1",
            status="failure",
            error_information={"error_class": "BadRequestError"},
        )
    )
    attrs = GenAIMapper().map(data)

    assert attrs[GenAI.OPERATION_NAME] == GenAIOperation.GENERATE_CONTENT.value
    assert attrs[GenAI.OUTPUT_TYPE] == GenAIOutputType.SPEECH.value
    assert attrs[Error.TYPE] == "BadRequestError"


def test_vendor_operation_values_are_namespaced():
    """A vendor value must stay under the ``litellm.`` prefix: an unprefixed invented
    name could collide with a value the convention adds later, silently changing what
    a conformant consumer thinks it is reading."""
    vendor = [op for op in GenAIOperation if op.name.startswith("LITELLM_")]
    assert vendor, "no vendor operation values defined"
    assert all(op.value.startswith("litellm.") for op in vendor)


@pytest.mark.parametrize("call_type", ["send_message", "asend_message", "asend_message_streaming"])
def test_agent_message_is_an_invoke_agent_operation(call_type):
    """An agent (A2A) message send is an agent invocation, not a chat completion.

    The streaming spelling counts: ``_build_streaming_logging_obj`` in
    ``litellm/a2a_protocol/main.py`` stamps ``asend_message_streaming`` on the
    logging object the streaming iterator dispatches success handlers with, so a
    missing entry sends every streamed agent turn into the chat series. There is
    no sync spelling because A2A streaming is async-only.
    """
    assert resolve_operation(call_type) is GenAIOperation.INVOKE_AGENT
    assert resolve_operation(call_type).value == "invoke_agent"


def test_every_call_type_the_a2a_package_stamps_is_an_agent_operation():
    """Pins the map to the call types the A2A code actually stamps on its logging
    objects. A new spelling added there without a map entry fails here instead of
    quietly landing in the chat series, which is how the streaming one was missed."""
    a2a_package = Path(litellm.__file__).parent / "a2a_protocol"
    stamped = {
        call_type
        for source in a2a_package.rglob("*.py")
        for call_type in re.findall(r'call_type="([^"]+)"', source.read_text())
    }
    assert stamped, "no call_type literals found in litellm/a2a_protocol"
    unmapped = {
        call_type: resolve_operation(call_type).value
        for call_type in stamped
        if resolve_operation(call_type) is not GenAIOperation.INVOKE_AGENT
    }
    assert not unmapped, f"add these to _OPERATION_BY_CALL_TYPE: {unmapped}"


def test_unmapped_call_type_falls_back_to_chat_loudly(caplog):
    """The fallback still labels the series ``chat`` so it is never unlabelled,
    but it says so at debug: a silent default is how retrieval and agent calls
    ended up in the chat charts in the first place."""
    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        assert resolve_operation("some_future_call_type") is GenAIOperation.CHAT
    assert any("some_future_call_type" in record.getMessage() for record in caplog.records)


# --- MCP tool-call (source of truth #1/#2/#3) ------------------------------- #


def _mcp_payload(capture=False, **overrides):
    payload = {
        "call_type": "call_mcp_tool",
        "status": "success",
        "litellm_call_id": "mcp_call_1",
        "response_cost": 0.01,
        "metadata": {
            "user_api_key_team_id": "t1",
            "mcp_tool_call_metadata": {
                "name": "get_weather",
                "arguments": {"city": "Paris"},
                "result": {"temp_c": 21},
                "mcp_server_name": "weather-mcp",
                "mcp_session_id": "sess-abc123",
            },
        },
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def test_mcp_method_values_match_wire_format():
    from litellm.integrations.otel import MCP, MCPMethod

    assert MCPMethod.TOOLS_CALL.value == "tools/call"
    assert MCPMethod.TOOLS_LIST.value == "tools/list"
    assert MCP.METHOD_NAME == "mcp.method.name"


def test_mcp_tool_call_adapter_extracts_fields():
    from litellm.integrations.otel import MCPToolCallSpanData

    data = MCPToolCallSpanData.from_standard_logging_payload(_mcp_payload())
    assert data.operation is GenAIOperation.EXECUTE_TOOL
    assert data.method == "tools/call"
    assert data.tool_name == "get_weather"
    assert data.server_name == "weather-mcp"
    assert data.session_id == "sess-abc123"
    assert data.response_cost == 0.01
    assert data.identity.call_id == "mcp_call_1"
    assert data.identity.team_id == "t1"
    assert data.error is None


def test_mcp_tool_call_content_gated_off_by_default():
    # Arguments and result are sensitive tool I/O: withheld unless content capture
    # is explicitly enabled, exactly like prompt/response bodies.
    from litellm.integrations.otel import MCPToolCallSpanData

    off = MCPToolCallSpanData.from_standard_logging_payload(_mcp_payload())
    assert off.arguments_json is None and off.result_json is None

    on = MCPToolCallSpanData.from_standard_logging_payload(
        _mcp_payload(), capture_content=True
    )
    assert on.arguments_json is not None and '"Paris"' in on.arguments_json
    assert on.result_json is not None and "21" in on.result_json


def test_mcp_tool_call_failure_path():
    from litellm.integrations.otel import MCPToolCallSpanData

    data = MCPToolCallSpanData.from_standard_logging_payload(
        _mcp_payload(
            status="failure",
            error_information={"error_class": "MCPError", "error_message": "boom"},
        )
    )
    assert data.error is not None
    assert data.error.error_type == "MCPError"
    assert data.error.message == "boom"


def test_is_mcp_tool_call_detection():
    from litellm.integrations.otel import is_mcp_tool_call

    assert is_mcp_tool_call(_mcp_payload()) is True
    # call_type alone is enough even before the gateway stamps its metadata.
    assert is_mcp_tool_call({"call_type": "call_mcp_tool"}) is True
    assert is_mcp_tool_call({"call_type": "acompletion"}) is False
    assert is_mcp_tool_call({}) is False


def test_mcp_tool_call_span_name():
    from litellm.integrations.otel import MCPToolCallSpanData
    from litellm.integrations.otel.model.spans import mcp_tool_call_span_name

    data = MCPToolCallSpanData.from_standard_logging_payload(_mcp_payload())
    assert mcp_tool_call_span_name(data) == "tools/call get_weather"


# --- typed adapter (source of truth #3) ------------------------------------- #


def test_llm_call_adapter_extracts_all_fields():
    data = LLMCallSpanData.from_standard_logging_payload(_sample_payload())
    assert data.operation is GenAIOperation.CHAT
    assert data.provider == "openai"
    assert data.request_model == "gpt-4o"
    assert data.response_model == "gpt-4o-2024"
    assert data.response_id == "resp_1"
    assert data.finish_reasons == ("stop",)
    assert (data.usage.input_tokens, data.usage.output_tokens) == (10, 5)
    assert data.request_params.temperature == 0.7
    assert data.request_params.top_k == 40
    assert data.request_params.stop_sequences == ("STOP",)
    assert data.request_params.seed == 42
    assert data.server is not None
    assert data.server.address == "api.openai.com"
    assert data.server.port == 443
    assert data.response_cost == 0.002
    assert data.error is None
    assert data.identity.team_id == "t1"
    assert data.identity.key_hash == "hsh"


def test_llm_call_adapter_failure_path():
    payload = _sample_payload(
        status="failure",
        error_information={
            "error_class": "RateLimitError",
            "error_message": "429 slow down",
        },
    )
    data = LLMCallSpanData.from_standard_logging_payload(payload)
    assert data.error is not None
    assert data.error.error_type == "RateLimitError"
    assert data.error.message == "429 slow down"


def test_llm_call_adapter_carries_error_detail_fields():
    """``_parse_error`` threads the full detail set from ``error_information``
    (``error_code``, ``traceback``, ``llm_provider``) onto ``SpanError`` so the
    emitter can stamp them as span attributes."""
    payload = _sample_payload(
        status="failure",
        error_information={
            "error_class": "BadRequestError",
            "error_message": "400 violated moderation policy",
            "error_code": "400",
            "traceback": "File proxy_server.py line 8570 ...",
            "llm_provider": "openai",
        },
    )
    data = LLMCallSpanData.from_standard_logging_payload(payload)
    assert data.error is not None
    assert data.error.error_type == "BadRequestError"
    assert data.error.message == "400 violated moderation policy"
    assert data.error.code == "400"
    assert data.error.stack_trace == "File proxy_server.py line 8570 ..."
    assert data.error.llm_provider == "openai"


def test_llm_call_adapter_error_details_default_to_none_when_absent():
    """Guardrail-shape payloads carry only ``error_class`` + ``error_message``.
    The detail fields must stay ``None`` so the emitter's ``if error.code:``
    guards skip stamping empty attributes."""
    payload = _sample_payload(
        status="failure",
        error_information={
            "error_class": "ContentFilter",
            "error_message": "guardrail rejected",
        },
    )
    data = LLMCallSpanData.from_standard_logging_payload(payload)
    assert data.error is not None
    assert data.error.code is None
    assert data.error.stack_trace is None
    assert data.error.llm_provider is None


def test_adapter_is_resilient_to_minimal_payload():
    data = LLMCallSpanData.from_standard_logging_payload({})
    assert data.request_model == ""
    assert data.operation is GenAIOperation.CHAT
    assert data.server is None
    assert data.usage.input_tokens is None


def test_content_capture_gated_off_by_default():
    # ``capture_content`` defaults off: prompt/response bodies must not reach the
    # span data (and so no vendor mapper can export them) unless explicitly
    # opted in. Non-content metadata (finish reasons) is still derived.
    payload = _sample_payload(
        messages=[{"role": "user", "content": "secret prompt"}],
    )
    payload["response"]["choices"] = [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": "secret"}}
    ]
    data = LLMCallSpanData.from_standard_logging_payload(payload)
    assert data.messages_in == ()
    assert data.choices_out == ()
    assert data.finish_reasons == ("stop",)


def test_request_identity_prefers_canonical_team_keys():
    from litellm.integrations.otel.model.payloads import RequestIdentity

    payload = _sample_payload(
        metadata={
            "user_api_key_team_id": "team-canonical",
            "user_api_key_team_alias": "alias-canonical",
            "user_api_key_hash": "hsh",
            "team_id": "legacy-ignored",  # legacy alias loses to the canonical key
        }
    )
    ident = RequestIdentity.from_payload(payload)
    assert ident.team_id == "team-canonical"
    assert ident.team_alias == "alias-canonical"
    assert ident.key_hash == "hsh"


def test_request_identity_falls_back_to_legacy_team_keys():
    from litellm.integrations.otel.model.payloads import RequestIdentity

    payload = _sample_payload(
        metadata={"team_id": "legacy-team", "team_alias": "legacy"}
    )
    ident = RequestIdentity.from_payload(payload)
    assert ident.team_id == "legacy-team"
    assert ident.team_alias == "legacy"


def test_guardrail_span_data_block_carries_verdict_and_error():
    from litellm.integrations.otel.model.payloads import GuardrailSpanData

    entry = {
        "guardrail_name": "openai-moderation",
        "guardrail_mode": "pre_call",
        "guardrail_status": "guardrail_intervened",
        "guardrail_provider": "openai",
        "guardrail_action": "BLOCKED",
        "guardrail_response": {"violated_categories": ["violence"]},
        "violation_categories": ["violence"],
        "masked_entity_count": {"EMAIL": 2, "PHONE": 1},
        "duration": 0.05,
    }
    d = GuardrailSpanData.from_logging_entry(entry)
    assert d.guardrail_name == "openai-moderation"
    assert d.status == "guardrail_intervened"
    assert d.provider == "openai"
    assert d.action == "BLOCKED"
    assert '"violence"' in (d.response_json or "")
    assert d.violation_categories == ("violence",)
    assert d.masked_entity_count == 3  # summed across entity types
    assert d.duration == 0.05
    assert d.error is not None  # intervention → span marked ERROR


def test_guardrail_span_data_success_has_no_error():
    from litellm.integrations.otel.model.payloads import GuardrailSpanData

    d = GuardrailSpanData.from_logging_entry(
        {
            "guardrail_name": "g",
            "guardrail_mode": "pre_call",
            "guardrail_status": "success",
        }
    )
    assert d.error is None
    assert d.status == "success"


def test_request_identity_from_user_api_key_auth():
    from litellm.integrations.otel.model.payloads import RequestIdentity

    class _Auth:
        team_id = "t9"
        team_alias = "team nine"
        api_key = "hashed-key"
        user_id = "u9"
        org_id = "o9"
        key_alias = "my-key"
        end_user_id = "eu9"

    ident = RequestIdentity.from_user_api_key_auth(_Auth())
    assert (ident.team_id, ident.team_alias, ident.key_hash) == (
        "t9",
        "team nine",
        "hashed-key",
    )
    assert ident.end_user == "eu9"
    assert ident.metadata["user_api_key_user_id"] == "u9"
    assert ident.metadata["user_api_key_org_id"] == "o9"
    assert ident.metadata["user_api_key_alias"] == "my-key"
    assert ident.metadata["user_api_key_end_user_id"] == "eu9"


# --- request-metadata translation layer (RequestContext) -------------------- #


def test_request_context_splits_group_from_dispatched_model():
    """On the proxy the caller asks for a model *group* that routes to a concrete
    deployment: ``gen_ai.request.model`` is the group, ``litellm.provider.model``
    is the dispatched (provider-prefixed) deployment model."""
    from litellm.integrations.otel.model.metadata import RequestContext

    payload = _sample_payload(
        model="openai/gpt-5.4-mini",  # reconstructed dispatched name
        model_group="gpt-5.4-mini",  # user-facing requested name
        model_id="dep-123",
    )
    ctx = RequestContext.from_standard_logging_payload(payload)
    assert ctx.request_model == "gpt-5.4-mini"
    assert ctx.provider_model == "openai/gpt-5.4-mini"
    assert ctx.identity.provider_model == "openai/gpt-5.4-mini"
    assert ctx.model_group == "gpt-5.4-mini"
    assert ctx.model_id == "dep-123"


def test_request_context_sdk_path_has_no_group():
    """Without a model group (the SDK path) the request and provider models
    coincide on the single call model."""
    from litellm.integrations.otel.model.metadata import RequestContext

    payload = _sample_payload()  # model="gpt-4o", no model_group
    ctx = RequestContext.from_standard_logging_payload(payload)
    assert ctx.request_model == "gpt-4o"
    assert ctx.provider_model == "gpt-4o"
    assert ctx.model_group is None


def test_request_context_prefers_explicit_dispatched_model():
    """``hidden_params.litellm_model_name`` is the authoritative dispatched model
    when present, winning over the reconstructed top-level ``model``."""
    from litellm.integrations.otel.model.metadata import RequestContext

    payload = _sample_payload(
        model="gpt-4o",
        model_group="gpt-4o",
        hidden_params={"litellm_model_name": "azure/my-deployment"},
    )
    ctx = RequestContext.from_standard_logging_payload(payload)
    assert ctx.request_model == "gpt-4o"
    assert ctx.provider_model == "azure/my-deployment"


def test_content_capture_opt_in_retains_bodies():
    payload = _sample_payload(
        messages=[{"role": "user", "content": "secret prompt"}],
    )
    payload["response"]["choices"] = [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
    ]
    data = LLMCallSpanData.from_standard_logging_payload(payload, capture_content=True)
    assert data.messages_in and data.messages_in[0]["content"] == "secret prompt"
    assert data.choices_out and data.choices_out[0]["message"]["content"] == "hi"


# --- config ----------------------------------------------------------------- #


def test_capture_span_content_resolves_modes():
    from litellm.integrations.otel.model.config import (
        CaptureMessageContent,
        OpenTelemetryV2Config,
    )

    # default (no_content) → off
    assert OpenTelemetryV2Config().capture_span_content is False
    assert (
        OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.SPAN_ONLY
        ).capture_span_content
        is True
    )
    assert (
        OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.SPAN_AND_EVENT
        ).capture_span_content
        is True
    )
    # event-only does not authorize span-attribute content
    assert (
        OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.EVENT_ONLY
        ).capture_span_content
        is False
    )
    # V1 accepted UPPER_SNAKE_CASE; the env value is case-insensitive so an
    # operator carrying ``SPAN_AND_EVENT`` forward still enables capture.
    assert (
        OpenTelemetryV2Config(
            capture_message_content="SPAN_AND_EVENT"
        ).capture_span_content
        is True
    )
    assert (
        OpenTelemetryV2Config(capture_message_content="SPAN_ONLY").capture_span_content
        is True
    )
    assert (
        OpenTelemetryV2Config(capture_message_content="NO_CONTENT").capture_span_content
        is False
    )


def test_capture_message_content_normalizer_only_touches_strings():
    """The casing normalizer lower-cases strings and leaves anything else
    untouched, so a non-string value still fails the field's ``str`` validation
    instead of being silently coerced into a bogus capture mode."""
    import pytest
    from pydantic import ValidationError

    from litellm.integrations.otel.model.config import OpenTelemetryV2Config

    with pytest.raises(ValidationError):
        OpenTelemetryV2Config(capture_message_content=123)


def test_v2_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    is_otel_v2_enabled.cache_clear()
    assert is_otel_v2_enabled() is False
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    assert is_otel_v2_enabled() is True


def test_v2_flag_resolved_once_not_per_call(monkeypatch):
    """Regression for LIT-3895: ``is_otel_v2_enabled`` sits on the proxy hot path
    (auth, logging-callback setup). Building the pydantic-settings model on every
    call re-scanned the environment at ~28us a pop and dropped throughput, so the
    flag must be resolved once and cached rather than reconstructed per call."""
    from litellm.integrations.otel.model import config as config_mod

    constructions = 0
    real_flag = config_mod._OTelV2Flag

    def _counting_flag(*args, **kwargs):
        nonlocal constructions
        constructions += 1
        return real_flag(*args, **kwargs)

    monkeypatch.setattr(config_mod, "_OTelV2Flag", _counting_flag)
    config_mod.is_otel_v2_enabled.cache_clear()

    for _ in range(50):
        config_mod.is_otel_v2_enabled()

    assert constructions == 1


def test_config_from_env(monkeypatch):
    for var in (
        "OTEL_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_HEADERS",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_SERVICE_NAME",
        "LITELLM_OTEL_LEGACY_COMPAT",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector:4318")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-svc")
    cfg = OpenTelemetryV2Config.from_env()
    # endpoint with no explicit exporter implies OTLP/HTTP
    assert cfg.exporter == "otlp_http"
    assert cfg.endpoint == "https://collector:4318"
    assert cfg.service_name == "my-svc"
    assert cfg.legacy_compat is True  # dual-emit default during deprecation window


def test_config_legacy_compat_env_toggle(monkeypatch):
    monkeypatch.setenv("LITELLM_OTEL_LEGACY_COMPAT", "false")
    assert OpenTelemetryV2Config.from_env().legacy_compat is False


# --- baggage allowlist (the antipattern boundary) --------------------------- #


def test_promoted_baggage_is_bounded_allowlist():
    identity = RequestIdentity(
        call_id="c1",
        team_id="t1",
        team_alias="team one",
        key_hash="hsh",
        end_user="u1",
        metadata={"user_api_key_org_id": "org1", "secret_blob": "should-not-promote"},
    )
    promoted = promoted_baggage(identity, "gpt-4o", BAGGAGE_PROMOTED_KEYS)
    assert promoted[LiteLLM.TEAM_ID] == "t1"
    assert promoted[LiteLLM.TEAM_ALIAS] == "team one"
    assert promoted[GenAI.REQUEST_MODEL] == "gpt-4o"
    # allowlisted metadata sub-key is promoted under the litellm.metadata.* prefix
    assert promoted[f"{LiteLLM.METADATA_PREFIX}user_api_key_org_id"] == "org1"
    # full metadata blob is NOT promoted
    assert all("secret_blob" not in key for key in promoted)
    # http.* is never a promoted key
    assert HTTP.ROUTE not in promoted
    assert HTTP.REQUEST_METHOD not in promoted


@pytest.mark.parametrize(
    "resource, expected",
    [
        ("https://weather.example.com", ("weather.example.com", 443)),
        ("http://weather.example.com", ("weather.example.com", 80)),
        ("https://weather.example.com:8443", ("weather.example.com", 8443)),
        ("mcp://weather.example.com", ("weather.example.com", None)),
        ("http://::1:8080", (None, None)),
        ("http://fe80::1%25eth0:80", (None, None)),
        (None, (None, None)),
        ("", (None, None)),
    ],
    ids=[
        "https-default",
        "http-default",
        "explicit-port",
        "no-default-port",
        "ipv6-unbracketed",
        "ipv6-zone-scoped",
        "none",
        "empty",
    ],
)
def test_upstream_address_port(resource, expected):
    """The redacted MCP origin resolves to the address and port a consumer names its
    dependency from. A scheme outside the default-port map yields no port, and an IPv6
    origin yields nothing at all because the redactor rebuilds it without its brackets;
    both are why the mapper gates ``rpc.system`` on the complete pair."""
    assert _upstream_address_port(resource) == expected
