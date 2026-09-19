"""Tests for the vendor mappers (OpenInference, Langfuse, Weave, Langtrace).

Composition over inheritance: each vendor's vocabulary is a mapper. Layering
mappers on the same span carries multiple naming schemes for different
backends, so one trace lights up every configured destination.
"""

import json

import pytest

from litellm.integrations.otel import GenAIOperation
from litellm.integrations.otel.mappers import (
    LangfuseMapper,
    LangtraceMapper,
    OpenInferenceMapper,
    WeaveMapper,
    resolve_mappers,
)
from litellm.integrations.otel.model.payloads import (
    EmbeddingOutput,
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
    RequestIdentity,
    ServerInfo,
    ToolDefinition,
)
from litellm.integrations.otel.model.trace_controls import TraceControls


def _llm_call(**overrides):
    base = dict(
        operation=GenAIOperation.CHAT,
        provider="openai",
        request_model="gpt-4o",
        response_model="gpt-4o-2024",
        response_id="resp_1",
        request_params=LLMRequestParams(temperature=0.5, top_p=0.9, max_tokens=128),
        usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        finish_reasons=("stop",),
        error=None,
        response_cost=0.001,
        server=ServerInfo("api.openai.com", 443),
        identity=RequestIdentity(call_id="c1", team_id="t1", team_alias="team one"),
        is_streaming=False,
        tools=(
            ToolDefinition(
                name="lookup_weather",
                description="Get weather",
                parameters_json='{"type":"object"}',
            ),
        ),
        messages_in=(
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What's the weather?"},
        ),
        choices_out=(
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Sunny."},
            },
        ),
        system_fingerprint="fp_abc",
    )
    base.update(overrides)
    return LLMCallSpanData(**base)


# --------------------------------------------------------------------------- #
#  OpenInference (Arize + Phoenix shared vocabulary)
# --------------------------------------------------------------------------- #


def test_openinference_mapper_input_output_messages():
    attrs = OpenInferenceMapper().map(_llm_call())
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "gpt-4o"
    assert attrs["llm.provider"] == "openai"
    assert attrs["llm.input_messages.0.message.role"] == "system"
    assert attrs["llm.input_messages.0.message.content"] == "Be concise."
    assert attrs["llm.input_messages.1.message.role"] == "user"
    assert attrs["llm.output_messages.0.message.role"] == "assistant"
    assert attrs["llm.output_messages.0.message.content"] == "Sunny."
    assert attrs["llm.token_count.prompt"] == 12
    assert attrs["llm.token_count.completion"] == 8
    assert attrs["llm.token_count.total"] == 20
    # tool definitions ride the OpenInference schema
    assert attrs["llm.tools.0.tool.name"] == "lookup_weather"
    # invocation_parameters is JSON-serialized
    params = json.loads(attrs["llm.invocation_parameters"])
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 128


def test_openinference_mapper_skips_non_llm_roles():
    from litellm.integrations.otel.model.payloads import GuardrailSpanData

    assert OpenInferenceMapper().map(GuardrailSpanData("presidio")) == {}


def test_openinference_multimodal_content_text_only():
    data = _llm_call(
        messages_in=(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi "},
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "there"},
                ],
            },
        )
    )
    attrs = OpenInferenceMapper().map(data)
    assert attrs["llm.input_messages.0.message.content"] == "hi there"


# --------------------------------------------------------------------------- #
#  Langfuse
# --------------------------------------------------------------------------- #


def test_langfuse_mapper_observation_attrs():
    attrs = LangfuseMapper().map(_llm_call())
    assert attrs["langfuse.observation.type"] == "generation"
    assert attrs["langfuse.observation.model.name"] == "gpt-4o"
    assert attrs["langfuse.observation.metadata.provider"] == "openai"
    usage = json.loads(attrs["langfuse.observation.usage_details"])
    assert usage["input"] == 12 and usage["output"] == 8
    params = json.loads(attrs["langfuse.observation.model.parameters"])
    assert params["temperature"] == 0.5
    cost = json.loads(attrs["langfuse.observation.cost_details"])
    assert cost["total"] == 0.001
    assert attrs["langfuse.trace.metadata.team_id"] == "t1"


def test_langfuse_mapper_names_the_trace_from_the_caller():
    named = LangfuseMapper().map(_llm_call(trace=TraceControls(name="nightly-eval")))
    assert named["langfuse.trace.name"] == "nightly-eval"
    assert "langfuse.trace.name" not in LangfuseMapper().map(_llm_call(trace=TraceControls()))


def test_langfuse_mapper_carries_the_caller_user_session_and_tags():
    controls = TraceControls(user_id="u-42", session_id="s-7", tags=("prod", "eval", "nightly"))
    attrs = LangfuseMapper().map(_llm_call(trace=controls))

    assert attrs["user.id"] == "u-42"
    assert attrs["session.id"] == "s-7"
    assert attrs["langfuse.trace.tags"] == ("prod", "eval", "nightly")
    assert attrs["langfuse.trace.metadata.team_id"] == "t1"
    assert attrs["langfuse.trace.metadata.team_alias"] == "team one"


def test_langfuse_mapper_omits_unset_trace_controls():
    attrs = LangfuseMapper().map(_llm_call(trace=TraceControls(user_id="", session_id=None, tags=())))

    assert {"user.id", "session.id", "langfuse.trace.tags", "langfuse.trace.name"}.isdisjoint(attrs)


def test_langfuse_trace_attributes_match_between_root_and_generation():
    controls = TraceControls(name="n", user_id="u", session_id="s", tags=("t",))
    generation = LangfuseMapper().map(_llm_call(trace=controls))

    root = LangfuseMapper.trace_attributes(controls)
    assert root == {"langfuse.trace.name": "n", "user.id": "u", "session.id": "s", "langfuse.trace.tags": ("t",)}
    assert all(generation[key] == value for key, value in root.items())


def test_langfuse_mapper_skips_when_no_messages():
    data = _llm_call(messages_in=(), choices_out=())
    attrs = LangfuseMapper().map(data)
    assert "langfuse.observation.input" not in attrs
    assert "langfuse.observation.output" not in attrs


def test_langfuse_mapper_renders_an_embedding_call_with_a_vector_summary_as_output():
    data = _llm_call(
        operation=GenAIOperation.EMBEDDINGS,
        request_model="text-embedding-3-small",
        messages_in=({"role": "user", "content": "hello"},),
        choices_out=(),
        finish_reasons=(),
        embedding_output=EmbeddingOutput(count=2, dimensions=1536),
    )
    attrs = LangfuseMapper().map(data)

    assert attrs["langfuse.observation.type"] == "generation"
    assert json.loads(attrs["langfuse.observation.output"]) == {"count": 2, "dimensions": 1536}
    assert json.loads(attrs["langfuse.observation.input"]) == [{"role": "user", "content": "hello"}]


def test_langfuse_mapper_keeps_chat_output_when_no_embedding_summary():
    attrs = LangfuseMapper().map(_llm_call(embedding_output=None))

    assert json.loads(attrs["langfuse.observation.output"]) == [{"role": "assistant", "content": "Sunny."}]


def test_langfuse_mapper_renders_a_responses_api_call_from_the_standard_logging_payload():
    payload = {
        "call_type": "aresponses",
        "custom_llm_provider": "openai",
        "model": "gpt-5.4-nano",
        "messages": [{"role": "user", "content": "weather in sf?"}],
        "response": {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "sf"}'},
            ],
        },
    }
    data = LLMCallSpanData.from_standard_logging_payload(payload, capture_content=True)
    attrs = LangfuseMapper().map(data)

    assert json.loads(attrs["langfuse.observation.output"]) == [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "sf"}'}}
            ],
        }
    ]
    assert attrs["langfuse.observation.type"] == "generation"


# --------------------------------------------------------------------------- #
#  Weave
# --------------------------------------------------------------------------- #


def test_weave_mapper_display_and_output():
    attrs = WeaveMapper().map(_llm_call())
    assert attrs["weave.display_name"] == "chat gpt-4o"
    assert attrs["weave.call_id"] == "c1"
    decoded = json.loads(attrs["weave.output"])
    assert decoded[0]["message"]["content"] == "Sunny."


# --------------------------------------------------------------------------- #
#  Langtrace
# --------------------------------------------------------------------------- #


def test_langtrace_mapper_attrs():
    attrs = LangtraceMapper().map(_llm_call())
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["langtrace.service.name"] == "openai"
    assert attrs["llm.model"] == "gpt-4o"
    assert attrs["gen_ai.response.model"] == "gpt-4o-2024"
    assert attrs["gen_ai.system_fingerprint"] == "fp_abc"
    assert attrs["llm.temperature"] == 0.5
    assert attrs["llm.token.counts.total"] == 20


# --------------------------------------------------------------------------- #
#  Composition (the V2 punchline)
# --------------------------------------------------------------------------- #


def test_resolve_mappers_composition_layers_vocabularies():
    """One span, three vocabularies — Arize + Langfuse + canonical together."""
    chain = resolve_mappers(["genai", "openinference", "langfuse"])
    data = _llm_call()
    union: dict = {}
    for mapper in chain:
        union.update(mapper.map(data))
    # Canonical
    assert union["gen_ai.operation.name"] == "chat"
    # OpenInference
    assert union["llm.model_name"] == "gpt-4o"
    assert union["openinference.span.kind"] == "LLM"
    # Langfuse
    assert union["langfuse.observation.type"] == "generation"


def test_resolve_mappers_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown mapper name 'nope'"):
        resolve_mappers(["genai", "nope"])
