"""Tests for the vendor mappers (OpenInference, Langfuse, Weave, Langtrace).

Composition over inheritance: each vendor's vocabulary is a mapper. Layering
mappers on the same span carries multiple naming schemes for different
backends, so one trace lights up every configured destination.
"""

import json
from collections.abc import Mapping
from itertools import chain
from typing import Final

import pytest

from litellm.integrations.otel import GenAIOperation
from litellm.integrations.otel.mappers import (
    LangfuseMapper,
    LangtraceMapper,
    OpenInferenceMapper,
    WeaveMapper,
    resolve_mappers,
)
from litellm.integrations.otel.mappers.openinference import fit_indexed_messages
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
from tests.unit.integrations.otel.test_otel_v2_sources_of_truth import _responses_payload


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


def test_openinference_output_tool_calls_preserve_calls_in_attributes_and_value():
    tool_calls: Final = [
        {
            "id": "call_paris",
            "type": "function",
            "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
        },
        {
            "id": "call_search",
            "function": {"name": "search", "arguments": {"q": 1}},
            "index": 0,
        },
        "ignored",
    ]
    data: Final = _llm_call(
        choices_out=(
            {
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
            },
        )
    )
    attrs: Final = OpenInferenceMapper().map(data)
    assert {key: value for key, value in attrs.items() if ".tool_calls." in key} == {
        "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_paris",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup_weather",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city": "Paris"}',
        "llm.output_messages.0.message.tool_calls.1.tool_call.id": "call_search",
        "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "search",
        "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": '{"q": 1}',
    }
    assert json.loads(attrs["output.value"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_paris",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": 1}'},
                },
            ],
        }
    ]


def test_openinference_responses_tool_calls_are_emitted_as_output_attributes():
    data: Final = LLMCallSpanData.from_standard_logging_payload(
        _responses_payload(
            [
                {
                    "type": "function_call",
                    "call_id": "call_resp",
                    "name": "lookup_weather",
                    "arguments": '{"city": "Paris"}',
                }
            ]
        ),
        capture_content=True,
    )
    attrs: Final = OpenInferenceMapper().map(data)
    tool_call: Final = "llm.output_messages.0.message.tool_calls.0.tool_call."

    assert {key: value for key, value in attrs.items() if ".tool_calls." in key} == {
        tool_call + "id": "call_resp",
        tool_call + "function.name": "lookup_weather",
        tool_call + "function.arguments": '{"city": "Paris"}',
    }
    assert json.loads(attrs["output.value"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_resp",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
    ]


def test_openinference_input_tool_calls_stay_in_value_only():
    data: Final = _llm_call(
        messages_in=(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
        )
    )
    attrs: Final = OpenInferenceMapper().map(data)
    assert all(".tool_calls." not in key for key in attrs)
    assert json.loads(attrs["input.value"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
    ]


def test_openinference_output_tool_calls_do_not_shed_input_roles_under_budget():
    message_groups: Final = tuple(
        (
            {"role": "user", "content": f"Question {index}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{index}", "content": f"Result {index}"},
        )
        for index in range(13)
    )
    messages_in: Final = tuple(chain.from_iterable(message_groups)) + ({"role": "user", "content": "Final request"},)
    tools: Final = tuple(
        ToolDefinition(name=name, description="Tool", parameters_json='{"type":"object"}')
        for name in ("lookup_weather", "search", "get_location", "convert_units")
    )
    data: Final = _llm_call(
        messages_in=messages_in,
        tools=tools,
        choices_out=(
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_output",
                            "type": "function",
                            "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            },
        ),
    )
    attrs: Final = fit_indexed_messages(OpenInferenceMapper().map(data), 128)
    tool_call: Final = "llm.output_messages.0.message.tool_calls.0.tool_call."

    assert {
        f"llm.input_messages.{index}.message.role": attrs.get(f"llm.input_messages.{index}.message.role")
        for index in range(40)
    } == {f"llm.input_messages.{index}.message.role": message["role"] for index, message in enumerate(messages_in)}
    assert {key: value for key, value in attrs.items() if ".tool_calls." in key} == {
        tool_call + "id": "call_output",
        tool_call + "function.name": "lookup_weather",
        tool_call + "function.arguments": '{"city": "Paris"}',
    }


def test_openinference_plain_output_messages_keep_the_existing_value_shape():
    attrs: Final = OpenInferenceMapper().map(_llm_call())
    assert all(".tool_calls." not in key for key in attrs)
    assert json.loads(attrs["output.value"]) == [{"role": "assistant", "content": "Sunny."}]


def test_openinference_metadata_contains_only_promoted_metadata():
    attrs: Final = OpenInferenceMapper().map(
        _llm_call(promoted_metadata={"trace_marker": "m", "user_api_key_alias": "k"})
    )
    assert json.loads(attrs["metadata"]) == {"trace_marker": "m", "user_api_key_alias": "k"}
    assert "metadata" not in OpenInferenceMapper().map(_llm_call())


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


def _langfuse_usage_details(usage_object: Mapping[str, object]) -> dict[str, object]:
    payload: Final = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": usage_object["prompt_tokens"],
        "completion_tokens": usage_object["completion_tokens"],
        "total_tokens": usage_object["total_tokens"],
        "metadata": {"usage_object": usage_object},
    }
    attrs: Final = LangfuseMapper().map(LLMCallSpanData.from_standard_logging_payload(payload))
    return json.loads(attrs["langfuse.observation.usage_details"])


def test_langfuse_usage_details_split_openai_cached_and_reasoning_tokens():
    usage: Final = _langfuse_usage_details(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )
    assert usage == {
        "input": 40,
        "input_cached_tokens": 60,
        "output": 20,
        "output_reasoning_tokens": 30,
        "total": 150,
    }


def test_langfuse_usage_details_split_anthropic_cache_read_and_creation_tokens():
    usage: Final = _langfuse_usage_details(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 40,
            "total_tokens": 1040,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 800, "cache_creation_tokens": 150},
        }
    )
    assert usage == {
        "input": 50,
        "input_cached_tokens": 800,
        "input_cache_creation": 150,
        "output": 40,
        "total": 1040,
    }


def test_langfuse_usage_details_omit_zero_cache_and_reasoning_counts():
    usage: Final = _langfuse_usage_details(
        {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )
    assert usage == {"input": 12, "output": 8, "total": 20}


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
            "refusal": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "sf"}'}}
            ],
        }
    ]
    assert attrs["langfuse.observation.type"] == "generation"


def test_langfuse_mapper_renders_an_ocr_call_with_the_page_markdown_as_output():
    payload = {
        "call_type": "aocr",
        "custom_llm_provider": "mistral",
        "model": "mistral-ocr-latest",
        "messages": None,
        "response": {
            "object": "ocr",
            "model": "mistral-ocr-latest",
            "pages": [{"index": 0, "markdown": "# Invoice"}, {"index": 1, "markdown": "Total: 42"}],
            "usage_info": {"pages_processed": 2},
        },
    }
    data = LLMCallSpanData.from_standard_logging_payload(payload, capture_content=True)
    attrs = LangfuseMapper().map(data)

    assert json.loads(attrs["langfuse.observation.output"]) == [
        {"role": "assistant", "content": "# Invoice\n\nTotal: 42", "refusal": None, "tool_calls": None}
    ]
    assert attrs["langfuse.observation.type"] == "generation"


@pytest.mark.parametrize(
    ("call_type", "response", "expected_content"),
    [
        ("atext_completion", {"choices": [{"text": "Paris.", "finish_reason": "stop"}]}, "Paris."),
        ("atranscription", {"text": "What is the weather like?"}, "What is the weather like?"),
        ("amoderation", {"results": [{"flagged": True, "categories": {"violence": True}}]}, "flagged: violence"),
        ("aimage_generation", {"data": [{"b64_json": "QUJDRA=="}]}, "b64_json image (4 bytes)"),
        ("aspeech", {"object": "binary", "content_type": "audio/mpeg", "num_bytes": 9}, "audio/mpeg (9 bytes)"),
    ],
)
def test_langfuse_mapper_renders_every_non_chat_route_output_as_an_assistant_message(
    call_type: str, response: Mapping[str, object], expected_content: str
) -> None:
    payload: Final[dict[str, object]] = {
        "call_type": call_type,
        "custom_llm_provider": "openai",
        "model": "m",
        "messages": None,
        "response": response,
    }
    attrs: Final = LangfuseMapper().map(LLMCallSpanData.from_standard_logging_payload(payload, capture_content=True))

    assert json.loads(attrs["langfuse.observation.output"]) == [
        {"role": "assistant", "content": expected_content, "refusal": None, "tool_calls": None}
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
