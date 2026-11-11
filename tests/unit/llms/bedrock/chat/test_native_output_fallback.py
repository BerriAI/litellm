import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.llms.bedrock.chat.native_output_fallback import NativeOutputAttemptResult, call_with_native_output_fallback
from litellm.llms.bedrock.common_utils import BedrockError, NativeOutputFallback
from litellm.types.utils import ModelResponse

_MODEL: Final = "us.anthropic.claude-opus-4-8"
_FALLBACK: Final = NativeOutputFallback(
    optional_params={"tools": "json_tool_call"}, litellm_params={"model_info": {"opted_out": True}}
)
_NATIVE_RETRY_WITH_STRICT_TOOLS: Final = NativeOutputFallback(
    optional_params={
        "tools": (
            {"function": {"name": "json_schema", "strict": True}},
            {"function": {"name": "custom", "strict": True}},
        )
    },
    litellm_params={
        "model_info": {
            "supports_native_structured_output": False,
            "supports_bedrock_native_structured_output": False,
        }
    },
)
_COMPOSED_FALLBACK: Final = NativeOutputFallback(
    optional_params={"tools": [{"function": {"name": "json_schema"}}, {"function": {"name": "custom"}}]},
    litellm_params=_NATIVE_RETRY_WITH_STRICT_TOOLS.litellm_params,
)
_REJECTION: Final = BedrockError(
    status_code=400,
    message='{"message":"The model returned the following errors: output_config.format: Extra inputs are not permitted"}',
)
_GRAMMAR_REJECTION: Final = BedrockError(
    status_code=400,
    message='{"message":"The model returned the following errors: The compiled grammar is too large, which would cause '
    'performance issues. Simplify your tool schemas or reduce the number of strict tools."}',
)
_SCHEMA_REFUSAL: Final = BedrockError(
    status_code=400, message='{"message":"output_config.format.schema: pattern is too complex for structured output"}'
)
_OTHER_400: Final = BedrockError(status_code=400, message='{"message":"messages.0.content: Field required"}')


@dataclass(slots=True)
class ScriptedBedrock:
    errors: tuple[BedrockError | None, ...]
    asynchronous: bool
    sent: tuple[NativeOutputFallback | None, ...] = ()

    def _answer(self, error: BedrockError | None) -> ModelResponse:
        if error is not None:
            raise error
        return ModelResponse(id=f"answer-{len(self.sent)}")

    async def _answer_later(self, error: BedrockError | None) -> ModelResponse:
        await asyncio.sleep(0)
        return self._answer(error)

    def __call__(self, fallback: NativeOutputFallback | None) -> NativeOutputAttemptResult:
        self.sent = (*self.sent, fallback)
        error: Final = self.errors[len(self.sent) - 1]
        return self._answer_later(error) if self.asynchronous else self._answer(error)


def _resolve(result: NativeOutputAttemptResult) -> object:
    if isinstance(result, Coroutine):
        return asyncio.run(result)
    return result


def _call(bedrock: ScriptedBedrock, fallback: NativeOutputFallback | None) -> object:
    return _resolve(call_with_native_output_fallback(bedrock, None if fallback is None else lambda: fallback, _MODEL))


_MODES: Final = pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])


@_MODES
@pytest.mark.parametrize(
    "rejection",
    [_REJECTION, _GRAMMAR_REJECTION],
    ids=["format_not_permitted", "compiled_grammar_too_large"],
)
def test_a_native_output_rejection_is_sent_again_once_with_the_fallback_params(
    asynchronous: bool, rejection: BedrockError
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(rejection, None), asynchronous=asynchronous)

    response: Final = _call(bedrock, _FALLBACK)

    assert isinstance(response, ModelResponse) and response.id == "answer-2"
    assert bedrock.sent == (None, _FALLBACK)


@_MODES
def test_a_second_native_output_rejection_reaches_the_caller(asynchronous: bool) -> None:
    bedrock: Final = ScriptedBedrock(errors=(_REJECTION, _REJECTION), asynchronous=asynchronous)

    with pytest.raises(BedrockError):
        _call(bedrock, _FALLBACK)

    assert len(bedrock.sent) == 2


@_MODES
def test_a_grammar_rejection_after_a_native_retry_reaches_the_caller_without_strict_tools_fallback(
    asynchronous: bool,
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(_GRAMMAR_REJECTION, _GRAMMAR_REJECTION), asynchronous=asynchronous)

    with pytest.raises(BedrockError) as raised:
        _resolve(call_with_native_output_fallback(bedrock, lambda: _FALLBACK, _MODEL))

    assert raised.value is _GRAMMAR_REJECTION
    assert bedrock.sent == (None, _FALLBACK)


@_MODES
@pytest.mark.parametrize(
    ("error", "fallback"),
    [(_OTHER_400, _FALLBACK), (_REJECTION, None)],
    ids=["other_400", "no_fallback"],
)
def test_no_second_request_without_both_a_rejection_and_a_fallback(
    asynchronous: bool, error: BedrockError, fallback: NativeOutputFallback | None
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(error,), asynchronous=asynchronous)

    with pytest.raises(BedrockError) as raised:
        _call(bedrock, fallback)

    assert raised.value is error
    assert bedrock.sent == (None,)


@_MODES
def test_a_rejection_reaches_the_caller_when_the_retry_would_send_native_output_again(asynchronous: bool) -> None:
    bedrock: Final = ScriptedBedrock(errors=(_REJECTION,), asynchronous=asynchronous)

    with pytest.raises(BedrockError) as raised:
        _resolve(call_with_native_output_fallback(bedrock, lambda: None, _MODEL))

    assert raised.value is _REJECTION
    assert bedrock.sent == (None,)


# Bedrock's 403 for a role without the streaming action, seen live on us.anthropic.claude-sonnet-5 on 2026-11-06
_STREAM_PERMISSION_DENIAL: Final = BedrockError(
    status_code=403,
    message='{"Message":"User: arn:aws:sts::123456789012:assumed-role/invoke-only/qa is not authorized to perform: '
    "bedrock:InvokeModelWithResponseStream on resource: arn:aws:bedrock:us-west-2:123456789012:inference-profile/"
    "us.anthropic.claude-sonnet-5 because no identity-based policy allows the "
    'bedrock:InvokeModelWithResponseStream action"}',
)
_INVOKE_PERMISSION_DENIAL: Final = BedrockError(
    status_code=403,
    message='{"Message":"User: arn:aws:sts::123456789012:assumed-role/no-bedrock/qa is not authorized to perform: '
    'bedrock:InvokeModel because no identity-based policy allows the bedrock:InvokeModel action"}',
)
_BUFFERED_NATIVE_STREAM: Final = NativeOutputFallback(
    optional_params={"outputConfig": "native", "fake_stream": True}, litellm_params={}
)


def _buffered(fallback: NativeOutputFallback) -> NativeOutputFallback:
    return NativeOutputFallback(
        optional_params={**fallback.optional_params, "fake_stream": True}, litellm_params=fallback.litellm_params
    )


def _call_with_buffered_stream_fallback(
    bedrock: ScriptedBedrock, fallback: NativeOutputFallback | None, buffered_stream_fallback: NativeOutputFallback
) -> object:
    return _resolve(
        call_with_native_output_fallback(
            bedrock,
            None if fallback is None else lambda: fallback,
            _MODEL,
            buffered_stream_fallback=lambda: buffered_stream_fallback,
        )
    )


@_MODES
def test_a_denied_native_stream_is_sent_again_once_as_a_buffered_native_request(
    asynchronous: bool, caplog: pytest.LogCaptureFixture
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(_STREAM_PERMISSION_DENIAL, None), asynchronous=asynchronous)

    with caplog.at_level("WARNING", logger="LiteLLM"):
        response: Final = _call_with_buffered_stream_fallback(bedrock, _FALLBACK, _BUFFERED_NATIVE_STREAM)

    assert isinstance(response, ModelResponse) and response.id == "answer-2"
    assert bedrock.sent == (None, _BUFFERED_NATIVE_STREAM)
    assert "bedrock:InvokeModelWithResponseStream" in caplog.text and _MODEL in caplog.text


@_MODES
def test_a_buffered_native_request_bedrock_rejects_is_sent_again_buffered_with_the_fallback_params(
    asynchronous: bool,
) -> None:
    bedrock: Final = ScriptedBedrock(
        errors=(_STREAM_PERMISSION_DENIAL, _GRAMMAR_REJECTION, None), asynchronous=asynchronous
    )

    response: Final = _call_with_buffered_stream_fallback(bedrock, _FALLBACK, _BUFFERED_NATIVE_STREAM)

    assert isinstance(response, ModelResponse) and response.id == "answer-3"
    assert bedrock.sent == (None, _BUFFERED_NATIVE_STREAM, _buffered(_FALLBACK))


@_MODES
def test_strict_tools_bedrock_rejects_after_a_buffered_native_retry_are_dropped_on_the_buffered_json_tool_retry(
    asynchronous: bool,
) -> None:
    bedrock: Final = ScriptedBedrock(
        errors=(_STREAM_PERMISSION_DENIAL, _GRAMMAR_REJECTION, _GRAMMAR_REJECTION, None), asynchronous=asynchronous
    )

    response: Final = _resolve(
        call_with_native_output_fallback(
            bedrock,
            lambda: _NATIVE_RETRY_WITH_STRICT_TOOLS,
            "us.anthropic.claude-sonnet-4-6",
            strict_tools_fallback=lambda: _STRICT_FALLBACK,
            buffered_stream_fallback=lambda: _BUFFERED_NATIVE_STREAM,
        )
    )

    assert isinstance(response, ModelResponse) and response.id == "answer-4"
    assert bedrock.sent == (
        None,
        _BUFFERED_NATIVE_STREAM,
        _buffered(_NATIVE_RETRY_WITH_STRICT_TOOLS),
        _buffered(_COMPOSED_FALLBACK),
    )


@_MODES
@pytest.mark.parametrize(
    ("error", "buffered_stream_fallback"),
    [
        (
            _STREAM_PERMISSION_DENIAL,
            NativeOutputFallback(optional_params={"outputConfig": "native"}, litellm_params={}),
        ),
        (_INVOKE_PERMISSION_DENIAL, _BUFFERED_NATIVE_STREAM),
    ],
    ids=["retry_would_stream_again", "invoke_denied"],
)
def test_a_permission_denial_reaches_the_caller_unless_the_retry_buffers_the_stream(
    asynchronous: bool, error: BedrockError, buffered_stream_fallback: NativeOutputFallback
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(error,), asynchronous=asynchronous)

    with pytest.raises(BedrockError) as raised:
        _call_with_buffered_stream_fallback(bedrock, _FALLBACK, buffered_stream_fallback)

    assert raised.value is error
    assert bedrock.sent == (None,)


@pytest.mark.parametrize(
    ("error", "advises_opting_out"),
    [(_REJECTION, True), (_SCHEMA_REFUSAL, False)],
    ids=["native_output_unsupported", "schema_refused"],
)
def test_a_rejection_logs_the_retry_and_advises_opting_out_only_when_native_output_is_unsupported(
    error: BedrockError, advises_opting_out: bool, caplog: pytest.LogCaptureFixture
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(error, None), asynchronous=False)

    with caplog.at_level("WARNING", logger="LiteLLM"):
        _call(bedrock, _FALLBACK)

    assert _MODEL in caplog.text
    assert ("supports_bedrock_native_structured_output: false" in caplog.text) is advises_opting_out


_STRICT_FALLBACK: Final = NativeOutputFallback(optional_params={"tools": "without_strict"}, litellm_params={})
# Bedrock's 400s for strict tools, seen live on us.anthropic.claude-sonnet-4-6 from 2026-10-20 to 2026-10-28 and quoted
# in issue #34388, each answered 200 once the request dropped strict
_STRICT_TOOLS_REJECTIONS: Final = tuple(
    BedrockError(status_code=400, message=f'{{"message":"The model returned the following errors: {text}"}}')
    for text in (
        "Too many strict tools (21). The maximum number of strict tools supported is 20.",
        "Schemas contains too many optional parameters (27), which would make grammar compilation inefficient.",
        "Schemas contains too many parameters with union types (18 parameters with type arrays or anyOf). "
        "This causes exponential compilation cost.",
        "The compiled grammar is too large, which would cause performance issues.",
        "tools.0.custom: Invalid schema: Schema type is missing for schema: {'description': 'tag'}",
        "tools.3.custom: For 'integer' type, property 'minimum' is not supported",
        "Compiled grammar size (410.4MB) exceeds maximum allowed size (300MB)",
        "tools.1.custom.strict: Extra inputs are not permitted",
        "Grammar compilation timed out.",
    )
)
_TOO_MANY_STRICT_TOOLS: Final = _STRICT_TOOLS_REJECTIONS[0]
# Bedrock's 400s for requests with a strict tool that fail for other reasons, seen live on
# us.anthropic.claude-sonnet-4-6 on 2026-10-21, which echo tool names and tool call ids
_OTHER_400S_WITH_A_STRICT_TOOL: Final = tuple(
    BedrockError(status_code=400, message=f'{{"message":"{text}"}}')
    for text in (
        "The tool check_grammar is already defined at toolConfig.tools.1.",
        "Expected toolResult blocks at messages.0.content for the following Ids: call_grammar_9",
        "tool_use ids were found without tool_result blocks immediately after: call_grammar_1",
    )
)


def _call_with_strict_tools_fallback(
    bedrock: ScriptedBedrock,
    fallback: NativeOutputFallback | None,
    strict_tools_fallback: NativeOutputFallback | None,
    model: str = _MODEL,
) -> object:
    return _resolve(
        call_with_native_output_fallback(
            bedrock,
            None if fallback is None else lambda: fallback,
            model,
            strict_tools_fallback=None if strict_tools_fallback is None else lambda: strict_tools_fallback,
        )
    )


@_MODES
@pytest.mark.parametrize("rejection", _STRICT_TOOLS_REJECTIONS, ids=lambda error: error.message[60:100])
def test_a_strict_tools_rejection_is_sent_again_once_with_the_strict_tools_fallback(
    asynchronous: bool, rejection: BedrockError
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(rejection, None), asynchronous=asynchronous)

    response: Final = _call_with_strict_tools_fallback(bedrock, None, _STRICT_FALLBACK)

    assert isinstance(response, ModelResponse) and response.id == "answer-2"
    assert bedrock.sent == (None, _STRICT_FALLBACK)


@_MODES
@pytest.mark.parametrize(
    ("error", "strict_tools_fallback"),
    [
        (_OTHER_400, _STRICT_FALLBACK),
        *((error, _STRICT_FALLBACK) for error in _OTHER_400S_WITH_A_STRICT_TOOL),
        (BedrockError(status_code=500, message=_TOO_MANY_STRICT_TOOLS.message), _STRICT_FALLBACK),
        (_TOO_MANY_STRICT_TOOLS, None),
    ],
    ids=[
        "other_400",
        "duplicate_tool_name",
        "missing_tool_result",
        "orphan_tool_use",
        "server_error",
        "no_strict_tools_fallback",
    ],
)
def test_no_second_request_without_both_a_strict_tools_rejection_and_a_strict_tools_fallback(
    asynchronous: bool, error: BedrockError, strict_tools_fallback: NativeOutputFallback | None
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(error,), asynchronous=asynchronous)

    with pytest.raises(BedrockError) as raised:
        _call_with_strict_tools_fallback(bedrock, None, strict_tools_fallback)

    assert raised.value is error
    assert bedrock.sent == (None,)


@pytest.mark.parametrize(
    ("error", "retried_with"),
    [(_REJECTION, _FALLBACK), (_TOO_MANY_STRICT_TOOLS, _STRICT_FALLBACK)],
    ids=["native_output_rejection", "strict_tools_rejection"],
)
def test_each_rejection_is_sent_again_with_its_own_fallback_when_both_are_set(
    error: BedrockError, retried_with: NativeOutputFallback
) -> None:
    bedrock: Final = ScriptedBedrock(errors=(error, None), asynchronous=False)

    _call_with_strict_tools_fallback(bedrock, _FALLBACK, _STRICT_FALLBACK)

    assert bedrock.sent == (None, retried_with)


@_MODES
def test_a_strict_tools_rejection_after_native_retry_composes_both_fallbacks(asynchronous: bool) -> None:
    bedrock: Final = ScriptedBedrock(
        errors=(_GRAMMAR_REJECTION, _GRAMMAR_REJECTION, None),
        asynchronous=asynchronous,
    )

    response: Final = _call_with_strict_tools_fallback(
        bedrock, _NATIVE_RETRY_WITH_STRICT_TOOLS, _STRICT_FALLBACK, "us.anthropic.claude-sonnet-4-6"
    )

    assert isinstance(response, ModelResponse) and response.id == "answer-3"
    assert bedrock.sent == (None, _NATIVE_RETRY_WITH_STRICT_TOOLS, _COMPOSED_FALLBACK)
