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
_REJECTION: Final = BedrockError(
    status_code=400,
    message='{"message":"The model returned the following errors: output_config.format: Extra inputs are not permitted"}',
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
def test_a_native_output_rejection_is_sent_again_once_with_the_fallback_params(asynchronous: bool) -> None:
    bedrock: Final = ScriptedBedrock(errors=(_REJECTION, None), asynchronous=asynchronous)

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
