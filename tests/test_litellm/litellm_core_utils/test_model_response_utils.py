from litellm.litellm_core_utils.model_response_utils import (
    is_model_response_stream_empty,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def test_is_model_response_stream_empty():
    chunk = ModelResponseStream(
        id="chatcmpl-C3sWKN2RWbn6CZ1IGU2QCpRh4RhYf",
        created=1755040596,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        system_fingerprint="fp_34a54ae93c",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
    )
    assert is_model_response_stream_empty(chunk) is True


def test_is_model_response_stream_empty_with_custom_value():
    chunk = ModelResponseStream(
        id="chatcmpl-C3sWKN2RWbn6CZ1IGU2QCpRh4RhYf",
        created=1755040596,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        system_fingerprint="fp_34a54ae93c",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
    )

    setattr(chunk.choices[0].delta, "custom_field", "test")
    assert is_model_response_stream_empty(chunk) is False
