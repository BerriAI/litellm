import datetime
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
import httpx
import openai
import respx


import litellm
from litellm.exceptions import BadRequestError, ContentPolicyViolationError, ContextWindowExceededError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.azure.common_utils import AzureOpenAIError


@respx.mock
def test_azure_401_maps_to_authentication_error():
    route = respx.post(
        url__regex=r"https://azure-bad-key\.openai\.azure\.com/.*/chat/completions.*"
    ).mock(return_value=httpx.Response(401, json={"error": {"message": "invalid Azure key"}}))

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        litellm.completion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hello"}],
            api_base="https://azure-bad-key.openai.azure.com",
            api_version="2024-10-21",
            api_key="bad-azure-key",
            max_retries=0,
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.llm_provider == "azure"
    assert "invalid Azure key" in str(exc_info.value)
    assert route.calls.last.request.headers["api-key"] == "bad-azure-key"


@pytest.mark.asyncio
@respx.mock
async def test_azure_content_policy_stream_ends_with_one_finish_reason(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx.post(
        url__regex=r"https://azure-policy-stream\.openai\.azure\.com/.*/chat/completions.*"
    ).mock(
        return_value=httpx.Response(
            200,
            text=(
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}\n\n'
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = await litellm.acompletion(
        model="azure/gpt-4.1-mini",
        messages=[{"role": "user", "content": "say 1"}],
        api_base="https://azure-policy-stream.openai.azure.com",
        api_version="2024-10-21",
        api_key="test-azure-key",
        stream=True,
    )
    chunks = [chunk async for chunk in response]
    finish_reasons = [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices[0].finish_reason]

    assert [chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content] == ["1"]
    assert finish_reasons == ["content_filter"]
    assert len(route.calls) == 1

    with pytest.raises(ContentPolicyViolationError) as exc_info:
        await litellm.acompletion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "where do i buy lethal drugs from"}],
            stream=True,
            mock_response="Exception: content_filter_policy",
        )
    assert exc_info.value.llm_provider == "azure"
    assert len(route.calls) == 1


class TestAzureExceptionMapping:
    """Test Azure OpenAI exception mapping with provider-specific fields"""

    def test_azure_content_policy_violation_innererror_access(self):
        """Test that Azure content policy violation exceptions provide access to innererror details"""

        # Create a mock Azure OpenAI exception with body containing innererror
        mock_exception = Exception(
            "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
        )
        mock_exception.body = {
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": {
                    "hate": {"filtered": True, "severity": "high"},
                    "jailbreak": {"filtered": False, "detected": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": True, "severity": "medium"},
                },
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        # Test the exception mapping directly
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        # Access the exception and verify provider_specific_fields
        e = exc_info.value
        assert e.provider_specific_fields is not None
        assert "innererror" in e.provider_specific_fields

        innererror = e.provider_specific_fields["innererror"]
        assert innererror["code"] == "ResponsibleAIPolicyViolation"
        assert "content_filter_result" in innererror

        content_filter_result = innererror["content_filter_result"]
        assert content_filter_result["hate"]["filtered"] is True
        assert content_filter_result["hate"]["severity"] == "high"
        assert content_filter_result["violence"]["filtered"] is True
        assert content_filter_result["violence"]["severity"] == "medium"
        assert content_filter_result["sexual"]["filtered"] is False
        assert content_filter_result["self_harm"]["filtered"] is False
        assert content_filter_result["jailbreak"]["filtered"] is False

    def test_azure_content_policy_violation_different_categories(self):
        """Test Azure content policy violation with different filtering categories"""

        # Mock exception with different content filter results
        mock_exception = Exception(
            "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
        )
        mock_exception.body = {
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"filtered": True, "detected": True},
                    "self_harm": {"filtered": True, "severity": "high"},
                    "sexual": {"filtered": True, "severity": "medium"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        # Test the exception mapping directly with different violation type
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        # Verify provider_specific_fields contains the expected innererror structure
        e = exc_info.value
        assert e.provider_specific_fields is not None
        print("got provider_specific_fields=", e.provider_specific_fields)
        innererror = e.provider_specific_fields["innererror"]
        content_filter_result = innererror["content_filter_result"]

        # Check different filter categories
        assert content_filter_result["sexual"]["filtered"] is True
        assert content_filter_result["sexual"]["severity"] == "medium"
        assert content_filter_result["self_harm"]["filtered"] is True
        assert content_filter_result["self_harm"]["severity"] == "high"
        assert content_filter_result["jailbreak"]["filtered"] is True
        assert content_filter_result["jailbreak"]["detected"] is True
        assert content_filter_result["hate"]["filtered"] is False
        assert content_filter_result["violence"]["filtered"] is False

    def test_azure_content_policy_violation_missing_innererror(self):
        """Test Azure content policy violation when innererror is missing from response"""

        # Mock exception without body attribute
        mock_exception = Exception(
            "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
        )
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response
        # Note: no mock_exception.body attribute set

        # Test the exception mapping directly
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        # Verify that even without innererror, the exception is still raised properly
        e = exc_info.value
        print("got exception=", e)
        # provider_specific_fields should still exist but innererror should be None
        assert e.provider_specific_fields is not None
        assert e.provider_specific_fields.get("innererror") is None

    def test_azure_content_policy_violation_non_dict_body(self):
        """Test Azure content policy violation when body is not a dictionary"""

        # Mock exception with non-dict body
        mock_exception = Exception(
            "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
        )
        mock_exception.body = "invalid body format"
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        # Test the exception mapping directly
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        # Verify that with invalid body format, innererror should be None
        e = exc_info.value
        print("got exception=", e)
        print("exception fields=", vars(e))
        assert e.provider_specific_fields is not None
        assert e.provider_specific_fields.get("innererror") is None

    def test_azure_images_content_policy_violation_preserves_nested_inner_error(self):
        """Azure Images endpoints return errors nested under body['error'] with inner_error.

        Ensure we:
        - Detect the violation via structured payload (code=content_policy_violation)
        - Preserve code/type/message
        - Surface inner_error + revised_prompt + content_filter_results
        """

        mock_exception = Exception("Bad request")  # does not include policy substrings
        mock_exception.body = {
            "error": {
                "code": "content_policy_violation",
                "inner_error": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_results": {
                        "violence": {"filtered": True, "severity": "low"}
                    },
                    "revised_prompt": "revised",
                },
                "message": "Your request was rejected as a result of our safety system.",
                "type": "invalid_request_error",
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/dall-e-3",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        e = exc_info.value

        # OpenAI-style error fields should be populated
        assert getattr(e, "code", None) == "content_policy_violation"
        assert getattr(e, "type", None) == "invalid_request_error"
        assert "safety system" in str(e)

        # Provider-specific nested details must be preserved
        assert e.provider_specific_fields is not None
        assert (
            e.provider_specific_fields["inner_error"]["code"]
            == "ResponsibleAIPolicyViolation"
        )
        assert e.provider_specific_fields["inner_error"]["revised_prompt"] == "revised"
        assert (
            e.provider_specific_fields["inner_error"]["content_filter_results"][
                "violence"
            ]["filtered"]
            is True
        )

    def test_azure_content_policy_violation_detected_via_inner_error_code(self):
        """Regression test for #20811: Azure returns inner_error with
        ResponsibleAIPolicyViolation but the top-level error message is
        generic.  Previously this fell through to the generic
        BadRequestError handler and all error details were lost."""

        mock_exception = Exception("Bad request")
        # This body structure mirrors what Azure OpenAI Images API returns
        # for DALL-E 3 content policy violations (issue #20811).
        mock_exception.body = {
            "error": {
                "code": "content_policy_violation",
                "inner_error": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "profanity": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": True, "severity": "low"},
                    },
                    "revised_prompt": (
                        "A dark and intense illustration of a man "
                        "in a dramatic action scene."
                    ),
                },
                "message": (
                    "Your request was rejected as a result of our safety system."
                ),
                "type": "invalid_request_error",
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/dall-e-3",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        e = exc_info.value
        # Must surface as ContentPolicyViolationError, not generic BadRequestError
        assert "safety system" in str(e)
        assert e.provider_specific_fields is not None
        inner = e.provider_specific_fields["inner_error"]
        assert inner["code"] == "ResponsibleAIPolicyViolation"
        assert inner["content_filter_results"]["violence"]["filtered"] is True
        assert inner["revised_prompt"] is not None

    def test_azure_policy_violation_detected_via_inner_error_without_top_code(self):
        """When the top-level code is NOT 'content_policy_violation' but
        inner_error.code IS 'ResponsibleAIPolicyViolation', the error
        should still be recognized as a content policy violation."""

        mock_exception = Exception("Some error")
        mock_exception.body = {
            "error": {
                "code": "BadRequest",
                "inner_error": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_results": {
                        "violence": {"filtered": True, "severity": "medium"},
                    },
                },
                "message": "The request was rejected.",
                "type": "invalid_request_error",
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/dall-e-3",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        e = exc_info.value
        assert e.provider_specific_fields is not None
        assert (
            e.provider_specific_fields["inner_error"]["code"]
            == "ResponsibleAIPolicyViolation"
        )

    def test_azure_image_polling_error_preserves_body(self):
        """Verify that AzureOpenAIError raised from the DALL-E polling path
        carries the structured body so exception_type() can inspect it."""
        from litellm.llms.azure.common_utils import AzureOpenAIError

        error_payload = {
            "status": "failed",
            "error": {
                "code": "content_policy_violation",
                "message": "Your request was rejected.",
                "inner_error": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_results": {
                        "violence": {"filtered": True, "severity": "low"},
                    },
                },
            },
        }

        # Simulate what the fixed polling path now does
        _error_body = error_payload.get("error", error_payload)
        _error_msg = (
            _error_body.get("message", "Image generation failed")
            if isinstance(_error_body, dict)
            else json.dumps(error_payload)
        )
        exc = AzureOpenAIError(
            status_code=400,
            message=_error_msg,
            body=error_payload,
        )

        assert exc.body is not None
        assert isinstance(exc.body, dict)
        assert exc.body["error"]["code"] == "content_policy_violation"
        assert "Your request was rejected" in exc.message

    def test_azure_safety_system_message_detected_as_policy_violation(self):
        """Azure's rejection message 'Your request was rejected as a result
        of our safety system' should be detected by string matching even
        when the structured body is unavailable."""

        mock_exception = Exception(
            "Your request was rejected as a result of our safety system. "
            "The revised prompt may contain text that is not allowed."
        )
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(ContentPolicyViolationError):
            exception_type(
                model="azure/dall-e-3",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

    def test_invalid_encrypted_content_error_with_helpful_message(self):
        """Test that invalid_encrypted_content errors include helpful guidance
        about enabling encrypted_content_affinity."""
        from litellm.exceptions import BadRequestError

        mock_exception = Exception(
            "The encrypted content gAAAAABpnW_yEYmSNEyOG... could not be verified. "
            "Reason: Encrypted content organization_id did not match the target organization."
        )
        mock_exception.body = {
            "error": {
                "message": "The encrypted content could not be verified.",
                "type": "invalid_request_error",
                "code": "invalid_encrypted_content",
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(BadRequestError) as exc_info:
            exception_type(
                model="azure/gpt-5.1-codex",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        error = exc_info.value
        assert "encrypted_content_affinity" in error.message
        assert "enable_pre_call_checks" in error.message
        assert "optional_pre_call_checks" in error.message
        assert "docs.litellm.ai" in error.message

    def test_openai_invalid_encrypted_content_error(self):
        """Test that OpenAI invalid_encrypted_content errors also get helpful guidance."""
        from litellm.exceptions import BadRequestError

        mock_exception = Exception("The encrypted content could not be verified.")
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(BadRequestError) as exc_info:
            exception_type(
                model="gpt-5.1-codex",
                original_exception=mock_exception,
                custom_llm_provider="openai",
            )

        error = exc_info.value
        assert "encrypted_content_affinity" in error.message
        assert "enable_pre_call_checks" in error.message


_AZURE_API_BASE: Final = "https://azure-context.openai.azure.com"
_AZURE_CALL_KWARGS: Final = {
    "api_base": _AZURE_API_BASE,
    "api_version": "2024-10-21",
    "api_key": "test-azure-key",
    "max_retries": 0,
}


_CONTEXT_LENGTH_ERROR: Final = {
    "message": "Your input exceeds the context window of this model. Please adjust your input and try again.",
    "type": "invalid_request_error",
    "param": "input",
    "code": "context_length_exceeded",
}


def _azure_error_response(status_code: int, error: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        status_code, text=json.dumps({"error": error}, indent=2), headers={"content-type": "application/json"}
    )


async def _azure_chat(*, stream: bool) -> None:
    messages: Final = [{"role": "user", "content": "hi"}]
    if stream:
        stream_response: Final = await litellm.acompletion(
            model="azure/gpt-4o", messages=messages, stream=True, **_AZURE_CALL_KWARGS
        )
        assert isinstance(stream_response, litellm.CustomStreamWrapper)
        async for _chunk in stream_response:
            pass
    else:
        litellm.completion(model="azure/gpt-4o", messages=messages, **_AZURE_CALL_KWARGS)


async def _azure_responses(*, is_async: bool) -> None:
    if is_async:
        await litellm.aresponses(model="azure/gpt-4o", input="hi", **_AZURE_CALL_KWARGS)
    else:
        litellm.responses(model="azure/gpt-4o", input="hi", **_AZURE_CALL_KWARGS)


async def _azure_embedding() -> None:
    litellm.embedding(model="azure/text-embedding-3-small", input="hi", **_AZURE_CALL_KWARGS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: _azure_chat(stream=False), id="chat"),
        pytest.param(lambda: _azure_chat(stream=True), id="async-chat-stream"),
        pytest.param(lambda: _azure_responses(is_async=False), id="responses"),
        pytest.param(lambda: _azure_responses(is_async=True), id="async-responses"),
        pytest.param(_azure_embedding, id="embedding"),
    ],
)
@pytest.mark.parametrize(
    ("code", "expected"),
    [("context_length_exceeded", ContextWindowExceededError), ("invalid_value", BadRequestError)],
)
@respx.mock
async def test_azure_400_is_classified_by_error_code(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], Awaitable[None]], code: str, expected: type[Exception]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx.post(url__startswith=_AZURE_API_BASE).mock(
        return_value=_azure_error_response(400, {**_CONTEXT_LENGTH_ERROR, "code": code})
    )

    with pytest.raises(BadRequestError) as exc_info:
        await call()

    assert type(exc_info.value) is expected


@pytest.mark.asyncio
@respx.mock
async def test_azure_context_length_code_triggers_context_window_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx.post(url__startswith=_AZURE_API_BASE).mock(
        return_value=_azure_error_response(400, _CONTEXT_LENGTH_ERROR)
    )
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "small", "litellm_params": {"model": "azure/gpt-4o", **_AZURE_CALL_KWARGS}},
            {"model_name": "large", "litellm_params": {"model": "gpt-4.1", "mock_response": "served by large"}},
        ],
        context_window_fallbacks=[{"small": ["large"]}],
        num_retries=0,
    )

    response: Final = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # untyped **kwargs
        model="small", messages=[{"role": "user", "content": "hi"}]
    )

    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "served by large"
    assert len(route.calls) == 1


def _azure_chat_error(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> openai.APIError:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx.post(url__startswith=_AZURE_API_BASE).mock(return_value=response)
    with pytest.raises(openai.APIError) as exc_info:
        litellm.completion(model="azure/gpt-4o", messages=[{"role": "user", "content": "hi"}], **_AZURE_CALL_KWARGS)
    return exc_info.value


@respx.mock
def test_azure_context_window_error_keeps_upstream_code_and_param(monkeypatch: pytest.MonkeyPatch) -> None:
    error: Final = _azure_chat_error(monkeypatch, _azure_error_response(400, _CONTEXT_LENGTH_ERROR))

    assert isinstance(error, ContextWindowExceededError)
    assert error.code == "context_length_exceeded"
    assert error.param == "input"


@pytest.mark.parametrize("status_code", [429, 500])
@respx.mock
def test_azure_non_400_with_context_length_code_keeps_its_status(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    error: Final = _azure_chat_error(monkeypatch, _azure_error_response(status_code, _CONTEXT_LENGTH_ERROR))

    assert not isinstance(error, ContextWindowExceededError)
    assert getattr(error, "status_code", None) == status_code


@respx.mock
def test_azure_content_policy_wins_over_context_length_code(monkeypatch: pytest.MonkeyPatch) -> None:
    policy_error: Final = {**_CONTEXT_LENGTH_ERROR, "inner_error": {"code": "ResponsibleAIPolicyViolation"}}

    error: Final = _azure_chat_error(monkeypatch, _azure_error_response(400, policy_error))

    assert type(error) is ContentPolicyViolationError


_AZURE_REFUSAL_BODY: Final = '{"error":{"message":"azure refused","type":"upstream_refusal","code":"refused"}}'
_AZURE_DEPLOYMENT_NOT_FOUND_BODY: Final = '{"error":{"code":"DeploymentNotFound","message":"no such deployment"}}'
_AZURE_INTERNAL_ERROR_BODY: Final = '{"error":{"code":"InternalServerError","message":"Internal server error"}}'
_AZURE_CONTENT_FILTER_BODY: Final = json.dumps(
    {
        "error": {
            "message": "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
            "type": None,
            "param": "prompt",
            "code": "content_filter",
            "status": 400,
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": {"violence": {"filtered": True, "severity": "high"}},
            },
        }
    }
)
_AZURE_KEY: Final = "azure-provider-secret"
_AZURE_BASE: Final = "https://azure-upstream.openai.azure.com"
_AZURE_ARGS: Final = {"api_base": _AZURE_BASE, "api_version": "2024-10-21", "api_key": _AZURE_KEY, "max_retries": 0}


async def _azure_completion() -> None:
    litellm.completion(model="azure/gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}], **_AZURE_ARGS)


async def _azure_completion_stream() -> None:
    list(
        litellm.completion(
            model="azure/gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}], stream=True, **_AZURE_ARGS
        )
    )


async def _azure_acompletion() -> None:
    await litellm.acompletion(model="azure/gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}], **_AZURE_ARGS)


async def _azure_acompletion_stream() -> None:
    stream: Final = await litellm.acompletion(
        model="azure/gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}], stream=True, **_AZURE_ARGS
    )
    async for _ in stream:
        pass


async def _azure_embedding() -> None:
    litellm.embedding(model="azure/text-embedding-3-large", input=["hi"], **_AZURE_ARGS)


async def _azure_aembedding() -> None:
    await litellm.aembedding(model="azure/text-embedding-3-large", input=["hi"], **_AZURE_ARGS)


async def _azure_image_generation() -> None:
    litellm.image_generation(model="azure/gpt-image-2", prompt="a cat", **_AZURE_ARGS)


async def _azure_text_completion() -> None:
    litellm.text_completion(model="azure_text/gpt-35-turbo-instruct", prompt="hi", **_AZURE_ARGS)


async def _azure_text_completion_stream() -> None:
    list(litellm.text_completion(model="azure_text/gpt-35-turbo-instruct", prompt="hi", stream=True, **_AZURE_ARGS))


async def _azure_atext_completion() -> None:
    await litellm.atext_completion(model="azure_text/gpt-35-turbo-instruct", prompt="hi", **_AZURE_ARGS)


async def _azure_atext_completion_stream() -> None:
    stream: Final = await litellm.atext_completion(
        model="azure_text/gpt-35-turbo-instruct", prompt="hi", stream=True, **_AZURE_ARGS
    )
    async for _ in stream:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    (
        _azure_completion,
        _azure_completion_stream,
        _azure_acompletion,
        _azure_acompletion_stream,
        _azure_embedding,
        _azure_aembedding,
        _azure_image_generation,
        _azure_text_completion,
        _azure_text_completion_stream,
        _azure_atext_completion,
        _azure_atext_completion_stream,
    ),
    ids=lambda route: route.__name__.removeprefix("_azure_"),
)
@pytest.mark.parametrize(
    ("status_code", "body", "mapped_error"),
    (
        (400, _AZURE_REFUSAL_BODY, BadRequestError),
        (400, _AZURE_CONTENT_FILTER_BODY, BadRequestError),
        (403, _AZURE_REFUSAL_BODY, litellm.APIError),
        (404, _AZURE_DEPLOYMENT_NOT_FOUND_BODY, litellm.NotFoundError),
        (500, _AZURE_INTERNAL_ERROR_BODY, openai.APIError),
        (502, _AZURE_REFUSAL_BODY, litellm.BadGatewayError),
        (504, _AZURE_REFUSAL_BODY, litellm.Timeout),
    ),
)
@respx.mock
async def test_azure_error_keeps_the_upstream_body_without_the_api_key(
    route: Callable[[], Awaitable[None]],
    status_code: int,
    body: str,
    mapped_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx.post(url__regex=r"https://azure-upstream\.openai\.azure\.com/.*").mock(
        return_value=httpx.Response(status_code, content=body, headers={"content-type": "application/json"})
    )

    with pytest.raises(mapped_error) as raised:
        await route()

    error_response: Final = getattr(raised.value, "response")
    assert error_response.status_code == status_code
    assert error_response.text == body
    assert error_response.json() == json.loads(body)
    for exposed in (error_response.request, getattr(raised.value, "request")):
        assert _AZURE_KEY not in str(exposed.headers)
        assert exposed.content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    (
        _azure_completion,
        _azure_completion_stream,
        _azure_acompletion,
        _azure_acompletion_stream,
        _azure_embedding,
        _azure_aembedding,
        _azure_text_completion,
        _azure_atext_completion,
        _azure_atext_completion_stream,
    ),
    ids=lambda route: route.__name__.removeprefix("_azure_"),
)
@respx.mock
async def test_azure_error_response_keeps_the_upstream_elapsed_time(
    route: Callable[[], Awaitable[None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx.post(url__regex=r"https://azure-upstream\.openai\.azure\.com/.*").mock(
        return_value=httpx.Response(400, content=_AZURE_REFUSAL_BODY, headers={"content-type": "application/json"})
    )

    with pytest.raises(BadRequestError) as raised:
        await route()

    assert raised.value.response.elapsed > datetime.timedelta(0)


def test_azure_client_side_missing_key_error_keeps_the_empty_placeholder_response() -> None:
    missing_key: Final = openai.OpenAIError(
        "The api_key client option must be set either by passing api_key to the client or by setting the "
        "OPENAI_API_KEY environment variable"
    )

    with pytest.raises(litellm.AuthenticationError) as raised:
        exception_type(model="azure/gpt-4.1-mini", original_exception=missing_key, custom_llm_provider="azure")

    assert raised.value.response.status_code == 401
    assert raised.value.response.text == ""


@pytest.mark.parametrize(
    ("status_code", "message", "mapped_error"),
    (
        (500, "azure_client is not an instance of AzureOpenAI or OpenAI", litellm.APIError),
        (408, "the deployment did not answer in time", litellm.Timeout),
        (504, "the deployment did not answer in time", litellm.Timeout),
    ),
)
def test_azure_error_litellm_raises_itself_has_no_upstream_response(
    status_code: int, message: str, mapped_error: type[Exception]
) -> None:
    with pytest.raises(mapped_error) as raised:
        exception_type(
            model="azure/gpt-4.1-mini",
            original_exception=AzureOpenAIError(status_code=status_code, message=message),
            custom_llm_provider="azure",
        )

    assert getattr(raised.value, "response", None) is None
