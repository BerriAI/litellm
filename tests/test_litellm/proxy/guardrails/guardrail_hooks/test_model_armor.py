import asyncio
import base64
import io
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import httpx
from fastapi import HTTPException

import litellm
import litellm.types.utils
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.model_armor import ModelArmorGuardrail
from litellm.types.guardrails import GuardrailEventHooks


@pytest.mark.asyncio
async def test_model_armor_pre_call_hook_sanitization():
    """Test Model Armor pre-call hook with content sanitization"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_request_content=True,
    )

    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "sdp": {
                        "sdpFilterResult": {
                            "deidentifyResult": {
                                "matchState": "MATCH_FOUND",
                                "data": {
                                    "text": "Hello, my phone number is [REDACTED]"
                                },
                            }
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"}
            ],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Assert the message was sanitized
        assert (
            result["messages"][0]["content"] == "Hello, my phone number is [REDACTED]"
        )

        # Verify API was called correctly
        # Note: we need to use the captured mock from the patch if we want to assert on it
        # But for now, we'll just verify the behavior.
        # Actually, let's capture it.


@pytest.mark.asyncio
async def test_model_armor_pre_call_hook_blocked():
    """Test Model Armor pre-call hook when content is blocked"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock the Model Armor API response for blocked content
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "rai": {
                        "raiFilterResult": {
                            "matchState": "MATCH_FOUND",
                            "raiFilterTypeResults": {
                                "dangerous": {
                                    "matchState": "MATCH_FOUND",
                                    "reason": "Prohibited content detected",
                                }
                            },
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Some harmful content"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should raise HTTPException for blocked content
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=mock_cache,
                data=request_data,
                call_type="completion",
            )

        assert exc_info.value.status_code == 400
        assert "Content blocked by Model Armor" in str(exc_info.value.detail)

        # IMPORTANT: Verify that applied_guardrails is populated even when blocked
        # This is a regression test for the issue where applied_guardrails was null when blocked
        assert "applied_guardrails" in request_data["metadata"]
        assert "model-armor-test" in request_data["metadata"]["applied_guardrails"]


@pytest.mark.asyncio
async def test_model_armor_post_call_hook_sanitization():
    """Test Model Armor post-call hook with response sanitization"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_response_content=True,
    )

    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "sdp": {
                        "sdpFilterResult": {
                            "deidentifyResult": {
                                "matchState": "MATCH_FOUND",
                                "data": {"text": "Here is the information: [REDACTED]"},
                            }
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        # Create a mock response
        mock_llm_response = litellm.ModelResponse()
        mock_llm_response.choices = [
            litellm.Choices(
                message=litellm.Message(
                    content="Here is the information: Credit card 1234-5678-9012-3456"
                )
            )
        ]

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's my credit card?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        await guardrail.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_llm_response,
        )

        # Assert the response was sanitized
        assert (
            mock_llm_response.choices[0].message.content
            == "Here is the information: [REDACTED]"
        )


@pytest.mark.asyncio
async def test_model_armor_post_call_hook_blocked():
    """Test Model Armor post-call hook when response is blocked and applied_guardrails is populated"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock the Model Armor API response for blocked content
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "rai": {
                        "raiFilterResult": {
                            "matchState": "MATCH_FOUND",
                            "raiFilterTypeResults": {
                                "dangerous": {
                                    "matchState": "MATCH_FOUND",
                                    "reason": "Harmful response detected",
                                }
                            },
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        # Create a mock response
        mock_llm_response = litellm.ModelResponse()
        mock_llm_response.choices = [
            litellm.Choices(
                message=litellm.Message(content="Here is some harmful content...")
            )
        ]

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Some prompt"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should raise HTTPException for blocked response
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=request_data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_llm_response,
            )

        assert exc_info.value.status_code == 400
        assert "Response blocked by Model Armor" in str(exc_info.value.detail)

        # IMPORTANT: Verify that applied_guardrails is populated even when blocked
        # This is a regression test for the issue where applied_guardrails was null when blocked
        assert "applied_guardrails" in request_data["metadata"]
        assert "model-armor-test" in request_data["metadata"]["applied_guardrails"]


@pytest.mark.asyncio
async def test_model_armor_with_list_content():
    """Test Model Armor with messages containing list content"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={"sanitizationResult": {"filterMatchState": "NO_MATCH_FOUND"}}
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ) as mock_post:
        request_data = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello world"},
                        {"type": "text", "text": "How are you?"},
                    ],
                }
            ],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Verify the content was extracted correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert (
            call_args[1]["json"]["userPromptData"]["text"] == "Hello worldHow are you?"
        )


@pytest.mark.asyncio
async def test_model_armor_api_error_handling():
    """Test Model Armor error handling when API returns error"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        fail_on_error=True,
    )

    # Mock the Model Armor API error response
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should raise HTTPException for API error
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=mock_cache,
                data=request_data,
                call_type="completion",
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Model Armor API error (upstream 500)"
        assert "Internal Server Error" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_model_armor_credentials_handling():
    """Test Model Armor handling of different credential types"""
    try:
        from google.auth.credentials import Credentials
    except ImportError:
        # If google.auth is not installed, skip this test
        pytest.skip("google.auth not installed")
        return

    # Test with string credentials (file path)
    with patch("os.path.exists", return_value=True):
        with patch(
            "builtins.open",
            mock_open(
                read_data='{"type": "service_account", "project_id": "test-project"}'
            ),
        ):
            with patch.object(
                ModelArmorGuardrail, "_credentials_from_service_account"
            ) as mock_creds:
                mock_creds_obj = Mock()
                mock_creds_obj.token = "test-token"
                mock_creds_obj.expired = False
                mock_creds_obj.project_id = "test-project"  # Add project_id
                mock_creds.return_value = mock_creds_obj

                guardrail = ModelArmorGuardrail(
                    template_id="test-template",
                    credentials="/path/to/creds.json",
                    project_id="test-project",  # Provide project_id
                )

                # Force credential loading
                creds, project_id = guardrail.load_auth(
                    credentials="/path/to/creds.json", project_id="test-project"
                )

                assert mock_creds.called
                assert project_id == "test-project"


@pytest.mark.asyncio
async def test_model_armor_streaming_response():
    """Test Model Armor with streaming responses"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_response_content=True,
    )

    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "NO_MATCH_FOUND",
                "sanitizedText": "Sanitized response",
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ) as mock_post:
        # Create mock streaming chunks
        async def mock_stream():
            chunks = [
                litellm.ModelResponseStream(
                    choices=[
                        litellm.types.utils.StreamingChoices(
                            delta=litellm.types.utils.Delta(content="Sensitive ")
                        )
                    ]
                ),
                litellm.ModelResponseStream(
                    choices=[
                        litellm.types.utils.StreamingChoices(
                            delta=litellm.types.utils.Delta(content="information")
                        )
                    ]
                ),
            ]
            for chunk in chunks:
                yield chunk

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Tell me secrets"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Process streaming response
        result_chunks = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_stream(),
            request_data=request_data,
        ):
            result_chunks.append(chunk)

        # Should have processed the chunks through Model Armor
        assert len(result_chunks) > 0
        mock_post.assert_called()


@pytest.mark.asyncio
async def test_model_armor_streaming_block_yields_sse_error():
    """Test that streaming content block yields SSE error event instead of raising HTTPException."""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock Model Armor API response that triggers a block (SDP MATCH_FOUND)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "sdp": {
                        "sdpFilterResult": {
                            "inspectResult": {
                                "matchState": "MATCH_FOUND",
                                "findings": [
                                    {
                                        "infoType": "PASSWORD",
                                        "likelihood": "VERY_LIKELY",
                                    }
                                ],
                            }
                        }
                    }
                },
            }
        }
    )

    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):

        async def mock_stream():
            chunks = [
                litellm.ModelResponseStream(
                    choices=[
                        litellm.types.utils.StreamingChoices(
                            delta=litellm.types.utils.Delta(content="My password is ")
                        )
                    ]
                ),
                litellm.ModelResponseStream(
                    choices=[
                        litellm.types.utils.StreamingChoices(
                            delta=litellm.types.utils.Delta(content="hunter2")
                        )
                    ]
                ),
            ]
            for chunk in chunks:
                yield chunk

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's your password?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        result_chunks = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_stream(),
            request_data=request_data,
        ):
            result_chunks.append(chunk)

        # Should yield exactly one SSE error event (not raise HTTPException)
        assert len(result_chunks) == 1
        error_data = json.loads(result_chunks[0].removeprefix("data: "))
        assert "error" in error_data
        assert int(error_data["error"]["code"]) == 400


@pytest.mark.asyncio
async def test_model_armor_api_failure_returns_400():
    """Test that Model Armor API failures raise HTTP 400, not the upstream status code."""
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock a 500 response from the Model Armor GCP API
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_model_armor_request(
                content="test content",
                source="user_prompt",
            )

        # Should be 400, NOT the upstream 500
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Model Armor API error (upstream 500)"
        assert "Internal Server Error" not in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("sanitize", [True, False])
async def test_model_armor_error_output_sanitization(sanitize: bool):
    marker = "SYNTHETIC_MODEL_ARMOR_MARKER"
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        guardrail_name="model-armor-test",
        sanitize_error_detail=sanitize,
    )
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    error_response = AsyncMock(status_code=500, text=marker)
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=error_response)
    ), patch.object(verbose_proxy_logger, "debug") as debug_log, patch.object(
        verbose_proxy_logger, "error"
    ) as error_log, pytest.raises(HTTPException) as exc_info:
        await guardrail.make_model_armor_request(content=marker)

    direct_log = f"{debug_log.call_args_list} {error_log.call_args_list}"
    if sanitize:
        assert marker not in str(exc_info.value.detail)
        assert marker not in direct_log
    else:
        assert marker in str(exc_info.value.detail)
        assert marker in direct_log


@pytest.mark.asyncio
@pytest.mark.parametrize("sanitize", [True, False])
async def test_model_armor_handler_raised_http_error_sanitized(sanitize: bool):
    """The real AsyncHTTPHandler raises on non-2xx via raise_for_status, so a non-200
    never returns a response object. The raised MaskedHTTPStatusError carries the raw
    upstream body in its message; the guardrail must convert it to a sanitized
    HTTPException instead of letting it bubble raw to callers and logs."""
    marker = "SYNTHETIC_MODEL_ARMOR_MARKER"
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        guardrail_name="model-armor-test",
        sanitize_error_detail=sanitize,
    )
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    request = httpx.Request("POST", "https://modelarmor.example.test/v1")
    upstream = httpx.Response(403, content=marker.encode(), request=request)
    original = httpx.HTTPStatusError("Forbidden", request=request, response=upstream)
    masked = MaskedHTTPStatusError(original, message=marker, text=marker)

    with patch.object(
        guardrail.async_handler, "post", AsyncMock(side_effect=masked)
    ), patch.object(verbose_proxy_logger, "debug") as debug_log, patch.object(
        verbose_proxy_logger, "error"
    ) as error_log, pytest.raises(HTTPException) as exc_info:
        await guardrail.make_model_armor_request(content=marker)

    direct_log = f"{debug_log.call_args_list} {error_log.call_args_list}"
    assert "403" in str(exc_info.value.detail)
    if sanitize:
        assert marker not in str(exc_info.value.detail)
        assert marker not in direct_log
    else:
        assert marker in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("sanitize", [True, False])
async def test_model_armor_post_call_logging_redacts_scanned_content(sanitize: bool):
    marker = "SYNTHETIC_POST_CALL_MARKER"
    armor_response = {
        "sanitizationResult": {
            "filterMatchState": "NO_MATCH_FOUND",
            "filterResults": {
                "sdp": {
                    "sdpFilterResult": {
                        "deidentifyResult": {
                            "matchState": "MATCH_FOUND",
                            "data": {"text": marker},
                        }
                    }
                }
            },
        }
    }
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        guardrail_name="model-armor-test",
        mask_response_content=True,
        sanitize_error_detail=sanitize,
    )
    guardrail.make_model_armor_request = AsyncMock(return_value=armor_response)
    guardrail.should_run_guardrail = Mock(return_value=True)

    mock_llm_response = litellm.ModelResponse()
    mock_llm_response.choices = [
        litellm.Choices(message=litellm.Message(content="model output"))
    ]
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "synthetic input"}],
        "metadata": {},
        "litellm_logging_obj": MagicMock(),
    }

    with patch(
        "litellm.proxy.common_utils.callback_utils.add_guardrail_response_to_standard_logging_object"
    ) as add_logging:
        await guardrail.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=UserAPIKeyAuth(),
            response=mock_llm_response,
        )

    logged = add_logging.call_args.kwargs["guardrail_response"]
    assert logged["guardrail_status"] == "success"
    logged_armor_response = logged["guardrail_response"]["model_armor_response"]
    if sanitize:
        assert marker not in str(logged_armor_response)
        assert (
            logged_armor_response["sanitizationResult"]["filterResults"]["sdp"][
                "sdpFilterResult"
            ]["deidentifyResult"]["matchState"]
            == "MATCH_FOUND"
        )
    else:
        assert logged_armor_response == armor_response


@pytest.mark.asyncio
@pytest.mark.parametrize("sanitize", [True, False])
async def test_model_armor_streaming_logging_redacts_scanned_content(sanitize: bool):
    marker = "SYNTHETIC_STREAMING_MARKER"
    armor_response = {
        "sanitizationResult": {
            "filterMatchState": "NO_MATCH_FOUND",
            "sanitizedText": marker,
        }
    }
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        guardrail_name="model-armor-test",
        sanitize_error_detail=sanitize,
    )
    guardrail.make_model_armor_request = AsyncMock(return_value=armor_response)
    guardrail.should_run_guardrail = Mock(return_value=True)

    async def mock_stream():
        yield litellm.ModelResponseStream(
            choices=[
                litellm.types.utils.StreamingChoices(
                    delta=litellm.types.utils.Delta(content="streamed output")
                )
            ]
        )

    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "synthetic input"}],
        "metadata": {},
    }

    async for _ in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        response=mock_stream(),
        request_data=request_data,
    ):
        pass

    logged_response = request_data["metadata"]["_model_armor_response"]
    if sanitize:
        assert logged_response == {
            "sanitizationResult": {
                "filterMatchState": "NO_MATCH_FOUND",
                "sanitizedText": "[REDACTED]",
            }
        }
        assert marker not in str(logged_response)
    else:
        assert logged_response == armor_response


@pytest.mark.asyncio
@pytest.mark.parametrize("sanitize", [True, False])
async def test_model_armor_match_found_sanitizes_caller_and_logging(sanitize: bool):
    marker = "SYNTHETIC_MATCH_FOUND_MARKER"
    armor_response = {
        "sanitizationResult": {
            "filterResults": {
                "sdp": {
                    "sdpFilterResult": {
                        "inspectResult": {
                            "matchState": "MATCH_FOUND",
                            "findings": [{"marker": marker}],
                        }
                    }
                }
            }
        }
    }
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        guardrail_name="model-armor-test",
        event_hook=[GuardrailEventHooks.pre_mcp_call],
        sanitize_error_detail=sanitize,
    )
    guardrail.make_model_armor_request = AsyncMock(return_value=armor_response)
    guardrail.should_run_guardrail = Mock(return_value=True)
    request_data = {
        "messages": [{"role": "user", "content": "synthetic input"}],
        "metadata": {},
    }

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type=litellm.types.utils.CallTypes.call_mcp_tool.value,
        )

    detail = exc_info.value.detail
    logged_response = request_data["metadata"]["_model_armor_response"]
    if sanitize:
        assert detail == {"error": "Content blocked by Model Armor"}
        assert logged_response == {
            "sanitizationResult": {
                "filterResults": {
                    "sdp": {
                        "sdpFilterResult": {
                            "inspectResult": {
                                "matchState": "MATCH_FOUND",
                                "findings": "[REDACTED]",
                            }
                        }
                    }
                }
            }
        }
        assert marker not in str(detail)
        assert marker not in str(logged_response)
    else:
        assert detail["model_armor_response"] == armor_response
        assert logged_response == armor_response
        assert marker in str(detail)
        assert marker in str(logged_response)


def test_model_armor_sanitize_error_detail_config_wiring():
    from litellm.proxy.guardrails.guardrail_hooks.model_armor import (
        initialize_guardrail,
    )
    from litellm.types.guardrails import LitellmParams

    config = {"guardrail_name": "model-armor-test"}
    params = {
        "guardrail": "model_armor",
        "mode": "pre_mcp_call",
        "template_id": "test-template",
        "project_id": "test-project",
    }
    opted_out = initialize_guardrail(
        LitellmParams(**params, sanitize_error_detail=False), config
    )
    explicit_null = initialize_guardrail(
        LitellmParams(**params, sanitize_error_detail=None), config
    )
    default = initialize_guardrail(LitellmParams(**params), config)

    assert opted_out.sanitize_error_detail is False
    assert explicit_null.sanitize_error_detail is True
    assert default.sanitize_error_detail is True


def test_model_armor_ui_friendly_name():
    """Test the UI-friendly name of the Model Armor guardrail"""
    from litellm.types.proxy.guardrails.guardrail_hooks.model_armor import (
        ModelArmorGuardrailConfigModel,
    )

    assert (
        ModelArmorGuardrailConfigModel.ui_friendly_name() == "Google Cloud Model Armor"
    )


@pytest.mark.asyncio
async def test_model_armor_no_messages():
    """Test Model Armor when request has no messages"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    request_data = {"model": "gpt-4", "metadata": {"guardrails": ["model-armor-test"]}}

    # Should return data unchanged when no messages
    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=request_data,
        call_type="completion",
    )

    assert result == request_data


@pytest.mark.asyncio
async def test_model_armor_empty_message_content():
    """Test Model Armor when message content is empty"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    request_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Previous response"},
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    # Should return data unchanged when no content
    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=request_data,
        call_type="completion",
    )

    assert result == request_data


@pytest.mark.asyncio
async def test_model_armor_system_assistant_messages():
    """Test Model Armor with only system/assistant messages (no user messages)"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    request_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "assistant", "content": "How can I help you?"},
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    # Should return data unchanged when no user messages
    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=request_data,
        call_type="completion",
    )

    assert result == request_data


@pytest.mark.asyncio
async def test_model_armor_fail_on_error_false():
    """Test Model Armor with fail_on_error=False when API fails"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        fail_on_error=False,
    )

    # Mock the async handler to raise an exception
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    # Make it raise a non-HTTP exception to test the fail_on_error logic
    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(side_effect=Exception("Connection error")),
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should not raise exception when fail_on_error=False
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Should return original data
        assert result == request_data


@pytest.mark.asyncio
async def test_model_armor_custom_api_endpoint():
    """Test Model Armor with custom API endpoint"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    custom_endpoint = "https://custom-modelarmor.example.com"
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        api_endpoint=custom_endpoint,
    )

    # Mock successful response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={"action": "NONE"})

    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ) as mock_post:
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test message"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Verify custom endpoint was used
        call_args = mock_post.call_args
        assert call_args[1]["url"].startswith(custom_endpoint)


@pytest.mark.asyncio
async def test_model_armor_dict_credentials():
    """Test Model Armor with dictionary credentials instead of file path"""
    try:
        from google.auth import default
    except ImportError:
        pytest.skip("google.auth not installed")
        return

    # Use patch context manager properly
    mock_creds_obj = Mock()
    mock_creds_obj.token = "test-token"
    mock_creds_obj.expired = False
    mock_creds_obj.project_id = "test-project"

    with patch.object(
        ModelArmorGuardrail,
        "_credentials_from_service_account",
        return_value=mock_creds_obj,
    ) as mock_creds:
        creds_dict = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key": "test-key",
            "client_email": "test@example.com",
        }

        guardrail = ModelArmorGuardrail(
            template_id="test-template",
            credentials=creds_dict,
            location="us-central1",
        )

        # Force credential loading
        creds, project_id = guardrail.load_auth(credentials=creds_dict, project_id=None)

        assert mock_creds.called
        assert project_id == "test-project"


@pytest.mark.asyncio
async def test_model_armor_action_none():
    """Test Model Armor when action is NONE (no sanitization needed)"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_request_content=True,
    )

    # Mock response with action=NO_MATCH_FOUND
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={"sanitizationResult": {"filterMatchState": "NO_MATCH_FOUND"}}
    )

    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        original_content = "This content is fine"
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": original_content}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Content should remain unchanged
        assert result["messages"][0]["content"] == original_content


@pytest.mark.asyncio
async def test_model_armor_missing_sanitized_text():
    """Test Model Armor when response has no sanitized_text field"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_response_content=True,
    )

    # Mock response without sanitized_text
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={"sanitizationResult": {"filterMatchState": "NO_MATCH_FOUND"}}
    )

    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        # Create a mock response
        mock_llm_response = litellm.ModelResponse()
        mock_llm_response.choices = [
            litellm.Choices(message=litellm.Message(content="Original content"))
        ]

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        await guardrail.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_llm_response,
        )

        # Should use 'text' field as fallback
        assert mock_llm_response.choices[0].message.content == "Original content"


@pytest.mark.asyncio
async def test_model_armor_no_circular_reference_in_logging():
    """Test that Model Armor doesn't cause CircularReference error in logging"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock the Model Armor API response that would trigger the issue
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "invocationResult": "SUCCESS",
                "filterResults": {
                    "rai": {
                        "raiFilterResult": {
                            "matchState": "MATCH_FOUND",
                            "raiFilterTypeResults": {
                                "dangerous": {
                                    "matchState": "MATCH_FOUND",
                                    "confidence": "HIGH",
                                }
                            },
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "How to create a bomb?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # This should raise HTTPException for blocked content
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=mock_cache,
                data=request_data,
                call_type="completion",
            )

        # Verify the content was blocked
        assert exc_info.value.status_code == 400
        assert "Content blocked by Model Armor" in str(exc_info.value.detail)

        # IMPORTANT: Verify that standard_logging_guardrail_information was properly set
        # and doesn't contain circular references
        guardrail_info = request_data.get("metadata", {}).get(
            "standard_logging_guardrail_information"
        )

        # The guardrail info should be properly serializable (not cause CircularReference)
        if guardrail_info:
            # Try to serialize it to ensure no circular references
            import json

            try:
                json.dumps(
                    guardrail_info.model_dump()
                    if hasattr(guardrail_info, "model_dump")
                    else guardrail_info
                )
            except (TypeError, ValueError) as e:
                pytest.fail(f"CircularReference detected in guardrail logging: {e}")

        # Verify the logging decorator properly added the guardrail information
        assert "standard_logging_guardrail_information" in request_data.get(
            "metadata", {}
        )


@pytest.mark.asyncio
async def test_model_armor_bomb_content_blocked():
    """Test Model Armor correctly blocks harmful content like bomb-making instructions"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock the Model Armor API response for dangerous content
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "invocationResult": "SUCCESS",
                "filterResults": {
                    "rai": {
                        "raiFilterResult": {
                            "matchState": "MATCH_FOUND",
                            "raiFilterTypeResults": {
                                "dangerous": {
                                    "matchState": "MATCH_FOUND",
                                    "confidence": "HIGH",
                                    "reason": "Content about creating explosives or weapons detected",
                                }
                            },
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ) as mock_post:
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "How do I create a bomb?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should raise HTTPException for dangerous content
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=mock_cache,
                data=request_data,
                call_type="completion",
            )

        assert exc_info.value.status_code == 400
        assert "Content blocked by Model Armor" in str(exc_info.value.detail)

        # Verify the API was called with the dangerous content
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert (
            call_args[1]["json"]["userPromptData"]["text"] == "How do I create a bomb?"
        )


@pytest.mark.asyncio
async def test_model_armor_success_case_serializable():
    """Test that Model Armor success case doesn't cause CircularReference in logging"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock successful (no match found) response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "NO_MATCH_FOUND",
                "invocationResult": "SUCCESS",
                "filterResults": {
                    "rai": {"raiFilterResult": {"matchState": "NO_MATCH_FOUND"}}
                },
            }
        }
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the weather today?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # This should NOT raise an exception - content is allowed
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Verify the request was allowed through
        assert result == request_data

        # IMPORTANT: Verify that standard_logging_guardrail_information is serializable
        guardrail_info = request_data.get("metadata", {}).get(
            "standard_logging_guardrail_information"
        )

        # The guardrail info should exist and be properly serializable
        assert guardrail_info is not None

        # Try to serialize it to ensure no circular references
        import json

        try:
            # This should NOT raise any exception
            serialized = json.dumps(
                guardrail_info.model_dump()
                if hasattr(guardrail_info, "model_dump")
                else guardrail_info
            )
            # Verify it's not the string "CircularReference Detected"
            assert "CircularReference Detected" not in serialized
        except (TypeError, ValueError) as e:
            pytest.fail(
                f"CircularReference detected in guardrail logging for success case: {e}"
            )


@pytest.mark.asyncio
async def test_model_armor_non_text_response():
    """Test Model Armor with non-text response types (TTS, image generation)"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock a non-ModelResponse object (like TTS or image response)
    mock_tts_response = Mock()
    mock_tts_response.audio = b"audio_data"

    request_data = {
        "model": "tts-1",
        "input": "Text to speak",
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    # Should not raise an error for non-text responses
    await guardrail.async_post_call_success_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        response=mock_tts_response,
    )


@pytest.mark.asyncio
async def test_model_armor_token_refresh():
    """Test Model Armor handling expired auth tokens"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock successful response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={"action": "NONE"})

    # Mock token refresh - first call returns expired token, second returns fresh
    call_count = 0

    async def mock_token_method(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return (f"token-{call_count}", "test-project")

    guardrail._ensure_access_token_async = AsyncMock(side_effect=mock_token_method)
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Verify token method was called
        assert guardrail._ensure_access_token_async.called


@pytest.mark.asyncio
async def test_model_armor_non_model_response():
    """Test Model Armor handles non-ModelResponse types (e.g., TTS) correctly"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock a TTS response (not a ModelResponse)
    class TTSResponse:
        def __init__(self):
            self.audio_data = b"fake audio data"

    tts_response = TTSResponse()

    # Mock the access token
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    guardrail.async_handler = AsyncMock()

    # Call post-call hook with non-ModelResponse
    await guardrail.async_post_call_success_hook(
        data={
            "model": "tts-1",
            "input": "Hello world",
            "metadata": {"guardrails": ["model-armor-test"]},
        },
        user_api_key_dict=mock_user_api_key_dict,
        response=tts_response,
    )

    # Verify that Model Armor API was NOT called since there's no text content
    assert not guardrail.async_handler.post.called


@pytest.mark.asyncio
async def test_model_armor_guardrail_status_intervened_vs_failed():
    """
    regression test for bug where _process_error always set 'guardrail_failed_to_respond'
    even for intentional blocks (error 400).
    """
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    # 1: Blocked content should raise exception and show guardrail status: guardrail_intervened"
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "rai": {
                        "raiFilterResult": {
                            "matchState": "MATCH_FOUND",
                        }
                    }
                },
            }
        }
    )

    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "bad content"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }
        with pytest.raises(HTTPException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=mock_user_api_key_dict,
                cache=mock_cache,
                data=request_data,
                call_type="completion",
            )

        info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert info[0]["guardrail_name"] == guardrail.guardrail_name
        assert info[0]["guardrail_status"] == "guardrail_intervened"
        assert "model_armor_response" not in info[0]["guardrail_response"]
        assert "sanitizationResult" not in info[0]["guardrail_response"]

    # 2: if an API error - guardrail status should be guardrail_failed_to_respond"
    guardrail2 = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test2",
        fail_on_error=True,
    )

    guardrail2._ensure_access_token_async = AsyncMock(
        side_effect=ConnectionError("timeout")
    )
    request_data2 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"guardrails": ["model-armor-test2"]},
    }
    with pytest.raises(ConnectionError):
        await guardrail2.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data2,
            call_type="completion",
        )

    info2 = request_data2["metadata"]["standard_logging_guardrail_information"]
    assert info2[0]["guardrail_status"] == "guardrail_failed_to_respond"


def mock_open(read_data=""):
    """Helper to create a mock file object"""
    import io
    from unittest.mock import MagicMock

    file_object = io.StringIO(read_data)
    file_object.__enter__ = lambda self: self
    file_object.__exit__ = lambda self, *args: None

    mock_file = MagicMock(return_value=file_object)
    return mock_file


def test_model_armor_initialization_preserves_project_id():
    """Test that ModelArmorGuardrail initialization preserves the project_id correctly"""
    # This tests the fix for issue #12757 where project_id was being overwritten to None
    # due to incorrect initialization order with VertexBase parent class

    test_project_id = "cloud-xxxxx-yyyyy"
    test_template_id = "global-armor"
    test_location = "eu"

    guardrail = ModelArmorGuardrail(
        template_id=test_template_id,
        project_id=test_project_id,
        location=test_location,
        guardrail_name="model-armor-test",
    )

    # Assert that project_id is preserved after initialization
    assert guardrail.project_id == test_project_id
    assert guardrail.template_id == test_template_id
    assert guardrail.location == test_location

    # Also check that the VertexBase initialization didn't reset project_id to None
    assert hasattr(guardrail, "project_id")
    assert guardrail.project_id is not None


@pytest.mark.asyncio
async def test_model_armor_with_default_credentials():
    """Test Model Armor with default credentials and explicit project_id"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    # Initialize with explicit project_id but no credentials (simulating default auth)
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="cloud-test-project",
        location="eu",
        guardrail_name="model-armor-test",
        credentials=None,  # Explicitly set to None to test default auth
    )

    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={"sanitized_text": "Test content", "action": "SANITIZE"}
    )

    # Mock the access token method to simulate successful auth
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "cloud-test-project")
    )

    # Mock the async handler
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ) as mock_post:
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test content"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # This should not raise ValueError about project_id
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion",
        )

        # Verify the project_id was used correctly in the API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "cloud-test-project" in call_args[1]["url"]


# ===== ASYNC MODERATION HOOK TESTS =====


@pytest.mark.asyncio
async def test_async_moderation_hook_success_no_blocking():
    """Test async_moderation_hook with successful response (no blocking)"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock successful (no match found) response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "NO_MATCH_FOUND",
                "filterResults": {
                    "rai": {"raiFilterResult": {"matchState": "NO_MATCH_FOUND"}}
                },
            }
        }
    )

    # Mock the access token method and async handler
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        result = await guardrail.async_moderation_hook(
            data=request_data,
            user_api_key_dict=mock_user_api_key_dict,
            call_type="completion",
        )

        # Should return the original data unchanged
        assert result == request_data
        # Should have metadata added
        assert "_model_armor_response" in request_data["metadata"]
        assert request_data["metadata"]["_model_armor_status"] == "success"


@pytest.mark.asyncio
async def test_async_moderation_hook_content_blocked():
    """Test async_moderation_hook when content should be blocked"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    # Mock response that indicates content should be blocked
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "rai": {"raiFilterResult": {"matchState": "MATCH_FOUND"}}
                },
            }
        }
    )

    # Mock the access token method and async handler
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Some harmful content"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should raise HTTPException for blocked content
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_moderation_hook(
                data=request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion",
            )

        assert exc_info.value.status_code == 400
        assert "Content blocked by Model Armor" in str(exc_info.value.detail)
        # Should have metadata added even when blocked
        assert "_model_armor_response" in request_data["metadata"]
        assert request_data["metadata"]["_model_armor_status"] == "blocked"

        # IMPORTANT: Verify that applied_guardrails is populated even when blocked
        # This is a regression test for the issue where applied_guardrails was null when blocked
        assert "applied_guardrails" in request_data["metadata"]
        assert "model-armor-test" in request_data["metadata"]["applied_guardrails"]


@pytest.mark.asyncio
async def test_async_moderation_hook_with_sanitization():
    """Test async_moderation_hook with content sanitization enabled"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_request_content=True,  # Enable sanitization
    )

    # Mock response with sanitized content
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "sdp": {
                        "sdpFilterResult": {
                            "deidentifyResult": {
                                "matchState": "MATCH_FOUND",
                                "data": {
                                    "text": "Hello, my phone number is [REDACTED]"
                                },
                            }
                        }
                    }
                },
            }
        }
    )

    # Mock the access token method and async handler
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(return_value=mock_response)
    ):
        original_content = "Hello, my phone number is 555-123-4567"
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": original_content}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        result = await guardrail.async_moderation_hook(
            data=request_data,
            user_api_key_dict=mock_user_api_key_dict,
            call_type="completion",
        )

        # Should return data with sanitized content
        assert result == request_data
        # Content should be sanitized
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        sanitized_content = get_last_user_message(request_data["messages"])
        assert sanitized_content == "Hello, my phone number is [REDACTED]"
        assert sanitized_content != original_content
        # Should have metadata added
        assert "_model_armor_response" in request_data["metadata"]
        assert request_data["metadata"]["_model_armor_status"] == "success"


@pytest.mark.asyncio
async def test_async_moderation_hook_no_user_messages():
    """Test async_moderation_hook when there are no user messages to check"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )

    request_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "assistant", "content": "How can I help you?"},
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    result = await guardrail.async_moderation_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        call_type="completion",
    )

    # Should return the original data unchanged since no user messages to check
    assert result == request_data


@pytest.mark.asyncio
async def test_async_moderation_hook_should_not_run():
    """Test async_moderation_hook when guardrail should not run due to missing guardrail name"""
    try:
        import google.auth
    except ImportError:
        pytest.skip("google.auth not installed")
        return

    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="different-guardrail-name",  # Different name than what's in metadata
    )

    # Request data with a different guardrail name
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "metadata": {
            "guardrails": ["some-other-guardrail"]
        },  # Different guardrail name
    }

    result = await guardrail.async_moderation_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        call_type="completion",
    )

    # Should return the original data unchanged since guardrail name doesn't match
    assert result == request_data


@pytest.mark.asyncio
async def test_async_moderation_hook_api_error_fail_on_error_true():
    """Test async_moderation_hook when API call fails and fail_on_error is True"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        optional_params={"fail_on_error": True},
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler to raise an exception
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(side_effect=Exception("API Error"))
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Should raise the exception since fail_on_error is True
        with pytest.raises(Exception) as exc_info:
            await guardrail.async_moderation_hook(
                data=request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion",
            )

        assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_moderation_hook_api_error_fail_on_error_false():
    """Test async_moderation_hook when API call fails and fail_on_error is False"""
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        optional_params={"fail_on_error": False},
    )

    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(
        return_value=("test-token", "test-project")
    )

    # Mock the async handler to raise an exception
    with patch.object(
        guardrail.async_handler, "post", AsyncMock(side_effect=Exception("API Error"))
    ):
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "metadata": {"guardrails": ["model-armor-test"]},
        }

        # Even with fail_on_error=False, the decorator may still raise the exception
        # This test verifies that the exception is properly logged and handled
        with pytest.raises(Exception) as exc_info:
            await guardrail.async_moderation_hook(
                data=request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion",
            )

        assert "API Error" in str(exc_info.value)


# ===== FILE / DOCUMENT ATTACHMENT SCANNING TESTS (LIT-4084) =====

PDF_BYTES = b"%PDF-1.4\nfake pdf payload with policy-violating content\n%%EOF"
DOCX_BYTES = b"PK\x03\x04 fake docx zip payload"


def _make_guardrail(**overrides):
    params = dict(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )
    params.update(overrides)
    guardrail = ModelArmorGuardrail(**params)
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    return guardrail


def _armor_response(blocked: bool):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    if blocked:
        body = {
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {"rai": {"raiFilterResult": {"matchState": "MATCH_FOUND"}}},
            }
        }
    else:
        body = {"sanitizationResult": {"filterMatchState": "NO_MATCH_FOUND"}}
    mock_response.json = AsyncMock(return_value=body)
    return mock_response


def _byte_items_sent(mock_post):
    """Return every byteItem payload (file scans) submitted to Model Armor."""
    items = []
    for call in mock_post.call_args_list:
        body = call.kwargs.get("json", {})
        byte_item = body.get("userPromptData", {}).get("byteItem")
        if byte_item is not None:
            items.append(byte_item)
    return items


def _text_payloads_sent(mock_post):
    """Return every text payload (text scans) submitted to Model Armor."""
    texts = []
    for call in mock_post.call_args_list:
        body = call.kwargs.get("json", {})
        user_prompt = body.get("userPromptData", {})
        if "text" in user_prompt:
            texts.append(user_prompt["text"])
    return texts


def _file_message(file_data_b64: str, mime: str = "application/pdf", filename: str = "doc.pdf"):
    return {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "file_data": f"data:{mime};base64,{file_data_b64}",
                    "filename": filename,
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_pre_call_blocks_harmful_pdf_attachment():
    """Pre-call hook must scan an inline PDF attachment and block on a Model Armor match.

    Regression for LIT-4084: before the fix, a file-only message has no extractable
    text, so the hook returned early and the document was never sent to Model Armor.
    """
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=True)),
    ) as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "Content blocked by Model Armor" in str(exc_info.value.detail)

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "PDF"
    assert base64.b64decode(byte_items[0]["byteData"]) == PDF_BYTES

    assert "model-armor-test" in request_data["metadata"]["applied_guardrails"]
    assert request_data["metadata"]["_model_armor_status"] == "blocked"


@pytest.mark.asyncio
async def test_pre_call_allows_safe_pdf_attachment_but_still_scans_it():
    """A safe PDF attachment passes through, but the bytes are still submitted to Model Armor."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert result == request_data
    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "PDF"
    assert base64.b64decode(byte_items[0]["byteData"]) == PDF_BYTES


@pytest.mark.asyncio
async def test_moderation_hook_blocks_harmful_file_attachment():
    """The during-call moderation hook must scan file attachments the same way."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=True)),
    ) as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_moderation_hook(
                data=request_data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "Content blocked by Model Armor" in str(exc_info.value.detail)
    assert len(_byte_items_sent(mock_post)) == 1
    assert "model-armor-test" in request_data["metadata"]["applied_guardrails"]


@pytest.mark.asyncio
async def test_pre_call_scans_both_text_and_file():
    """When a message has both text and a file, both are submitted to Model Armor."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this"},
                    {
                        "type": "file",
                        "file": {"file_data": f"data:application/pdf;base64,{pdf_b64}"},
                    },
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert len(_byte_items_sent(mock_post)) == 1
    assert "summarize this" in _text_payloads_sent(mock_post)


@pytest.mark.asyncio
async def test_pre_call_scans_anthropic_document_block():
    """Anthropic-style `type: document` blocks with inline base64 are scanned and typed."""
    guardrail = _make_guardrail()
    docx_b64 = base64.b64encode(DOCX_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "data": docx_b64,
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "WORD_DOCUMENT"
    assert base64.b64decode(byte_items[0]["byteData"]) == DOCX_BYTES


@pytest.mark.asyncio
async def test_pre_call_blocks_unresolvable_file_id_reference():
    """A bare file_id has no inline bytes to scan, so by default the guardrail fails closed."""
    guardrail = _make_guardrail()
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this"},
                    {"type": "file", "file": {"file_id": "file-abc123"}},
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "could not scan" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_pre_call_blocks_remote_url_document_reference():
    """A remote (https) document reference cannot be fetched here, so it fails closed by default."""
    guardrail = _make_guardrail()
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_data": "https://example.com/secret.pdf", "filename": "secret.pdf"},
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_pre_call_file_id_reference_skipped_when_fail_open():
    """With fail_on_error=False an unresolvable reference is skipped and the text is still scanned."""
    guardrail = _make_guardrail(fail_on_error=False)
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this"},
                    {"type": "file", "file": {"file_id": "file-abc123"}},
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert _byte_items_sent(mock_post) == []
    assert _text_payloads_sent(mock_post) == ["summarize this"]


@pytest.mark.asyncio
async def test_pre_call_file_id_reference_passthrough_when_skip_unscannable_enabled():
    """skip_unscannable_attachments lets a file_id reference through even with fail_on_error=True."""
    guardrail = _make_guardrail(skip_unscannable_attachments=True)
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this"},
                    {"type": "file", "file": {"file_id": "file-abc123"}},
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert _byte_items_sent(mock_post) == []
    assert _text_payloads_sent(mock_post) == ["summarize this"]


@pytest.mark.asyncio
async def test_pre_call_gs_uri_reference_passthrough_when_skip_unscannable_enabled():
    """A gs:// document reference passes through when skip_unscannable_attachments is enabled."""
    guardrail = _make_guardrail(skip_unscannable_attachments=True)
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_data": "gs://my-bucket/report.pdf", "filename": "report.pdf"},
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert _byte_items_sent(mock_post) == []


def test_initialize_guardrail_forwards_skip_unscannable_attachments():
    """skip_unscannable_attachments configured in litellm_params reaches the guardrail instance."""
    from litellm.proxy.guardrails.guardrail_hooks.model_armor import initialize_guardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

    litellm_params = LitellmParams(
        guardrail="model_armor",
        mode="pre_call",
        template_id="demo-template",
        project_id="demo-project",
        skip_unscannable_attachments=True,
    )
    guardrail = initialize_guardrail(
        litellm_params=litellm_params,
        guardrail=Guardrail(guardrail_name="model-armor-config-test"),
    )

    assert guardrail.optional_params.get("skip_unscannable_attachments") is True


def test_initialize_guardrail_skip_unscannable_defaults_false():
    """A config that omits skip_unscannable_attachments keeps the secure default (block)."""
    from litellm.proxy.guardrails.guardrail_hooks.model_armor import initialize_guardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

    litellm_params = LitellmParams(
        guardrail="model_armor",
        mode="pre_call",
        template_id="demo-template",
        project_id="demo-project",
    )
    guardrail = initialize_guardrail(
        litellm_params=litellm_params,
        guardrail=Guardrail(guardrail_name="model-armor-config-default"),
    )

    assert guardrail.optional_params.get("skip_unscannable_attachments") is False


@pytest.mark.asyncio
async def test_skip_unscannable_still_fails_closed_on_api_error():
    """skip_unscannable_attachments only affects references; a real API error still fails closed."""
    guardrail = _make_guardrail(skip_unscannable_attachments=True, fail_on_error=True)
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(side_effect=Exception("model armor upstream 500")),
    ):
        with pytest.raises(Exception) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert "model armor upstream 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pre_call_scans_every_attachment_without_a_count_cap():
    """There is no per-request attachment cap: every scannable attachment is submitted to Model Armor."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    block = {
        "type": "file",
        "file": {"file_data": f"data:application/pdf;base64,{pdf_b64}"},
    }
    count = 25
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": [block] * count}],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    mock_post = AsyncMock(return_value=_armor_response(blocked=False))
    with patch.object(guardrail.async_handler, "post", mock_post):
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert len(_byte_items_sent(mock_post)) == count


@pytest.mark.asyncio
async def test_file_scan_error_isolated_when_fail_open():
    """A transient error on one attachment does not skip the remaining attachments (fail_on_error=False)."""
    guardrail = _make_guardrail(fail_on_error=False)
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    block = {
        "type": "file",
        "file": {"file_data": f"data:application/pdf;base64,{pdf_b64}"},
    }
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": [block, block]}],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    # First attachment raises a transient error, second returns a normal response
    post = AsyncMock(side_effect=[Exception("transient"), _armor_response(blocked=False)])
    with patch.object(guardrail.async_handler, "post", post):
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    # Both attachments are attempted: the first errors and is isolated, the second still scans
    assert post.call_count == 2


@pytest.mark.asyncio
async def test_pre_call_skips_unsupported_file_type():
    """An image attachment (no Model Armor byteDataType) is not submitted as a document."""
    guardrail = _make_guardrail()
    png_b64 = base64.b64encode(b"\x89PNG fake").decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(png_b64, mime="image/png", filename="x.png")],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_blocks_file_over_size_limit():
    """A recognized document over Model Armor's 4 MB limit cannot be scanned, so it is blocked."""
    from litellm.proxy.guardrails.guardrail_hooks.model_armor.file_scanning import (
        MODEL_ARMOR_MAX_FILE_SIZE_BYTES,
    )

    guardrail = _make_guardrail()
    oversize_b64 = base64.b64encode(b"x" * (MODEL_ARMOR_MAX_FILE_SIZE_BYTES + 1)).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(oversize_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "scan limit" in str(exc_info.value.detail)
    # The oversized document is never forwarded to the Model Armor API
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_oversize_file_skipped_when_fail_open():
    """With fail_on_error=False the operator opts into fail-open, so an oversized file proceeds."""
    from litellm.proxy.guardrails.guardrail_hooks.model_armor.file_scanning import (
        MODEL_ARMOR_MAX_FILE_SIZE_BYTES,
    )

    guardrail = _make_guardrail(fail_on_error=False)
    oversize_b64 = base64.b64encode(b"x" * (MODEL_ARMOR_MAX_FILE_SIZE_BYTES + 1)).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(oversize_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert result == request_data
    mock_post.assert_not_called()


def _armor_sdp_deidentify_response():
    """A response that only trips the SDP deidentify (PII masking) filter."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(
        return_value={
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND",
                "filterResults": {
                    "sdp": {
                        "sdpFilterResult": {
                            "deidentifyResult": {
                                "matchState": "MATCH_FOUND",
                                "data": {"text": "[REDACTED]"},
                            }
                        }
                    }
                },
            }
        }
    )
    return mock_response


@pytest.mark.asyncio
async def test_pre_call_blocks_pii_document_even_when_masking_enabled():
    """A PII document must block, not pass, even when mask_request_content=True.

    Documents have no masking fallback (Model Armor returns findings, not a sanitized
    file), so a deidentify-only match has to block. Without this the original bytes
    would reach the provider with PII intact.
    """
    guardrail = _make_guardrail(mask_request_content=True)
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_sdp_deidentify_response()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert request_data["metadata"]["_model_armor_status"] == "blocked"


@pytest.mark.asyncio
async def test_pre_call_scans_raw_base64_file_without_data_uri():
    """A `type: file` with raw base64 (no data: URI) resolves its MIME from the filename."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_data": pdf_b64, "filename": "report.pdf"},
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "PDF"
    assert base64.b64decode(byte_items[0]["byteData"]) == PDF_BYTES


@pytest.mark.asyncio
async def test_first_of_multiple_attachments_blocks():
    """Scanning stops and blocks at the first flagged attachment."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    file_block = {
        "type": "file",
        "file": {"file_data": f"data:application/pdf;base64,{pdf_b64}"},
    }
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": [file_block, file_block]}],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=True)),
    ) as mock_post:
        with pytest.raises(HTTPException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_file_scan_fail_on_error_false_proceeds():
    """When the Model Armor call errors and fail_on_error=False, the request proceeds."""
    guardrail = _make_guardrail(fail_on_error=False)
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(side_effect=Exception("Connection error")),
    ):
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert result == request_data


SUPPORTED_MIME_TYPE_MATRIX = [
    ("application/pdf", "PDF"),
    ("application/msword", "WORD_DOCUMENT"),
    (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "WORD_DOCUMENT",
    ),
    ("application/vnd.ms-excel", "EXCEL_DOCUMENT"),
    (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "EXCEL_DOCUMENT",
    ),
    ("application/vnd.ms-powerpoint", "POWERPOINT_DOCUMENT"),
    (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "POWERPOINT_DOCUMENT",
    ),
    ("text/csv", "CSV"),
    ("text/plain", "TXT"),
]


@pytest.mark.parametrize("mime,expected_byte_data_type", SUPPORTED_MIME_TYPE_MATRIX)
@pytest.mark.asyncio
async def test_pre_call_submits_correct_byte_data_type_for_every_supported_mime(mime, expected_byte_data_type):
    """Every supported MIME type maps to the right Model Armor byteDataType and is submitted."""
    guardrail = _make_guardrail()
    payload = b"file content for %s" % mime.encode()
    payload_b64 = base64.b64encode(payload).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(payload_b64, mime=mime, filename="attachment")],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == expected_byte_data_type
    assert base64.b64decode(byte_items[0]["byteData"]) == payload


@pytest.mark.asyncio
async def test_pre_call_resolves_mime_from_filename_when_data_uri_is_generic():
    """A data URI with a generic MIME still scans when the filename identifies a document.

    Regression for the case where attachments were skipped because only the data URI
    header MIME was consulted, ignoring file.format and the filename.
    """
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:application/octet-stream;base64,{pdf_b64}",
                            "filename": "report.pdf",
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "PDF"
    assert base64.b64decode(byte_items[0]["byteData"]) == PDF_BYTES


@pytest.mark.asyncio
async def test_pre_call_normalizes_mime_with_charset_suffix():
    """A MIME with a charset parameter (text/plain; charset=utf-8) still maps to TXT."""
    guardrail = _make_guardrail()
    payload = b"plain text body"
    payload_b64 = base64.b64encode(payload).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:text/plain;charset=utf-8;base64,{payload_b64}",
                            "filename": "notes.txt",
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "TXT"


@pytest.mark.asyncio
async def test_pre_call_scans_macro_enabled_office_document():
    """Macro-enabled and template Office MIME types map to their document family, not skipped."""
    guardrail = _make_guardrail()
    payload = b"macro enabled word document bytes"
    payload_b64 = base64.b64encode(payload).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:application/vnd.ms-word.document.macroEnabled.12;base64,{payload_b64}",
                            "filename": "report.docm",
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "WORD_DOCUMENT"


@pytest.mark.asyncio
async def test_file_scan_does_not_log_document_bytes():
    """Debug logging must never emit the scanned document's base64 bytes."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    logged_args = []

    def _capture(*args, **kwargs):
        logged_args.append(args)

    with patch.object(verbose_proxy_logger, "debug", side_effect=_capture):
        with patch.object(
            guardrail.async_handler,
            "post",
            AsyncMock(return_value=_armor_response(blocked=False)),
        ):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    flattened = " ".join(str(arg) for call in logged_args for arg in call)
    assert pdf_b64 not in flattened
    # the file request is still logged, just with type and size instead of the bytes
    assert "byteDataType" in flattened


@pytest.mark.asyncio
async def test_pre_call_prefers_filename_over_conflicting_data_uri_mime():
    """A data URI mislabeled text/plain must not downgrade a .pdf attachment to TXT scanning."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:text/plain;base64,{pdf_b64}",
                            "filename": "report.pdf",
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    byte_items = _byte_items_sent(mock_post)
    assert len(byte_items) == 1
    assert byte_items[0]["byteDataType"] == "PDF"


@pytest.mark.asyncio
async def test_file_and_text_responses_are_both_recorded():
    """A request with both a file and text records both Model Armor responses, not just the last."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this"},
                    {
                        "type": "file",
                        "file": {"file_data": f"data:application/pdf;base64,{pdf_b64}"},
                    },
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ):
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    recorded = request_data["metadata"]["_model_armor_response"]
    # A list (not a tuple) so the guardrail logging redaction/serialization can walk it
    assert isinstance(recorded, list)
    assert len(recorded) == 2


@pytest.mark.asyncio
async def test_single_scan_response_stays_a_dict():
    """A single scan keeps the backward-compatible single-dict response shape."""
    guardrail = _make_guardrail()
    pdf_b64 = base64.b64encode(PDF_BYTES).decode("utf-8")
    request_data = {
        "model": "gpt-4",
        "messages": [_file_message(pdf_b64)],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ):
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert isinstance(request_data["metadata"]["_model_armor_response"], dict)


@pytest.mark.asyncio
async def test_pre_call_blocks_supported_document_with_undecodable_base64():
    """A supported document whose inline base64 will not decode cannot be scanned, so it fails closed."""
    guardrail = _make_guardrail()
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,@@@not-valid-base64@@@",
                            "filename": "broken.pdf",
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=request_data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "could not scan" in str(exc_info.value.detail)
    # The malformed document is never submitted to Model Armor
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_undecodable_document_skipped_when_fail_open():
    """With fail_on_error=False a malformed supported document is skipped rather than blocking."""
    guardrail = _make_guardrail(fail_on_error=False)
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,@@@not-valid-base64@@@",
                            "filename": "broken.pdf",
                        },
                    }
                ],
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=request_data,
            call_type="completion",
        )

    assert result == request_data
    mock_post.assert_not_called()


def test_accumulated_responses_are_redactable_as_a_list():
    """Accumulated file+text responses must be a list so guardrail logging can redact nested keys.

    Regression: a tuple is skipped by redact_nested_match_and_regex_keys (it only recurses into
    dicts and lists), which would leave sensitive match/regex findings un-redacted in logs.
    """
    from litellm.litellm_core_utils.core_helpers import (
        redact_nested_match_and_regex_keys,
    )

    first = {"sanitizationResult": {"filterResults": {"f": {"match": "secret-one"}}}}
    second = {"sanitizationResult": {"filterResults": {"f": {"match": "secret-two"}}}}

    accumulated = ModelArmorGuardrail._append_armor_response(first, second)
    assert isinstance(accumulated, list)

    redacted = redact_nested_match_and_regex_keys(accumulated)
    blob = json.dumps(redacted)
    assert "secret-one" not in blob
    assert "secret-two" not in blob
    assert blob.count("[REDACTED]") == 2


def _mcp_synthetic_data(tool_name: str = "send_email", arguments: dict = None):
    """Mirror ProxyLogging._convert_mcp_to_llm_format: an MCP tool call rendered as a
    synthetic user message so the existing prompt-scanning path can inspect it."""
    if arguments is None:
        arguments = {"to": "user@example.com", "body": "some content"}
    return {
        "model": "mcp-tool-call",
        "messages": [
            {
                "role": "user",
                "content": f"Tool: {tool_name}\nArguments: {arguments}",
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]},
        "mcp_tool_name": tool_name,
        "mcp_arguments": arguments,
    }


@pytest.mark.asyncio
async def test_pre_call_hook_scans_mcp_tool_call_when_configured_for_pre_mcp_call():
    """A guardrail configured with mode `pre_mcp_call` must scan MCP tool calls.

    Regression: async_pre_call_hook hardcoded its event-type gate to `pre_call`, so a
    `pre_mcp_call` guardrail's own inner should_run_guardrail check returned False for an
    MCP call (call_type=call_mcp_tool) and the scan was skipped entirely -- letting
    sensitive content in tool arguments through unscanned. The gate must remap
    call_mcp_tool -> pre_mcp_call.
    """
    guardrail = _make_guardrail(event_hook="pre_mcp_call", mask_request_content=True)
    data = _mcp_synthetic_data()

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=data,
            call_type="call_mcp_tool",
        )

    mock_post.assert_called()


@pytest.mark.asyncio
async def test_pre_call_hook_skips_chat_traffic_when_configured_for_pre_mcp_call():
    """A `pre_mcp_call` guardrail must NOT scan ordinary chat completions -- the remap is
    scoped to MCP calls, so a `completion` call_type still fails the gate and is skipped."""
    guardrail = _make_guardrail(event_hook="pre_mcp_call", mask_request_content=True)
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello there"}],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(spec=DualCache),
            data=data,
            call_type="completion",
        )

    assert result == data
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_moderation_hook_scans_mcp_tool_call_when_configured_for_during_mcp_call():
    """A guardrail configured with mode `during_mcp_call` must scan MCP tool calls.

    Regression: async_moderation_hook hardcoded its event-type gate to `during_call`, so a
    `during_mcp_call` guardrail skipped MCP calls (call_type=call_mcp_tool). The gate must
    remap call_mcp_tool -> during_mcp_call.
    """
    guardrail = _make_guardrail(event_hook="during_mcp_call", mask_request_content=True)
    data = _mcp_synthetic_data()

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="call_mcp_tool",
        )

    mock_post.assert_called()


@pytest.mark.asyncio
async def test_moderation_hook_skips_chat_traffic_when_configured_for_during_mcp_call():
    """A `during_mcp_call` guardrail must NOT scan ordinary chat completions."""
    guardrail = _make_guardrail(event_hook="during_mcp_call", mask_request_content=True)
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello there"}],
        "metadata": {"guardrails": ["model-armor-test"]},
    }

    with patch.object(
        guardrail.async_handler,
        "post",
        AsyncMock(return_value=_armor_response(blocked=False)),
    ) as mock_post:
        result = await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
        )

    assert result == data
    mock_post.assert_not_called()
