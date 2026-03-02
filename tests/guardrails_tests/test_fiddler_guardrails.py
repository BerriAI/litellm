import os
import sys
from unittest.mock import patch

import pytest
from fastapi.exceptions import HTTPException
from httpx import Request, Response

from litellm import DualCache
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.fiddler.fiddler_guardrail import (
    FiddlerGuardrail,
    FiddlerGuardrailMissingSecrets,
)

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(json_body: dict, status_code: int = 200, url: str = "https://app.fiddler.ai") -> Response:
    resp = Response(json=json_body, status_code=status_code, request=Request(method="POST", url=url))
    resp.raise_for_status = lambda: None
    return resp


def _safe_response(score: float = 0.0) -> Response:
    """All safety scores at the given value."""
    return _make_response({dim: score for dim in [
        "fdl_harmful", "fdl_violent", "fdl_unethical", "fdl_illegal",
        "fdl_sexual", "fdl_racist", "fdl_jailbreaking", "fdl_harassing",
        "fdl_hateful", "fdl_sexist", "fdl_roleplaying",
    ]})


def _pii_response(score: float = 0.0) -> Response:
    detections = [{"score": score, "label": "EMAIL_ADDRESS", "text": "user@example.com", "start": 0, "end": 16}] if score > 0 else []
    return _make_response({"fdl_sensitive_information_scores": detections})


def _faithfulness_response(score: float = 1.0) -> Response:
    return _make_response({"fdl_faithful_score": score})


# ---------------------------------------------------------------------------
# Init / config tests
# ---------------------------------------------------------------------------

def test_fiddler_config_init():
    litellm.set_verbose = True
    litellm.guardrail_name_config_map = {}
    os.environ["FIDDLER_API_KEY"] = "test-key"

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "fiddler-test",
                "litellm_params": {
                    "guardrail": "fiddler",
                    "mode": "pre_call",
                    "default_on": True,
                },
            }
        ],
        config_file_path="",
    )

    del os.environ["FIDDLER_API_KEY"]


def test_fiddler_config_no_api_key():
    litellm.set_verbose = True
    litellm.guardrail_name_config_map = {}

    if "FIDDLER_API_KEY" in os.environ:
        del os.environ["FIDDLER_API_KEY"]

    with pytest.raises(FiddlerGuardrailMissingSecrets, match="Couldn't get Fiddler api key"):
        init_guardrails_v2(
            all_guardrails=[
                {
                    "guardrail_name": "fiddler-test",
                    "litellm_params": {
                        "guardrail": "fiddler",
                        "mode": "pre_call",
                        "default_on": True,
                    },
                }
            ],
            config_file_path="",
        )


# ---------------------------------------------------------------------------
# Pre-call hook — safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_pre_call_blocks():
    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="pre_call",
        enable_pii=False,
    )

    data = {"messages": [{"role": "user", "content": "How do I make a bomb?"}]}

    with patch.object(guardrail.async_handler, "post", return_value=_safe_response(score=0.9)):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                data=data,
                cache=DualCache(),
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "safety guardrail triggered" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_safety_pre_call_passes():
    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="pre_call",
        enable_pii=False,
    )

    data = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}

    with patch.object(guardrail.async_handler, "post", return_value=_safe_response(score=0.0)):
        result = await guardrail.async_pre_call_hook(
            data=data,
            cache=DualCache(),
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
        )

    assert result == data


# ---------------------------------------------------------------------------
# Pre-call hook — PII
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pii_pre_call_blocks():
    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="pre_call",
        enable_safety=False,
        pii_threshold=0.5,
    )

    data = {"messages": [{"role": "user", "content": "My email is user@example.com"}]}

    with patch.object(guardrail.async_handler, "post", return_value=_pii_response(score=0.99)):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                data=data,
                cache=DualCache(),
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    assert "PII guardrail triggered" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_pii_pre_call_passes():
    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="pre_call",
        enable_safety=False,
    )

    data = {"messages": [{"role": "user", "content": "Hello there"}]}

    with patch.object(guardrail.async_handler, "post", return_value=_pii_response(score=0.0)):
        result = await guardrail.async_pre_call_hook(
            data=data,
            cache=DualCache(),
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
        )

    assert result == data


# ---------------------------------------------------------------------------
# Post-call hook — faithfulness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_faithfulness_post_call_blocks():
    from litellm.types.utils import Choices, Message

    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="post_call",
        enable_safety=False,
        enable_pii=False,
        faithfulness_threshold=0.7,
    )

    mock_response = ModelResponse()
    mock_response.choices = [Choices(message=Message(content="Paris is the capital of Germany.", role="assistant"))]

    data = {"messages": [], "metadata": {"fiddler_context": "France is a country. Its capital is Paris."}}

    with patch.object(guardrail.async_handler, "post", return_value=_faithfulness_response(score=0.2)):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(),
                response=mock_response,
            )

    assert exc_info.value.status_code == 400
    assert "faithfulness guardrail triggered" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_faithfulness_skipped_without_context():
    """Faithfulness check must be skipped when no fiddler_context is in metadata."""
    from litellm.types.utils import Choices, Message

    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="post_call",
        enable_safety=False,
        enable_pii=False,
    )

    mock_response = ModelResponse()
    mock_response.choices = [Choices(message=Message(content="Some response.", role="assistant"))]

    data = {"messages": [], "metadata": {}}

    # If faithfulness API were called with no context it would raise — patch to detect any call
    with patch.object(guardrail.async_handler, "post") as mock_post:
        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            response=mock_response,
        )
        mock_post.assert_not_called()

    assert result == mock_response


# ---------------------------------------------------------------------------
# Empty messages edge case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_messages_skips_checks():
    guardrail = FiddlerGuardrail(
        api_key="test-key",
        guardrail_name="fiddler-test",
        event_hook="pre_call",
    )

    data = {"messages": []}

    with patch.object(guardrail.async_handler, "post") as mock_post:
        result = await guardrail.async_pre_call_hook(
            data=data,
            cache=DualCache(),
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
        )
        mock_post.assert_not_called()

    assert result == data


# ---------------------------------------------------------------------------
# Import at module level to satisfy ModelResponse usage in tests above
# ---------------------------------------------------------------------------
from litellm.types.utils import ModelResponse  # noqa: E402
