"""
Tests for Vertex AI Gemma MaaS models that route through the partner-models
OpenAI-compatible path (https://aiplatform.googleapis.com/.../endpoints/openapi).

These tests verify that:
1. The correct global URL is constructed (https://aiplatform.googleapis.com)
2. get_vertex_region resolves to "global" when model_cost says so
3. acompletion() goes through the OpenAI-compatible handler and hits
   /endpoints/openapi/chat/completions
4. Function-calling payloads (tools + tool_choice) pass through unchanged
5. Vision/image_url payloads pass through unchanged
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.vertex_ai_partner_models.main import VertexAIPartnerModels
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai import VertexPartnerProvider

# ---------------------------------------------------------------------------
# Model-cost entry used by all tests that need the model to be known
# ---------------------------------------------------------------------------

_GEMMA_MODEL_COST_ENTRY = {
    "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
        "litellm_provider": "vertex_ai-openai_models",
        "max_input_tokens": 256000,
        "max_output_tokens": 128000,
        "max_tokens": 128000,
        "mode": "chat",
        "input_cost_per_token": 1.5e-07,
        "output_cost_per_token": 6e-07,
        "supported_regions": ["global"],
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_vision": True,
    }
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_litellm_http_client_cache():
    """Ensure each test gets a fresh async HTTP client mock."""
    from litellm import in_memory_llm_clients_cache

    in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(autouse=True)
def clean_vertex_env():
    """Clear Google/Vertex AI environment variables before each test to prevent test isolation issues."""
    saved_env = {}
    env_vars_to_clear = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_PROJECT",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_AI_PROJECT",
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    yield

    for var, value in saved_env.items():
        os.environ[var] = value


# ---------------------------------------------------------------------------
# Unit tests: region and URL construction
# ---------------------------------------------------------------------------


class TestVertexBaseGetVertexRegionGemma:
    """Test the get_vertex_region method for Gemma MaaS via model_cost lookup."""

    def test_global_model_no_user_region_returns_global(self):
        vertex_base = VertexBase()

        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
                    "supported_regions": ["global"]
                }
            },
            clear=False,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region=None,
                model="google/gemma-4-26b-a4b-it-maas",
            )
            assert result == "global"

    def test_global_model_with_unsupported_user_region_overrides(self):
        vertex_base = VertexBase()

        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
                    "supported_regions": ["global"]
                }
            },
            clear=False,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region="us-central1",
                model="google/gemma-4-26b-a4b-it-maas",
            )
            assert result == "global"


class TestCreateVertexURLGemma:
    """Test that create_vertex_url produces the expected OpenAI-compatible URL.

    Gemma MaaS models reach this code path via should_use_openai_handler(), which
    selects VertexPartnerProvider.llama for all OpenAI-compatible partners including
    Gemma.  test_gemma_routes_through_openai_handler() guards that mapping so the
    URL-format tests below are meaningful regression guards for the Gemma path.
    """

    def test_gemma_routes_through_openai_handler(self):
        """Gemma MaaS must be routed through the OpenAI-compatible handler.

        This is what causes VertexPartnerProvider.llama to be selected downstream,
        which in turn generates the /endpoints/openapi URL shape.  If this mapping
        ever changes, the URL-shape tests below become misleading.
        """
        assert VertexAIPartnerModels.should_use_openai_handler(
            "google/gemma-4-26b-a4b-it-maas"
        ), "Gemma MaaS must use the OpenAI-compatible handler (VertexPartnerProvider.llama path)"

    def test_global_location_url_format(self):
        # VertexPartnerProvider.llama is correct: Gemma MaaS reaches create_vertex_url
        # via should_use_openai_handler() → partner = VertexPartnerProvider.llama.
        # See test_gemma_routes_through_openai_handler for the routing guard.
        url = VertexBase.create_vertex_url(
            vertex_location="global",
            vertex_project="test-project",
            partner=VertexPartnerProvider.llama,
            stream=False,
            model="google/gemma-4-26b-a4b-it-maas",
        )

        assert url.startswith("https://aiplatform.googleapis.com")
        assert "global-aiplatform.googleapis.com" not in url
        assert "/locations/global/" in url
        assert url.endswith("/endpoints/openapi/chat/completions")

    def test_regional_location_url_format(self):
        url = VertexBase.create_vertex_url(
            vertex_location="us-central1",
            vertex_project="test-project",
            partner=VertexPartnerProvider.llama,
            stream=False,
            model="google/gemma-4-26b-a4b-it-maas",
        )

        assert url.startswith("https://us-central1-aiplatform.googleapis.com")
        assert "/locations/us-central1/" in url
        assert url.endswith("/endpoints/openapi/chat/completions")


# ---------------------------------------------------------------------------
# Capability-flag tests: verify get_model_info surfaces the advertised flags
# ---------------------------------------------------------------------------


def test_gemma_maas_supports_function_calling():
    """supports_function_calling=true in model_cost must be surfaced by the utility."""
    with patch.dict(litellm.model_cost, _GEMMA_MODEL_COST_ENTRY, clear=False):
        assert (
            litellm.utils.supports_function_calling(
                model="vertex_ai/google/gemma-4-26b-a4b-it-maas"
            )
            is True
        )


def test_gemma_maas_supports_vision():
    """supports_vision=true in model_cost must be surfaced by the utility."""
    with patch.dict(litellm.model_cost, _GEMMA_MODEL_COST_ENTRY, clear=False):
        assert (
            litellm.utils.supports_vision(
                model="vertex_ai/google/gemma-4-26b-a4b-it-maas"
            )
            is True
        )


# ---------------------------------------------------------------------------
# Integration tests: verify payloads reach the global OpenAI endpoint
#
# Patch target note (P1): AsyncHTTPHandler is patched at its *definition* site
# (litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler).  This works
# correctly because the client is created by get_async_httpx_client(), which is
# also defined in http_handler.py and calls AsyncHTTPHandler(...) using the
# module-local name — so the patch intercepts instantiation there.
# llm_http_handler.py only imports the class for type annotations; it never
# instantiates it directly.  Confirmed: without the mock the test raises
# AuthenticationError, proving the assertion would never silently pass against
# an un-mocked real call.
# ---------------------------------------------------------------------------

_MOCK_RESPONSE_JSON = {
    "id": "chatcmpl-gemma-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "google/gemma-4-26b-a4b-it-maas",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}


@pytest.mark.asyncio
async def test_vertex_ai_gemma_global_endpoint_url():
    """
    End-to-end: acompletion on vertex_ai/google/gemma-4-26b-a4b-it-maas should
    POST to the global endpoints/openapi/chat/completions URL.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = _MOCK_RESPONSE_JSON

    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.vertex_ai_partner_models.main.VertexAIPartnerModels._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            "sys.modules",
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
        patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
                    "supported_regions": ["global"]
                }
            },
            clear=False,
        ),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        response = await litellm.acompletion(
            model="vertex_ai/google/gemma-4-26b-a4b-it-maas",
            messages=[{"role": "user", "content": "Hello"}],
            vertex_ai_project="test-project",
        )

        mock_http_handler.return_value.post.assert_called_once()

        call_args = mock_http_handler.return_value.post.call_args
        called_url = call_args.kwargs["url"]

        assert called_url.startswith("https://aiplatform.googleapis.com")
        assert "global-aiplatform.googleapis.com" not in called_url
        assert "/locations/global/" in called_url
        assert "/endpoints/openapi/chat/completions" in called_url

        assert response.model == "google/gemma-4-26b-a4b-it-maas"


@pytest.mark.asyncio
async def test_vertex_ai_gemma_function_calling_passthrough():
    """
    Tools and tool_choice defined in the acompletion call must appear in the
    JSON body POSTed to the global endpoints/openapi/chat/completions URL.

    This confirms that supports_function_calling=true is backed by real
    pass-through behaviour and that callers gating on get_model_info won't
    silently send unsupported requests.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = _MOCK_RESPONSE_JSON

    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.vertex_ai_partner_models.main.VertexAIPartnerModels._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            "sys.modules",
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
        patch.dict(litellm.model_cost, _GEMMA_MODEL_COST_ENTRY, clear=False),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        await litellm.acompletion(
            model="vertex_ai/google/gemma-4-26b-a4b-it-maas",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            vertex_ai_project="test-project",
        )

        mock_http_handler.return_value.post.assert_called_once()
        call_args = mock_http_handler.return_value.post.call_args

        # Must route to the global OpenAI-compatible endpoint
        called_url = call_args.kwargs["url"]
        assert called_url.startswith("https://aiplatform.googleapis.com"), called_url
        assert "/endpoints/openapi/chat/completions" in called_url, called_url

        # Tools and tool_choice must be forwarded in the request body
        body = json.loads(call_args.kwargs["data"])
        assert "tools" in body, f"'tools' key missing from request body: {body}"
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert "tool_choice" in body, f"'tool_choice' missing from request body: {body}"
        assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_vertex_ai_gemma_vision_passthrough():
    """
    An image_url content part must survive transformation and appear in the
    JSON body POSTed to the global endpoints/openapi/chat/completions URL.

    This confirms that supports_vision=true is backed by real pass-through
    behaviour and that callers gating on get_model_info won't silently send
    unsupported multimodal requests.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = _MOCK_RESPONSE_JSON

    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.vertex_ai_partner_models.main.VertexAIPartnerModels._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            "sys.modules",
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
        patch.dict(litellm.model_cost, _GEMMA_MODEL_COST_ENTRY, clear=False),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        await litellm.acompletion(
            model="vertex_ai/google/gemma-4-26b-a4b-it-maas",
            messages=messages,
            vertex_ai_project="test-project",
        )

        mock_http_handler.return_value.post.assert_called_once()
        call_args = mock_http_handler.return_value.post.call_args

        # Must still route to the global OpenAI-compatible endpoint
        called_url = call_args.kwargs["url"]
        assert called_url.startswith("https://aiplatform.googleapis.com"), called_url
        assert "/endpoints/openapi/chat/completions" in called_url, called_url

        # The image_url content part must be present in the forwarded body
        body = json.loads(call_args.kwargs["data"])
        user_msg = next(m for m in body["messages"] if m["role"] == "user")
        content = user_msg["content"]
        assert isinstance(content, list), f"Expected list content, got: {content}"
        image_parts = [p for p in content if p.get("type") == "image_url"]
        assert image_parts, f"No image_url part in forwarded message content: {content}"
