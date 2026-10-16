from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.common_utils import GeminiModelInfo, GoogleAIStudioTokenCounter
from litellm.llms.gemini.count_tokens import handler as gemini_count_tokens_handler
from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter as GeminiCountTokensHandler

_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])


class _GeminiCountTokensPeer:
    def __init__(self) -> None:
        self.request_body: dict[str, JsonValue] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_body = _JSON_BODY.validate_json(request.content)
        return httpx.Response(200, json={"totalTokens": 11}, request=request)


@pytest.mark.asyncio
async def test_google_ai_studio_count_tokens_forwards_system_and_tools() -> None:
    peer: Final = _GeminiCountTokensPeer()
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        result: Final = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            deployment={
                "litellm_params": {
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                    "system_instruction": {"parts": [{"text": "deployment-level"}]},
                }
            },
            request_model="gemini/gemini-2.5-flash",
            tools=[
                {
                    "name": "lookup",
                    "description": "Find a record",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            system="You are terse.",
        )

    assert result is not None
    assert result.error is False
    assert result.total_tokens == 11
    assert peer.request_body == {
        "generateContentRequest": {
            "model": "models/gemini-2.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "systemInstruction": {"parts": [{"text": "You are terse."}]},
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "lookup",
                            "description": "Find a record",
                            "parameters": {"type": "object"},
                        }
                    ]
                }
            ],
        }
    }


@pytest.mark.asyncio
async def test_google_ai_studio_count_tokens_rejects_image_system_without_request() -> None:
    peer: Final = _GeminiCountTokensPeer()
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        result: Final = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://count-tokens.test"}},
            request_model="gemini/gemini-2.5-flash",
            system=[
                {"type": "text", "text": "You are terse."},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                },
            ],
        )

    assert result is not None
    assert result.error is True
    assert result.status_code == 400
    assert peer.request_body is None


@pytest.mark.asyncio
async def test_google_ai_studio_count_tokens_rejects_an_image_url_it_refuses_to_fetch_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", True)
    peer: Final = _GeminiCountTokensPeer()
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        result: Final = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "url", "url": "http://127.0.0.1:9/image.png"}}],
                }
            ],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://count-tokens.test"}},
            request_model="gemini/gemini-2.5-flash",
            messages_format="anthropic",
        )

    assert result is not None
    assert result.error is True
    assert result.status_code == 400
    assert "http://127.0.0.1:9/image.png" in (result.error_message or "")
    assert peer.request_body is None


@pytest.mark.asyncio
async def test_google_ai_studio_count_tokens_answers_with_the_status_google_rejected_it_with() -> None:
    rejection: Final = '{"error": {"code": 400, "message": "Invalid function name", "status": "INVALID_ARGUMENT"}}'
    client: Final = AsyncHTTPHandler(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text=rejection, request=request))
    )

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        result: Final = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://count-tokens.test"}},
            request_model="gemini/gemini-2.5-flash",
        )

    assert result is not None
    assert result.error is True
    assert result.status_code == 400
    assert "Invalid function name" in (result.error_message or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("system_instruction", "tools", "expected_body"),
    (
        (
            {"parts": [{"text": "Follow the system rule."}]},
            [{"functionDeclarations": [{"name": "lookup", "parameters": {"type": "OBJECT"}}]}],
            {
                "generateContentRequest": {
                    "model": "models/gemini-4-argon",
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    "systemInstruction": {"parts": [{"text": "Follow the system rule."}]},
                    "tools": [{"functionDeclarations": [{"name": "lookup", "parameters": {"type": "OBJECT"}}]}],
                }
            },
        ),
        (
            None,
            [],
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        ),
    ),
)
async def test_gemini_count_tokens_sends_optional_request_fields(
    system_instruction: dict[str, JsonValue] | None,
    tools: list[dict[str, JsonValue]],
    expected_body: dict[str, JsonValue],
) -> None:
    peer: Final = _GeminiCountTokensPeer()
    token_counter: Final = GeminiCountTokensHandler(client=AsyncHTTPHandler(transport=httpx.MockTransport(peer)))

    result: Final = await token_counter.acount_tokens(
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        model="gemini-4-argon",
        api_key="test-key",
        api_base="https://count-tokens.test",
        system_instruction=system_instruction,
        tools=tools,
    )

    assert result == {"totalTokens": 11}
    assert peer.request_body == expected_body


class TestGeminiModelInfo:
    """Test suite for GeminiModelInfo class"""

    def test_process_model_name_normal_cases(self):
        """Test process_model_name with normal model names"""
        gemini_model_info = GeminiModelInfo()

        # Test with normal model names
        models = [
            {"name": "models/gemini-1.5-flash"},
            {"name": "models/gemini-1.5-pro"},
            {"name": "models/gemini-2.0-flash-exp"},
        ]

        result = gemini_model_info.process_model_name(models)

        expected = [
            "gemini/gemini-1.5-flash",
            "gemini/gemini-1.5-pro",
            "gemini/gemini-2.0-flash-exp",
        ]

        assert result == expected

    def test_process_model_name_edge_cases(self):
        """Test process_model_name with edge cases that could be affected by strip() vs replace()"""
        gemini_model_info = GeminiModelInfo()

        # Test edge cases where model names end with characters from "models/"
        # These would be incorrectly processed if using strip("models/") instead of replace("models/", "")
        models = [
            {"name": "models/gemini-1.5-pro"},  # ends with 'o' - would become "gemini-1.5-pr" with strip()
            {"name": "models/test-model"},  # ends with 'l' - would become "gemini/test-mode" with strip()
            {"name": "models/custom-models"},  # ends with 's' - would become "gemini/custom-model" with strip()
            {"name": "models/demo"},  # ends with 'o' - would become "gemini/dem" with strip()
        ]

        result = gemini_model_info.process_model_name(models)

        expected = [
            "gemini/gemini-1.5-pro",  # 'o' should be preserved
            "gemini/test-model",  # 'l' should be preserved
            "gemini/custom-models",  # 's' should be preserved
            "gemini/demo",  # 'o' should be preserved
        ]

        assert result == expected

    def test_process_model_name_empty_list(self):
        """Test process_model_name with empty list"""
        gemini_model_info = GeminiModelInfo()

        result = gemini_model_info.process_model_name([])

        assert result == []

    def test_process_model_name_no_models_prefix(self):
        """Test process_model_name with model names that don't have 'models/' prefix"""
        gemini_model_info = GeminiModelInfo()

        models = [
            {"name": "gemini-1.5-flash"},  # No "models/" prefix
            {"name": "custom-model"},
        ]

        result = gemini_model_info.process_model_name(models)

        expected = [
            "gemini/gemini-1.5-flash",
            "gemini/custom-model",
        ]

        assert result == expected


class TestGoogleAIStudioTokenCounter:
    """Test suite for GoogleAIStudioTokenCounter class"""

    def test_should_use_token_counting_api(self):
        """Test should_use_token_counting_api method with different provider values"""
        from litellm.types.utils import LlmProviders

        token_counter = GoogleAIStudioTokenCounter()

        # Test with gemini provider - should return True
        assert token_counter.should_use_token_counting_api(LlmProviders.GEMINI.value) is True

        # Test with other providers - should return False
        assert token_counter.should_use_token_counting_api(LlmProviders.OPENAI.value) is False
        assert token_counter.should_use_token_counting_api("anthropic") is False
        assert token_counter.should_use_token_counting_api("vertex_ai") is False

        # Test with None - should return False
        assert token_counter.should_use_token_counting_api(None) is False

    @pytest.mark.asyncio
    async def test_count_tokens(self):
        """Test count_tokens method with mocked API response"""
        from litellm.types.utils import TokenCountResponse

        token_counter = GoogleAIStudioTokenCounter()

        # Mock the GoogleAIStudioTokenCounter from handler module
        mock_response = {
            "totalTokens": 31,
            "totalBillableCharacters": 96,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 31}],
        }

        with patch(
            "litellm.llms.gemini.count_tokens.handler.GoogleAIStudioTokenCounter.acount_tokens",
            new_callable=AsyncMock,
        ) as mock_acount_tokens:
            mock_acount_tokens.return_value = mock_response

            # Test data
            model_to_use = "gemini-1.5-flash"
            contents = [{"parts": [{"text": "Hello world"}]}]
            request_model = "gemini/gemini-1.5-flash"

            # Call the method
            result = await token_counter.count_tokens(
                model_to_use=model_to_use,
                messages=None,
                contents=contents,
                deployment=None,
                request_model=request_model,
            )

            # Verify the result
            assert result is not None
            assert isinstance(result, TokenCountResponse)
            assert result.total_tokens == 31
            assert result.request_model == request_model
            assert result.model_used == model_to_use
            assert result.original_response == mock_response

            # Verify the mock was called correctly
            mock_acount_tokens.assert_called_once_with(
                model=model_to_use,
                contents=contents,
                system_instruction=None,
                tools=None,
            )

    def test_clean_contents_for_gemini_api_removes_id_field(self):
        """Test that _clean_contents_for_gemini_api removes unsupported 'id' field from function responses"""
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter

        token_counter = GoogleAIStudioTokenCounter()

        # Test contents with function response containing 'id' field (camelCase)
        contents_with_id = [
            {"parts": [{"text": "Hello world"}], "role": "user"},
            {
                "parts": [
                    {
                        "functionResponse": {
                            "id": "read_many_files-1757526647518-730a691aac11c",  # This should be removed
                            "name": "read_many_files",
                            "response": {"output": "No files matching the criteria were found or all were skipped."},
                        }
                    }
                ],
                "role": "user",
            },
        ]

        # Clean the contents
        cleaned_contents = token_counter._clean_contents_for_gemini_api(contents_with_id)

        # Verify the 'id' field was removed
        function_response = cleaned_contents[1]["parts"][0]["functionResponse"]
        assert "id" not in function_response
        assert "name" in function_response
        assert "response" in function_response
        assert function_response["name"] == "read_many_files"
        assert (
            function_response["response"]["output"] == "No files matching the criteria were found or all were skipped."
        )

    def test_clean_contents_for_gemini_api_preserves_other_fields(self):
        """Test that _clean_contents_for_gemini_api preserves other fields and structure"""
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter

        token_counter = GoogleAIStudioTokenCounter()

        # Test contents without function responses
        contents_without_function_response = [
            {"parts": [{"text": "This is a regular message"}], "role": "user"},
            {"parts": [{"text": "This is a model response"}], "role": "model"},
        ]

        # Clean the contents
        cleaned_contents = token_counter._clean_contents_for_gemini_api(contents_without_function_response)

        # Verify the contents are unchanged
        assert cleaned_contents == contents_without_function_response
