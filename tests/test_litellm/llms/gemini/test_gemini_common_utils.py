import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.common_utils import GeminiModelInfo, GoogleAIStudioTokenCounter
from litellm.types.utils import TokenCountResponse


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
                api_key=None,
                api_base=None,
                contents=contents,
                system_instruction=None,
                tools=None,
                client=None,
            )

    @staticmethod
    def _counter_with_upstream(
        upstream_response: httpx.Response,
    ) -> tuple[GoogleAIStudioTokenCounter, list[httpx.Request]]:
        seen_requests: list[httpx.Request] = []

        def upstream(request: httpx.Request) -> httpx.Response:
            seen_requests.append(request)
            return upstream_response

        client = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
        return GoogleAIStudioTokenCounter(client=client), seen_requests

    @pytest.mark.asyncio
    async def test_anthropic_messages_are_sent_as_gemini_contents_with_system_and_tools(self):
        counter, seen = self._counter_with_upstream(httpx.Response(200, json={"totalTokens": 42}))

        result = await counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            contents=None,
            deployment={"litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "test-key"}},
            request_model="gemini-flash",
            tools=[
                {"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
            ],
            system="Be terse",
        )

        assert len(seen) == 1, seen
        assert seen[0].url.path == "/v1beta/models/gemini-2.5-flash:countTokens"
        assert seen[0].headers["x-goog-api-key"] == "test-key"
        assert json.loads(seen[0].content) == {
            "generateContentRequest": {
                "model": "models/gemini-2.5-flash",
                "contents": [{"role": "user", "parts": [{"text": "What is the weather in Paris?"}]}],
                "systemInstruction": {"parts": [{"text": "Be terse"}]},
                "tools": [
                    {
                        "function_declarations": [
                            {
                                "name": "get_weather",
                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                            }
                        ]
                    }
                ],
            }
        }
        assert result == TokenCountResponse(
            total_tokens=42,
            request_model="gemini-flash",
            model_used="gemini-2.5-flash",
            tokenizer_type="",
            original_response={"totalTokens": 42},
        )

    @pytest.mark.asyncio
    async def test_system_without_tools_still_wraps_in_generate_content_request(self):
        counter, seen = self._counter_with_upstream(httpx.Response(200, json={"totalTokens": 9}))

        await counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini-flash",
            system=[{"type": "text", "text": "Be terse"}],
        )

        assert json.loads(seen[0].content) == {
            "generateContentRequest": {
                "model": "models/gemini-2.5-flash",
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "systemInstruction": {"parts": [{"text": "Be terse"}]},
            }
        }

    @pytest.mark.asyncio
    async def test_native_contents_are_sent_unchanged(self):
        counter, seen = self._counter_with_upstream(httpx.Response(200, json={"totalTokens": 3}))
        contents = [{"role": "user", "parts": [{"text": "Hello world"}]}]

        result = await counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=contents,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini-flash",
        )

        assert json.loads(seen[0].content) == {"contents": contents}
        assert result is not None and result.total_tokens == 3

    @pytest.mark.asyncio
    async def test_provider_rejection_is_returned_as_error_value_not_raised(self):
        counter, _ = self._counter_with_upstream(
            httpx.Response(400, json={"error": {"message": "contents is not specified"}})
        )

        result = await counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini-flash",
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 400
        assert result.error_message is not None and "contents is not specified" in result.error_message

    @pytest.mark.asyncio
    async def test_success_without_total_tokens_is_an_error_value_not_zero(self):
        counter, _ = self._counter_with_upstream(httpx.Response(200, json={"promptTokensDetails": []}))

        result = await counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini-flash",
        )

        assert result == TokenCountResponse(
            total_tokens=0,
            request_model="gemini-flash",
            model_used="gemini-2.5-flash",
            tokenizer_type="gemini_api",
            original_response={"promptTokensDetails": []},
            error=True,
            error_message="Google Gen AI Studio countTokens response has no totalTokens",
            status_code=502,
        )

    @pytest.mark.asyncio
    async def test_nothing_to_count_skips_the_provider_call(self):
        counter, seen = self._counter_with_upstream(httpx.Response(200, json={"totalTokens": 0}))

        result = await counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini-flash",
        )

        assert result is None
        assert seen == []

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
