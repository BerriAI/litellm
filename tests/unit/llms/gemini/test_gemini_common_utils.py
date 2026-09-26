import json
from unittest.mock import AsyncMock, patch

import pytest

from litellm.llms.gemini.common_utils import GeminiModelInfo, GoogleAIStudioTokenCounter


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
            {"name": "models/gemini-1.5-pro"},
            {"name": "models/test-model"},
            {"name": "models/custom-models"},
            {"name": "models/demo"},
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
                system_instruction=None,
                tools=None,
                client=None,
                model=model_to_use,
                contents=tuple(contents),
            )

    @pytest.mark.asyncio
    async def test_count_tokens_translates_anthropic_messages_system_and_tools(self):
        import httpx

        recorded: list = []

        def _handler(request):
            recorded.append(request)
            return httpx.Response(200, json={"totalTokens": 12})

        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello world"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://gemini.example.test"}},
            request_model="gemini/gemini-2.5-flash",
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            system="You are a helpful assistant",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.total_tokens == 12
        body = json.loads(recorded[-1].content)
        generate_content_request = body["generateContentRequest"]
        assert generate_content_request["contents"]
        assert generate_content_request["contents"][0]["parts"][0].get("text") == "hello world"
        assert generate_content_request["systemInstruction"]["parts"][0].get("text") == "You are a helpful assistant"
        assert generate_content_request["tools"][0]["function_declarations"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_count_tokens_passes_system_and_tools_with_native_contents(self):
        import httpx

        recorded: list = []

        def _handler(request):
            recorded.append(request)
            return httpx.Response(200, json={"totalTokens": 20})

        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=[{"role": "user", "parts": [{"text": "hello world"}]}],
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://gemini.example.test"}},
            request_model="gemini/gemini-2.5-flash",
            system={"parts": [{"text": "You are a helpful assistant"}]},
            tools=[{"function_declarations": [{"name": "get_weather"}]}],
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.total_tokens == 20
        body = json.loads(recorded[-1].content)
        generate_content_request = body["generateContentRequest"]
        assert generate_content_request["contents"] == [{"role": "user", "parts": [{"text": "hello world"}]}]
        assert generate_content_request["systemInstruction"] == {"parts": [{"text": "You are a helpful assistant"}]}
        assert generate_content_request["tools"][0]["function_declarations"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_count_tokens_provider_error_returns_error_response(self):
        import httpx

        def _handler(request):
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}},
            )

        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello world"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://gemini.example.test"}},
            request_model="gemini/gemini-2.5-flash",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 400
        assert result.total_tokens == 0
        assert result.error_message is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_messages",
        [
            [{"role": "tool", "content": "orphaned result", "tool_call_id": "missing-call"}],
            [{"role": "user", "content": [{"type": "text", "text": 123}]}],
        ],
    )
    async def test_count_tokens_translation_error_falls_back(self, bad_messages):
        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=bad_messages,
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini/gemini-2.5-flash",
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 400
        assert result.total_tokens == 0
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_count_tokens_malformed_anthropic_input_returns_400_without_http_call(self):
        import httpx

        recorded: list = []

        def _handler(request):
            recorded.append(request)
            return httpx.Response(200, json={"totalTokens": 7})

        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": [{"type": "tool_result", "content": "18C"}]}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://gemini.example.test"}},
            request_model="gemini/gemini-2.5-flash",
            system=None,
            tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 400
        assert result.total_tokens == 0
        assert recorded == []

    @pytest.mark.asyncio
    async def test_count_tokens_without_api_key_returns_provider_error_response(self, monkeypatch):
        import httpx

        import litellm

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        recorded = []

        def _handler(request):
            recorded.append(request)
            return httpx.Response(
                403, json={"error": {"code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED"}}
            )

        result = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"model": "gemini/gemini-2.5-flash"}},
            request_model="gemini/gemini-2.5-flash",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 403
        assert result.total_tokens == 0
        assert len(recorded) == 1 and "x-goog-api-key" not in recorded[0].headers

    @pytest.mark.asyncio
    async def test_count_tokens_connection_error_returns_error_response(self):
        import litellm

        token_counter = GoogleAIStudioTokenCounter()

        with patch(
            "litellm.llms.gemini.count_tokens.handler.GoogleAIStudioTokenCounter.acount_tokens",
            new_callable=AsyncMock,
        ) as mock_acount_tokens:
            mock_acount_tokens.side_effect = litellm.APIConnectionError(
                message="connection refused", llm_provider="gemini", model="gemini-2.5-flash"
            )

            result = await token_counter.count_tokens(
                model_to_use="gemini-2.5-flash",
                messages=[{"role": "user", "content": "hello"}],
                contents=None,
                deployment=None,
                request_model="gemini/gemini-2.5-flash",
            )

        assert result is not None
        assert result.error is True
        assert result.status_code == 500
        assert "connection refused" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_count_tokens_response_without_total_tokens_returns_502(self):
        import httpx

        recorded: list = []

        def _handler(request):
            recorded.append(request)
            return httpx.Response(200, json={"promptTokensDetails": []})

        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key", "api_base": "https://gemini.example.test"}},
            request_model="gemini/gemini-2.5-flash",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert recorded != []
        assert result is not None
        assert result.error is True
        assert result.status_code == 502
        assert result.total_tokens == 0
        assert "totalTokens" in (result.error_message or "")
        assert result.original_response == {"promptTokensDetails": []}

    @pytest.mark.asyncio
    async def test_count_tokens_returns_none_without_contents_or_messages(self):
        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=None,
            deployment=None,
            request_model="gemini/gemini-2.5-flash",
        )

        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_tool_names, counted_tool_names",
        [(None, ["deployment_default_tool"]), (["request_tool"], ["request_tool"])],
    )
    async def test_count_tokens_counts_deployment_tools_the_router_would_send(
        self, request_tool_names: list[str] | None, counted_tool_names: list[str]
    ):
        import httpx

        recorded: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            recorded.append(request)
            return httpx.Response(200, json={"totalTokens": 9})

        def _function_tool(name: str) -> dict[str, object]:
            return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}

        result = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "api_key": "test-key",
                    "tools": [_function_tool("deployment_default_tool")],
                    "system_instruction": {"parts": [{"text": "deployment default"}]},
                    "client": object(),
                    "self": "bogus",
                }
            },
            request_model="gemini/gemini-2.5-flash",
            tools=None if request_tool_names is None else [_function_tool(name) for name in request_tool_names],
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.error is not True
        assert result.total_tokens == 9
        generate_content_request = json.loads(recorded[-1].content)["generateContentRequest"]
        assert "systemInstruction" not in generate_content_request
        declared_names = [
            declaration["name"]
            for tool in generate_content_request["tools"]
            for declaration in tool["function_declarations"]
        ]
        assert declared_names == counted_tool_names

    @pytest.mark.asyncio
    async def test_count_tokens_native_tools_not_mappings_returns_400_without_http_call(self):
        token_counter = GoogleAIStudioTokenCounter()

        result = await token_counter.count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=None,
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini/gemini-2.5-flash",
            tools=["just-a-string"],
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 400
        assert "Invalid token count request" in (result.error_message or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "upstream_json",
        [
            {"totalTokens": "abc"},
            {"promptTokensDetails": []},
            [{"totalTokens": 5}],
        ],
    )
    async def test_count_tokens_malformed_provider_response_returns_502(self, upstream_json):
        import httpx

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=upstream_json)

        result = await GoogleAIStudioTokenCounter().count_tokens(
            model_to_use="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="gemini/gemini-2.5-flash",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 502
        assert "totalTokens" in (result.error_message or "")

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
