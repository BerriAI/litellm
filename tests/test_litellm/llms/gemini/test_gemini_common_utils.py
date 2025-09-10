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
            {
                "name": "models/gemini-1.5-pro"
            },  # ends with 'o' - would become "gemini-1.5-pr" with strip()
            {
                "name": "models/test-model"
            },  # ends with 'l' - would become "gemini/test-mode" with strip()
            {
                "name": "models/custom-models"
            },  # ends with 's' - would become "gemini/custom-model" with strip()
            {
                "name": "models/demo"
            },  # ends with 'o' - would become "gemini/dem" with strip()
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
            "promptTokensDetails": [
                {
                    "modality": "TEXT",
                    "tokenCount": 31
                }
            ]
        }
        
        with patch('litellm.llms.gemini.count_tokens.handler.GoogleAIStudioTokenCounter.acount_tokens', 
                   new_callable=AsyncMock) as mock_acount_tokens:
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
                request_model=request_model
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
                contents=contents
            )

    def test_clean_contents_for_gemini_api_removes_id_field(self):
        """Test that _clean_contents_for_gemini_api removes unsupported 'id' field from function responses"""
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
        
        token_counter = GoogleAIStudioTokenCounter()
        
        # Test contents with function response containing 'id' field (camelCase)
        contents_with_id = [
            {
                "parts": [
                    {
                        "text": "Hello world"
                    }
                ],
                "role": "user"
            },
            {
                "parts": [
                    {
                        "functionResponse": {
                            "id": "read_many_files-1757526647518-730a691aac11c",  # This should be removed
                            "name": "read_many_files",
                            "response": {
                                "output": "No files matching the criteria were found or all were skipped."
                            }
                        }
                    }
                ],
                "role": "user"
            }
        ]
        
        # Clean the contents
        cleaned_contents = token_counter._clean_contents_for_gemini_api(contents_with_id)
        
        # Verify the 'id' field was removed
        function_response = cleaned_contents[1]["parts"][0]["functionResponse"]
        assert "id" not in function_response
        assert "name" in function_response
        assert "response" in function_response
        assert function_response["name"] == "read_many_files"
        assert function_response["response"]["output"] == "No files matching the criteria were found or all were skipped."

    def test_clean_contents_for_gemini_api_removes_id_field_snake_case(self):
        """Test that _clean_contents_for_gemini_api removes unsupported 'id' field from function responses (snake_case)"""
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
        
        token_counter = GoogleAIStudioTokenCounter()
        
        # Test contents with function response containing 'id' field (snake_case)
        contents_with_id = [
            {
                "parts": [
                    {
                        "text": "Hello world"
                    }
                ],
                "role": "user"
            },
            {
                "parts": [
                    {
                        "function_response": {  # snake_case variant
                            "id": "read_many_files-1757526647518-730a691aac11c",  # This should be removed
                            "name": "read_many_files",
                            "response": {
                                "output": "No files matching the criteria were found or all were skipped."
                            }
                        }
                    }
                ],
                "role": "user"
            }
        ]
        
        # Clean the contents
        cleaned_contents = token_counter._clean_contents_for_gemini_api(contents_with_id)
        
        # Verify the 'id' field was removed
        function_response = cleaned_contents[1]["parts"][0]["function_response"]
        assert "id" not in function_response
        assert "name" in function_response
        assert "response" in function_response
        assert function_response["name"] == "read_many_files"
        assert function_response["response"]["output"] == "No files matching the criteria were found or all were skipped."

    def test_clean_contents_for_gemini_api_preserves_other_fields(self):
        """Test that _clean_contents_for_gemini_api preserves other fields and structure"""
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
        
        token_counter = GoogleAIStudioTokenCounter()
        
        # Test contents without function responses
        contents_without_function_response = [
            {
                "parts": [
                    {
                        "text": "This is a regular message"
                    }
                ],
                "role": "user"
            },
            {
                "parts": [
                    {
                        "text": "This is a model response"
                    }
                ],
                "role": "model"
            }
        ]
        
        # Clean the contents
        cleaned_contents = token_counter._clean_contents_for_gemini_api(contents_without_function_response)
        
        # Verify the contents are unchanged
        assert cleaned_contents == contents_without_function_response

    def test_clean_contents_for_gemini_api_handles_empty_contents(self):
        """Test that _clean_contents_for_gemini_api handles empty or invalid contents gracefully"""
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
        
        token_counter = GoogleAIStudioTokenCounter()
        
        # Test with None
        assert token_counter._clean_contents_for_gemini_api(None) is None
        
        # Test with empty list
        assert token_counter._clean_contents_for_gemini_api([]) == []
        
        # Test with non-list input
        assert token_counter._clean_contents_for_gemini_api("not a list") == "not a list"
