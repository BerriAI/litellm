import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import respx
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.types.llms.openai import (
    ImageGenerationPartialImageEvent,
    IncompleteDetails,
    OutputTextDeltaEvent,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseInputParam,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import Choices, Message, ModelResponse
from tests.unit.proxy.conftest import httpx_transport
import time

_ARTIFACT_FIELD_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'


@pytest.fixture
def restore_litellm_set_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "set_verbose", litellm.set_verbose)


def validate_responses_api_response(response: ResponsesAPIResponse, final_chunk: bool = False) -> bool:
    response_fields: Final = cast(Mapping[str, object], response)
    assert isinstance(response, ResponsesAPIResponse)
    assert "id" in response_fields and isinstance(response_fields["id"], str)
    assert "created_at" in response_fields and isinstance(response_fields["created_at"], int)

    response_status: Final = response_fields.get("status")
    if response_status == "completed":
        assert "output" in response_fields and isinstance(response_fields["output"], list)

    optional_fields: Final = (
        ("error", (dict, type(None))),
        ("incomplete_details", (IncompleteDetails, type(None))),
        ("instructions", (str, type(None))),
        ("metadata", dict),
        ("model", str),
        ("object", str),
        ("parallel_tool_calls", (bool, type(None))),
        ("temperature", (int, float, type(None))),
        ("tool_choice", (dict, str, type(None))),
        ("tools", (list, type(None))),
        ("top_p", (int, float, type(None))),
        ("max_output_tokens", (int, type(None))),
        ("previous_response_id", (str, type(None))),
        ("reasoning", (dict, type(None))),
        ("status", str),
        ("text", dict),
        ("truncation", (str, type(None))),
        ("usage", ResponseAPIUsage if final_chunk else type(None)),
        ("user", (str, type(None))),
        ("store", (bool, type(None))),
    )
    for field, expected_type in optional_fields:
        if field in response_fields:
            assert isinstance(response_fields[field], expected_type)

    if final_chunk and response_status == "completed":
        assert len(cast(list[object], response_fields["output"])) > 0

    return True


class TestOpenAIResponsesAPIConfig:
    def setup_method(self):
        self.config = OpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {"input": "Hello world", "temperature": 0.7, "stream": True}

        result = self.config.map_openai_params(
            response_api_optional_params=test_params,
            model=self.model,
            drop_params=False,
        )

        # The function should return the params unchanged
        assert result == test_params

    @pytest.mark.parametrize("max_output_tokens", [1, 15])
    def test_map_openai_params_clamps_max_output_tokens_below_minimum(self, max_output_tokens):
        """OpenAI's Responses API rejects max_output_tokens < 16.

        Claude Code (via the Anthropic Messages -> Responses adapter) sends a
        max_tokens=1 warmup probe when running `/model`, which produced:
            "Invalid 'max_output_tokens': integer below minimum value.
             Expected a value >= 16, but got 1 instead."
        Clamp anything below the minimum up to 16 instead of erroring.
        """
        result = self.config.map_openai_params(
            response_api_optional_params={"max_output_tokens": max_output_tokens},
            model=self.model,
            drop_params=False,
        )

        assert result["max_output_tokens"] == 16

    def test_map_openai_params_preserves_max_output_tokens_at_or_above_minimum(self):
        """Values already >= 16 must pass through untouched."""
        result = self.config.map_openai_params(
            response_api_optional_params={"max_output_tokens": 256},
            model=self.model,
            drop_params=False,
        )

        assert result["max_output_tokens"] == 256

    def test_map_openai_params_leaves_max_output_tokens_absent(self):
        """A request without max_output_tokens must not gain the key."""
        result = self.config.map_openai_params(
            response_api_optional_params={"input": "hi"},
            model=self.model,
            drop_params=False,
        )

        assert "max_output_tokens" not in result

    @pytest.mark.parametrize(
        "value, expected",
        [
            (1, 16),
            (15, 16),
            (16, 16),
            (17, 17),
            (256, 256),
            (None, None),
        ],
    )
    def test_enforce_min_max_output_tokens(self, value, expected):
        """Below the minimum clamps to 16; the boundary, larger values, and None
        are returned unchanged so no previously-valid request regresses."""
        assert self.config._enforce_min_max_output_tokens(value) == expected

    def validate_responses_api_request_params(self, params, expected_fields):
        """
        Validate that the params dict has the expected structure of ResponsesAPIRequestParams

        Args:
            params: The dict to validate
            expected_fields: Dict of field names and their expected values
        """
        # Check that it's a dict
        assert isinstance(params, dict), "Result should be a dict"

        # Check expected fields have correct values
        for field, value in expected_fields.items():
            assert field in params, f"Missing expected field: {field}"
            assert (
                params[field] == value
            ), f"Field {field} has value {params[field]}, expected {value}"

    def test_transform_responses_api_request(self):
        """Test request transformation"""
        input_text = "What is the capital of France?"
        optional_params = {"temperature": 0.7, "stream": True, "background": True}

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_text,
            response_api_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Validate the result has the expected structure and values
        expected_fields = {
            "model": self.model,
            "input": input_text,
            "temperature": 0.7,
            "stream": True,
            "background": True,
        }

        self.validate_responses_api_request_params(result, expected_fields)

    def test_transform_strips_cache_control_from_input_content_blocks(self):
        """`cache_control` markers (Anthropic-only) must be stripped from
        Responses API input content blocks before sending to OpenAI.

        OpenAI rejects unknown params on input content blocks with HTTP 400:
            "Unknown parameter: 'input[0].content[0].cache_control'"
        Chat Completions strips these via
        `remove_cache_control_flag_from_messages_and_tools`; the Responses
        path must do the same.
        """
        input_with_cache_control = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_with_cache_control,
            response_api_optional_request_params={},
            litellm_params={},
            headers={},
        )

        assert "cache_control" not in result["input"][0]["content"][0]
        assert result["input"][0]["content"][0]["type"] == "input_text"
        assert result["input"][0]["content"][0]["text"] == "Hello"

    def test_transform_strips_cache_control_from_tools(self):
        """`cache_control` markers must also be stripped from tools for
        symmetry with the Chat Completions path. OpenAI currently accepts
        cache_control on tools silently but stripping keeps the wire payload
        clean and matches `remove_cache_control_flag_from_messages_and_tools`.
        """
        tools_with_cache_control = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                "cache_control": {"type": "ephemeral"},
            }
        ]

        result = self.config.transform_responses_api_request(
            model=self.model,
            input="hi",
            response_api_optional_request_params={"tools": tools_with_cache_control},
            litellm_params={},
            headers={},
        )

        assert "cache_control" not in result["tools"][0]
        assert result["tools"][0]["name"] == "get_weather"

    def test_transform_preserves_input_without_cache_control(self):
        """Inputs without cache_control must pass through unmodified."""
        input_clean = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_clean,
            response_api_optional_request_params={},
            litellm_params={},
            headers={},
        )

        assert result["input"] == input_clean

    def test_transform_drops_foreign_tool_call_item_ids(self):
        """Replayed tool call items whose ids are not OpenAI-shaped (e.g.
        Anthropic toolu_/srvtoolu_ ids after a router fallback) must be sent
        without an id: OpenAI 400s foreign ids ("Expected an ID that begins
        with 'fc'") but accepts the items with no id at all. Genuine fc_/ctc_
        ids and non-tool-call items pass through untouched."""
        replayed_input = [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "function_call",
                "id": "toolu_01Foreign",
                "call_id": "toolu_01Foreign",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {"type": "function_call_output", "call_id": "toolu_01Foreign", "output": "sunny"},
            {
                "type": "custom_tool_call",
                "id": "srvtoolu_01Foreign",
                "call_id": "srvtoolu_01Foreign",
                "name": "apply_patch",
                "input": "patch",
            },
            {
                "type": "function_call",
                "id": "fc_genuine",
                "call_id": "call_genuine",
                "name": "get_weather",
                "arguments": "{}",
            },
            {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
        ]

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=replayed_input,
            response_api_optional_request_params={},
            litellm_params={},
            headers={},
        )

        assert "id" not in result["input"][1]
        assert result["input"][1]["call_id"] == "toolu_01Foreign"
        assert "id" not in result["input"][3]
        assert result["input"][3]["call_id"] == "srvtoolu_01Foreign"
        assert result["input"][4]["id"] == "fc_genuine"
        assert result["input"][5]["id"] == "msg_1"
        assert replayed_input[1]["id"] == "toolu_01Foreign"
        assert replayed_input[3]["id"] == "srvtoolu_01Foreign"

    def test_transform_keeps_foreign_tool_call_item_ids_for_other_providers(self):
        """Providers reusing this config that do not enforce OpenAI's id
        shapes must keep replayed ids untouched."""
        from litellm.types.utils import LlmProviders

        class _OpenRouterLikeConfig(OpenAIResponsesAPIConfig):
            @property
            def custom_llm_provider(self) -> LlmProviders:
                return LlmProviders.OPENROUTER

        replayed_input = [
            {
                "type": "function_call",
                "id": "toolu_01Foreign",
                "call_id": "toolu_01Foreign",
                "name": "get_weather",
                "arguments": "{}",
            }
        ]

        result = _OpenRouterLikeConfig().transform_responses_api_request(
            model="openrouter/some-model",
            input=replayed_input,
            response_api_optional_request_params={},
            litellm_params={},
            headers={},
        )

        assert result["input"][0]["id"] == "toolu_01Foreign"

    @pytest.mark.parametrize(
        "raw_parameters",
        [
            '{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}',
            '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}',
        ],
    )
    def test_transform_decodes_json_string_tool_parameters(self, raw_parameters: str):
        """A JSON-encoded schema must reach the provider as an object."""
        result = self.config.transform_responses_api_request(
            model=self.model,
            input="weather in Paris",
            response_api_optional_request_params={
                "tools": [{"type": "function", "name": "get_weather", "parameters": raw_parameters}]
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0]["parameters"] == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    def test_transform_decodes_json_string_tool_parameters_on_compact_request(self):
        """The compact request path builds the same wire body, so it must decode too."""
        _url, data = self.config.transform_compact_response_api_request(
            model=self.model,
            input="weather in Paris",
            response_api_optional_request_params={
                "tools": [{"type": "function", "name": "get_weather", "parameters": '{"type": "object"}'}]
            },
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["tools"][0]["parameters"] == {"type": "object"}

    @pytest.mark.parametrize("raw_parameters", ['"just a string"', "not json at all", "[1, 2, 3]", 42])
    def test_transform_rejects_tool_parameters_that_are_not_an_object(self, raw_parameters: object):
        """Neither an object nor a string encoding one is a client error naming the tool index."""
        with pytest.raises(litellm.BadRequestError) as exc_info:
            self.config.transform_responses_api_request(
                model=self.model,
                input="weather in Paris",
                response_api_optional_request_params={
                    "tools": [
                        {"type": "web_search_preview"},
                        {"type": "function", "name": "get_weather", "parameters": raw_parameters},
                    ]
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        assert "tools[1].parameters" in str(exc_info.value)

    def test_transform_leaves_object_null_and_absent_tool_parameters_untouched(self):
        """The API accepts an object schema, an explicit null, an omitted schema and a built-in
        tool, so decoding must forward all four unchanged rather than raising."""
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        tools = [
            {"type": "function", "name": "get_weather", "parameters": schema},
            {"type": "function", "name": "null_args", "parameters": None},
            {"type": "function", "name": "no_args"},
            {"type": "web_search_preview"},
        ]

        result = self.config.transform_responses_api_request(
            model=self.model,
            input="weather in Paris",
            response_api_optional_request_params={"tools": tools},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0]["parameters"] == schema
        assert result["tools"][1]["parameters"] is None
        assert "parameters" not in result["tools"][2]
        assert result["tools"][3] == {"type": "web_search_preview"}

    def test_transform_compact_drops_foreign_tool_call_item_ids(self):
        """The compact request path replays input the same way, so it must
        apply the same id drop."""
        replayed_input = [
            {
                "type": "function_call",
                "id": "toolu_01Foreign",
                "call_id": "toolu_01Foreign",
                "name": "get_weather",
                "arguments": "{}",
            }
        ]

        _url, data = self.config.transform_compact_response_api_request(
            model=self.model,
            input=replayed_input,
            response_api_optional_request_params={},
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "id" not in data["input"][0]
        assert data["input"][0]["call_id"] == "toolu_01Foreign"

    def test_transform_streaming_response(self):
        """Test streaming response transformation"""
        # Test with a text delta event
        chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        assert result.delta == "Hello"
        assert result.item_id == "item_123"

        # Test with a completed event - providing all required fields
        completed_chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "created_at": 1234567890,
                "model": "gpt-4o",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": None,
                "temperature": 0.7,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "max_output_tokens": None,
                "previous_response_id": None,
                "reasoning": None,
                "status": "completed",
                "text": None,
                "truncation": "auto",
                "usage": None,
                "user": None,
            },
        }

        # Mock the get_event_model_class to avoid validation issues in tests
        with patch.object(
            OpenAIResponsesAPIConfig, "get_event_model_class"
        ) as mock_get_class:
            mock_get_class.return_value = ResponseCompletedEvent

            result = self.config.transform_streaming_response(
                model=self.model,
                parsed_chunk=completed_chunk,
                logging_obj=self.logging_obj,
            )

            assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            assert result.response.id == "resp_123"

    @pytest.mark.serial
    def test_validate_environment(self):
        """Test that validate_environment correctly sets the Authorization header"""
        # Test with provided API key
        headers = {}
        api_key = "test_api_key"
        litellm_params = GenericLiteLLMParams(api_key=api_key)
        result = self.config.validate_environment(
            headers=headers, model=self.model, litellm_params=litellm_params
        )

        assert "Authorization" in result
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

        # Test with empty headers
        headers = {}

        with patch("litellm.api_key", "litellm_api_key"):
            litellm_params = GenericLiteLLMParams()
            result = self.config.validate_environment(
                headers=headers, model=self.model, litellm_params=litellm_params
            )

            assert "Authorization" in result
            assert result["Authorization"] == "Bearer litellm_api_key"

        # Test with existing headers
        headers = {"Content-Type": "application/json"}

        with patch("litellm.openai_key", "openai_key"):
            with patch("litellm.api_key", None):
                litellm_params = GenericLiteLLMParams()
                result = self.config.validate_environment(
                    headers=headers, model=self.model, litellm_params=litellm_params
                )

                assert "Authorization" in result
                assert result["Authorization"] == "Bearer openai_key"
                assert "Content-Type" in result
                assert result["Content-Type"] == "application/json"

        # Test with environment variable
        headers = {}

        with patch("litellm.api_key", None):
            with patch("litellm.openai_key", None):
                with patch(
                    "litellm.llms.openai.responses.transformation.get_secret_str",
                    return_value="env_api_key",
                ):
                    litellm_params = GenericLiteLLMParams()
                    result = self.config.validate_environment(
                        headers=headers, model=self.model, litellm_params=litellm_params
                    )

                    assert "Authorization" in result
                    assert result["Authorization"] == "Bearer env_api_key"

    def test_get_complete_url(self):
        """Test that get_complete_url returns the correct URL"""
        # Test with provided API base
        api_base = "https://custom-openai.example.com/v1"

        result = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={},
        )

        assert result == "https://custom-openai.example.com/v1/responses"

        # Test with litellm.api_base
        with patch("litellm.api_base", "https://litellm-api-base.example.com/v1"):
            result = self.config.get_complete_url(
                api_base=None,
                litellm_params={},
            )

            assert result == "https://litellm-api-base.example.com/v1/responses"

        # Test with environment variable
        with patch("litellm.api_base", None):
            with patch(
                "litellm.llms.openai.responses.transformation.get_secret_str",
                return_value="https://env-api-base.example.com/v1",
            ):
                result = self.config.get_complete_url(
                    api_base=None,
                    litellm_params={},
                )

                assert result == "https://env-api-base.example.com/v1/responses"

        # Test with default API base
        with patch("litellm.api_base", None):
            with patch(
                "litellm.llms.openai.responses.transformation.get_secret_str",
                return_value=None,
            ):
                result = self.config.get_complete_url(
                    api_base=None,
                    litellm_params={},
                )

                assert result == "https://api.openai.com/v1/responses"

        # Test with trailing slash in API base
        api_base = "https://custom-openai.example.com/v1/"

        result = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={},
        )

        assert result == "https://custom-openai.example.com/v1/responses"

    def test_response_id_path_requests_encode_response_id(self):
        """Test response_id is treated as one upstream URL path segment."""
        api_base = "https://custom-openai.example.com/v1/responses"
        response_id = "../../files?x=1#frag"

        url, data = self.config.transform_list_input_items_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert (
            url
            == "https://custom-openai.example.com/v1/responses/..%2F..%2Ffiles%3Fx%3D1%23frag/input_items"
        )
        assert data["limit"] == 20

    def test_get_event_model_class_generic_event(self):
        """Test that get_event_model_class returns the correct event model class"""
        from litellm.types.llms.openai import GenericEvent

        event_type = "test"
        result = self.config.get_event_model_class(event_type)
        assert result == GenericEvent

    def test_transform_streaming_response_generic_event(self):
        """Test that transform_streaming_response returns the correct event model class"""
        from litellm.types.llms.openai import GenericEvent

        chunk = {"type": "test", "test": "test"}
        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )
        assert isinstance(result, GenericEvent)
        assert result.type == "test"

    def test_get_event_model_class_image_generation_partial_image(self):
        """Test that get_event_model_class returns ImageGenerationPartialImageEvent for image generation events"""
        event_type = ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE
        result = self.config.get_event_model_class(event_type)
        assert result == ImageGenerationPartialImageEvent

    def test_transform_streaming_response_image_generation_partial_image(self):
        """Test streaming response transformation for image generation partial image events"""
        # Test with a partial image event - simulating OpenAI's streaming image generation
        chunk = {
            "type": "image_generation.partial_image",
            "partial_image_index": 0,
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 red pixel PNG
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        # Verify the result is the correct event type
        assert isinstance(result, ImageGenerationPartialImageEvent)
        assert result.type == ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE
        assert result.partial_image_index == 0
        assert result.b64_json == chunk["b64_json"]
        assert len(result.b64_json) > 0  # Verify we have image data

    def test_transform_streaming_response_multiple_partial_images(self):
        """Test streaming response with multiple partial images (simulating progressive image generation)"""
        # Test with multiple partial images (as would happen with partial_images=2 or 3)
        test_cases = [
            {
                "type": "image_generation.partial_image",
                "partial_image_index": 0,
                "b64_json": "base64data_partial_0",
            },
            {
                "type": "image_generation.partial_image",
                "partial_image_index": 1,
                "b64_json": "base64data_partial_1",
            },
            {
                "type": "image_generation.partial_image",
                "partial_image_index": 2,
                "b64_json": "base64data_partial_2",
            },
        ]

        for idx, chunk in enumerate(test_cases):
            result = self.config.transform_streaming_response(
                model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
            )

            assert isinstance(result, ImageGenerationPartialImageEvent)
            assert (
                result.type == ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE
            )
            assert result.partial_image_index == idx
            assert result.b64_json == chunk["b64_json"]

    def test_transform_responses_api_request_with_partial_images_param(self):
        """Test request transformation with partial_images parameter for streaming image generation"""
        input_text = "Generate a beautiful landscape"
        optional_params = {
            "temperature": 0.7,
            "stream": True,
            "partial_images": 2,  # Request 2 partial images during generation
        }

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_text,
            response_api_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Validate the result includes partial_images parameter
        expected_fields = {
            "model": self.model,
            "input": input_text,
            "temperature": 0.7,
            "stream": True,
            "partial_images": 2,
        }

        self.validate_responses_api_request_params(result, expected_fields)

    def test_partial_images_parameter_validation(self):
        """Test that partial_images parameter accepts valid values (1-3)"""
        input_text = "Generate an image"

        # Test with different valid partial_images values
        for partial_images_value in [1, 2, 3]:
            optional_params = {
                "stream": True,
                "partial_images": partial_images_value,
            }

            result = self.config.transform_responses_api_request(
                model=self.model,
                input=input_text,
                response_api_optional_request_params=optional_params,
                litellm_params={},
                headers={},
            )

            assert result["partial_images"] == partial_images_value
            assert result["stream"] is True

    def test_transform_streaming_response_coalesces_null_error_code(self):
        """Ensure that when a streaming error event contains error.code=None,
        transform_streaming_response coalesces it to 'unknown_error' and returns
        an ErrorEvent instance without raising a ValidationError.
        """
        from litellm.types.llms.openai import ErrorEvent

        parsed_chunk = {
            "type": "error",
            "sequence_number": 1,
            "error": {
                "type": "invalid_request_error",
                "code": None,
                "message": "Something went wrong",
                "param": None,
            },
        }

        event = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        # Validate returned type and coalesced code
        assert isinstance(event, ErrorEvent)
        assert event.error.code == "unknown_error"
        assert event.error.message == "Something went wrong"

    def test_transform_streaming_response_missing_required_fields_response_created(
        self,
    ):
        """Test that ResponseCreatedEvent with missing required fields (created_at,
        output) does not crash but falls back to model_construct.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import ResponseCreatedEvent

        # Minimal payload an OpenAI-compatible provider might send,
        # omitting `created_at` and `output` inside the response object.
        parsed_chunk = {
            "type": "response.created",
            "response": {
                "id": "resp_q7BOLpck7clq",
                "model": "gpt-oss-120b",
                "status": "in_progress",
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, ResponseCreatedEvent)
        assert result.type == ResponsesAPIStreamEvents.RESPONSE_CREATED
        assert result.response["id"] == "resp_q7BOLpck7clq"

    def test_transform_streaming_response_missing_required_fields_output_text_delta(
        self,
    ):
        """Test that OutputTextDeltaEvent with missing output_index and
        content_index falls back to model_construct without crashing.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import OutputTextDeltaEvent

        # Provider omits output_index and content_index
        parsed_chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_456",
            "delta": "Hello",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        assert result.delta == "Hello"
        assert result.item_id == "item_456"

    def test_transform_streaming_response_missing_required_fields_content_part_added(
        self,
    ):
        """Test that ContentPartAddedEvent with missing output_index and
        content_index falls back to model_construct without crashing.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import ContentPartAddedEvent

        # Provider omits output_index and content_index
        parsed_chunk = {
            "type": "response.content_part.added",
            "item_id": "item_789",
            "part": {"type": "output_text", "text": ""},
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, ContentPartAddedEvent)
        assert result.type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
        assert result.item_id == "item_789"

    def test_transform_streaming_response_missing_required_fields_output_item_added(
        self,
    ):
        """Test that OutputItemAddedEvent with missing output_index falls back
        to model_construct without crashing.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import OutputItemAddedEvent

        # Provider omits output_index
        parsed_chunk = {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_001", "role": "assistant"},
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputItemAddedEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED

    def test_transform_streaming_response_valid_chunk_still_works(self):
        """Ensure that fully valid chunks still go through normal Pydantic
        validation (not model_construct) and work correctly."""
        parsed_chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "World",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.delta == "World"
        assert result.output_index == 0
        assert result.content_index == 0

    def test_base_strip_custom_tool_call_namespace_all_providers(self):
        """Base helper strips ``namespace`` from custom_tool_call for every provider path."""
        inp = [
            {"type": "function_call", "call_id": "a", "name": "f", "namespace": "keep"},
            {
                "type": "custom_tool_call",
                "call_id": "b",
                "name": "c",
                "namespace": "drop",
            },
        ]
        out = BaseResponsesAPIConfig.strip_custom_tool_call_namespace_from_responses_input(
            inp
        )
        assert out[0]["namespace"] == "keep"
        assert "namespace" not in out[1]

        body = {"model": "x", "input": inp}
        norm = BaseResponsesAPIConfig.normalize_responses_api_request_dict(body)
        assert norm["input"][0]["namespace"] == "keep"
        assert "namespace" not in norm["input"][1]

    def test_openai_transform_then_normalize_strips_custom_tool_call_namespace(self):
        """``transform_responses_api_request`` leaves input as validated; HTTP layer ``normalize_*`` strips."""
        input_items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "t",
                "arguments": "{}",
                "namespace": "my_tools",
            },
            {
                "type": "custom_tool_call",
                "call_id": "c2",
                "name": "agent",
                "input": "x",
                "namespace": "None",
                "status": "completed",
            },
        ]
        body = self.config.transform_responses_api_request(
            model=self.model,
            input=input_items,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"][0].get("namespace") == "my_tools"
        assert body["input"][1].get("namespace") == "None"

        norm = BaseResponsesAPIConfig.normalize_responses_api_request_dict(body)
        assert norm["input"][0].get("namespace") == "my_tools"
        assert norm["input"][1]["type"] == "custom_tool_call"
        assert "namespace" not in norm["input"][1]

    @staticmethod
    def _claude_turn_bridged_to_responses_output() -> list:
        claude_turn = ModelResponse(
            id="chatcmpl-claude",
            model="claude-sonnet-4-5",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Paris is 22C and sunny.",
                        reasoning_content="Check Paris first.",
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "Check Paris first.", "signature": "sig-paris"}
                        ],
                    ),
                )
            ],
        )
        bridged = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Weather in Paris?", responses_api_request={}, chat_completion_response=claude_turn
        )
        return list(bridged.output)

    @pytest.mark.parametrize("config", [OpenAIResponsesAPIConfig(), AzureOpenAIResponsesAPIConfig()])
    def test_claude_reasoning_minted_by_the_bridge_is_dropped_before_the_history_reaches_openai(self, config):
        saved_claude_turn = json.loads(
            json.dumps([item.model_dump() for item in self._claude_turn_bridged_to_responses_output()])
        )
        bridge_reasoning = [item for item in saved_claude_turn if item["type"] == "reasoning"]
        assert len(bridge_reasoning) == 1
        openai_reasoning = {
            "id": "rs_08d3a89dbb92277a006abf04f4266087d0b4eedacd7848f306",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "gAAAAABo-opaque-openai-blob",
        }
        history = [
            {"role": "user", "content": "Weather in Paris?"},
            *saved_claude_turn,
            openai_reasoning,
            {"role": "user", "content": "And Berlin?"},
        ]

        request = config.transform_responses_api_request(
            model="gpt-5.6",
            input=history,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        outbound = request["input"]
        assert len(outbound) == len(history) - 1
        assert [item["id"] for item in outbound if item.get("type") == "reasoning"] == [openai_reasoning["id"]]
        assert LiteLLMCompletionResponsesConfig._decode_thinking_blocks_from_input_item(bridge_reasoning[0]) == (
            {"type": "thinking", "thinking": "Check Paris first.", "signature": "sig-paris"},
        )

    def test_bridge_minted_reasoning_is_dropped_when_handed_back_as_pydantic_output_items(self):
        history = [*self._claude_turn_bridged_to_responses_output(), {"role": "user", "content": "And Berlin?"}]

        request = self.config.transform_responses_api_request(
            model="gpt-5.6",
            input=history,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert len(request["input"]) == len(history) - 1
        assert all(item.get("type") != "reasoning" for item in request["input"])

    @pytest.mark.parametrize(
        ("reasoning_item", "expected"),
        [
            (
                {"id": "rs_1", "type": "reasoning", "summary": [], "status": None, "note": None},
                {"id": "rs_1", "type": "reasoning", "summary": []},
            ),
            (
                {"id": "rs_1", "type": "reasoning", "summary": "not a list", "status": None, "note": None},
                {"id": "rs_1", "type": "reasoning", "summary": "not a list", "note": None},
            ),
        ],
    )
    def test_a_reasoning_input_item_loses_its_null_status_whether_or_not_it_fits_the_openai_model(
        self, reasoning_item: dict[str, object], expected: dict[str, object]
    ):
        request: Final = self.config.transform_responses_api_request(
            model="gpt-5.6",
            input=[reasoning_item, {"role": "user", "content": "And Berlin?"}],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [expected, {"role": "user", "content": "And Berlin?"}]


class TestAzureResponsesAPIConfig:
    def setup_method(self):
        self.config = AzureOpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_azure_decodes_json_string_tool_parameters(self):
        """Azure reaches the same wire through `super()`, after un-nesting a chat-shaped tool."""
        result = self.config.transform_responses_api_request(
            model=self.model,
            input="weather in Paris",
            response_api_optional_request_params={
                "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": '{"type":"object"}'}}]
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0]["parameters"] == {"type": "object"}

    def test_azure_get_complete_url_with_version_types(self):
        """Test Azure get_complete_url with different API version types"""
        base_url = "https://litellm8397336933.openai.azure.com"

        # Test with preview version - should use openai/v1/responses
        result_preview = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "preview"},
        )
        assert (
            result_preview
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=preview"
        )

        # Test with latest version - should use openai/v1/responses
        result_latest = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "latest"},
        )
        assert (
            result_latest
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=latest"
        )

        # Test with date-based version - should use openai/responses
        result_date = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "2025-01-01"},
        )
        assert (
            result_date
            == "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2025-01-01"
        )

    def test_azure_transform_then_normalize_strips_custom_tool_call_namespace(self):
        """Same as OpenAI path: ``normalize_responses_api_request_dict`` strips custom_tool_call only."""
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hi"}],
            },
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "input": "do thing",
                "name": "my_tool",
                "id": "ctc_1",
                "namespace": "None",
                "status": "completed",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "get_weather",
                "arguments": "{}",
                "id": "fc_1",
                "namespace": "tools",
                "status": "completed",
            },
        ]
        body = self.config.transform_responses_api_request(
            model=self.model,
            input=input_items,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"][1].get("namespace") == "None"
        assert body["input"][2].get("namespace") == "tools"

        norm = BaseResponsesAPIConfig.normalize_responses_api_request_dict(body)
        assert norm["input"][1]["type"] == "custom_tool_call"
        assert "namespace" not in norm["input"][1]
        assert norm["input"][2]["type"] == "function_call"
        assert norm["input"][2].get("namespace") == "tools"
        assert norm["input"][2]["name"] == "get_weather"


class TestTransformListInputItemsRequest:
    """Test suite for transform_list_input_items_request function"""

    def setup_method(self):
        """Setup test fixtures"""
        self.openai_config = OpenAIResponsesAPIConfig()
        self.azure_config = AzureOpenAIResponsesAPIConfig()
        self.response_id = "resp_abc123"
        self.api_base = "https://api.openai.com/v1/responses"
        self.litellm_params = GenericLiteLLMParams()
        self.headers = {"Authorization": "Bearer test-key"}

    def test_openai_transform_list_input_items_request_minimal(self):
        """Test OpenAI implementation with minimal parameters"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        expected_url = f"{self.api_base}/{self.response_id}/input_items"
        assert url == expected_url
        assert params == {"limit": 20, "order": "desc"}

    def test_openai_transform_list_input_items_request_all_params(self):
        """Test OpenAI implementation with all optional parameters"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            after="cursor_after_123",
            before="cursor_before_456",
            include=["metadata", "content"],
            limit=50,
            order="asc",
        )

        # Assert
        expected_url = f"{self.api_base}/{self.response_id}/input_items"
        expected_params = {
            "after": "cursor_after_123",
            "before": "cursor_before_456",
            "include": "metadata,content",  # Should be comma-separated string
            "limit": 50,
            "order": "asc",
        }
        assert url == expected_url
        assert params == expected_params

    def test_openai_transform_list_input_items_request_include_list_formatting(self):
        """Test that include list is properly formatted as comma-separated string"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            include=["metadata", "content", "annotations"],
        )

        # Assert
        assert params["include"] == "metadata,content,annotations"

    def test_openai_transform_list_input_items_request_none_values(self):
        """Test OpenAI implementation with None values for optional parameters"""
        # Execute - pass only required parameters and explicit None for truly optional params
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            after=None,
            before=None,
            include=None,
        )

        # Assert
        expected_url = f"{self.api_base}/{self.response_id}/input_items"
        expected_params = {
            "limit": 20,
            "order": "desc",
        }  # Default values should be present
        assert url == expected_url
        assert params == expected_params

    def test_openai_transform_list_input_items_request_empty_include_list(self):
        """Test OpenAI implementation with empty include list"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            include=[],
        )

        # Assert
        assert "include" not in params  # Empty list should not be included

    def test_openai_transform_compact_response_api_request_query_params_preserved(self):
        """Test compact URL construction preserves query params and appends path."""
        # Setup
        azure_style_api_base = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, data = self.openai_config.transform_compact_response_api_request(
            model="gpt-5.2-codex",
            input="hello",
            response_api_optional_request_params={},
            api_base=azure_style_api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        assert (
            url
            == "https://test.openai.azure.com/openai/responses/compact?api-version=2024-05-01-preview"
        )
        assert data["model"] == "gpt-5.2-codex"
        assert data["input"] == "hello"

    def test_openai_transform_compact_response_api_request_path_without_query(self):
        """Test compact URL construction for base URL without query params."""
        # Execute
        url, data = self.openai_config.transform_compact_response_api_request(
            model="gpt-4o",
            input="hello",
            response_api_optional_request_params={},
            api_base="https://api.openai.com/v1/responses",
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        assert url == "https://api.openai.com/v1/responses/compact"
        assert data["model"] == "gpt-4o"
        assert data["input"] == "hello"

    def test_azure_transform_list_input_items_request_minimal(self):
        """Test Azure implementation with minimal parameters"""
        # Setup
        AZURE_AI_API_BASE = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, params = self.azure_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=AZURE_AI_API_BASE,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        assert self.response_id in url
        assert "/input_items" in url
        assert params == {"limit": 20, "order": "desc"}

    def test_azure_transform_list_input_items_request_url_construction(self):
        """Test Azure implementation URL construction with response_id in path"""
        # Setup
        AZURE_AI_API_BASE = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, params = self.azure_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=AZURE_AI_API_BASE,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        # The Azure implementation should construct URL with response_id in path
        assert self.response_id in url
        assert "/input_items" in url
        assert "api-version=2024-05-01-preview" in url

    def test_azure_transform_list_input_items_request_with_all_params(self):
        """Test Azure implementation with all optional parameters"""
        # Setup
        AZURE_AI_API_BASE = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, params = self.azure_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=AZURE_AI_API_BASE,
            litellm_params=self.litellm_params,
            headers=self.headers,
            after="cursor_after_123",
            before="cursor_before_456",
            include=["metadata", "content"],
            limit=100,
            order="asc",
        )

        # Assert
        expected_params = {
            "after": "cursor_after_123",
            "before": "cursor_before_456",
            "include": "metadata,content",
            "limit": 100,
            "order": "asc",
        }
        assert params == expected_params

    @patch("litellm.router.Router")
    def test_mock_litellm_router_with_transform_list_input_items_request(
        self, mock_router
    ):
        """Mock test using litellm.router for transform_list_input_items_request"""
        # Setup mock router
        mock_router_instance = Mock()
        mock_router.return_value = mock_router_instance

        # Mock the provider config
        mock_provider_config = Mock(spec=OpenAIResponsesAPIConfig)
        mock_provider_config.transform_list_input_items_request.return_value = (
            "https://api.openai.com/v1/responses/resp_123/input_items",
            {"limit": 20, "order": "desc"},
        )

        # Setup router mock
        mock_router_instance.get_provider_responses_api_config.return_value = (
            mock_provider_config
        )

        # Test parameters
        response_id = "resp_test123"

        # Execute
        url, params = mock_provider_config.transform_list_input_items_request(
            response_id=response_id,
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={"Authorization": "Bearer test"},
            after="cursor_123",
            include=["metadata"],
            limit=30,
        )

        # Assert
        mock_provider_config.transform_list_input_items_request.assert_called_once_with(
            response_id=response_id,
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={"Authorization": "Bearer test"},
            after="cursor_123",
            include=["metadata"],
            limit=30,
        )
        assert url == "https://api.openai.com/v1/responses/resp_123/input_items"
        assert params == {"limit": 20, "order": "desc"}

    @patch("litellm.list_input_items")
    def test_mock_litellm_list_input_items_integration(self, mock_list_input_items):
        """Test integration with litellm.list_input_items function"""
        # Setup mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "input_item_123",
                    "object": "input_item",
                    "type": "message",
                    "role": "user",
                    "content": "Test message",
                }
            ],
            "has_more": False,
            "first_id": "input_item_123",
            "last_id": "input_item_123",
        }
        mock_list_input_items.return_value = mock_response

        # Execute
        result = mock_list_input_items(
            response_id="resp_test123",
            after="cursor_after",
            limit=10,
            custom_llm_provider="openai",
        )

        # Assert
        mock_list_input_items.assert_called_once_with(
            response_id="resp_test123",
            after="cursor_after",
            limit=10,
            custom_llm_provider="openai",
        )
        assert result["object"] == "list"
        assert len(result["data"]) == 1

    def test_parameter_validation_edge_cases(self):
        """Test edge cases for parameter validation"""
        # Test with limit=0
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            limit=0,
        )
        assert params["limit"] == 0

        # Test with very large limit
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            limit=1000,
        )
        assert params["limit"] == 1000

        # Test with single item in include list
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            include=["metadata"],
        )
        assert params["include"] == "metadata"

    def test_url_construction_with_different_api_bases(self):
        """Test URL construction with different API base formats"""
        test_cases = [
            {
                "api_base": "https://api.openai.com/v1/responses",
                "expected_suffix": "/resp_abc123/input_items",
            },
            {
                "api_base": "https://api.openai.com/v1/responses/",  # with trailing slash
                "expected_suffix": "/resp_abc123/input_items",
            },
            {
                "api_base": "https://custom-api.example.com/v1/responses",
                "expected_suffix": "/resp_abc123/input_items",
            },
        ]

        for case in test_cases:
            url, params = self.openai_config.transform_list_input_items_request(
                response_id=self.response_id,
                api_base=case["api_base"],
                litellm_params=self.litellm_params,
                headers=self.headers,
            )
            assert url.endswith(case["expected_suffix"])

    def test_return_type_validation(self):
        """Test that function returns correct types"""
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert return types
        assert isinstance(url, str)
        assert isinstance(params, dict)

        # Assert URL is properly formatted
        assert url.startswith("http")
        assert "input_items" in url

        # Assert params contains expected keys with correct types
        for key, value in params.items():
            assert isinstance(key, str)
            assert value is not None


def test_get_supported_openai_params():
    config = OpenAIResponsesAPIConfig()
    params = config.get_supported_openai_params("gpt-4o")
    assert "temperature" in params
    assert "stream" in params
    assert "background" in params
    assert "stream" in params


class TestPhaseParameter:
    """Tests for the `phase` parameter on assistant output items (gpt-5.3-codex)."""

    def setup_method(self):
        self.config = OpenAIResponsesAPIConfig()
        self.model = "gpt-5.3-codex"
        self.logging_obj = MagicMock()

    @staticmethod
    def _make_output_text(text: str):
        from litellm.types.responses.main import OutputText

        return OutputText(type="output_text", text=text, annotations=[])

    def test_generic_response_output_item_accepts_phase_commentary(self):
        from litellm.types.responses.main import GenericResponseOutputItem

        item = GenericResponseOutputItem(
            type="message",
            id="msg_001",
            status="completed",
            role="assistant",
            content=[self._make_output_text("Thinking...")],
            phase="commentary",
        )
        assert item.phase == "commentary"

    def test_generic_response_output_item_accepts_phase_final_answer(self):
        from litellm.types.responses.main import GenericResponseOutputItem

        item = GenericResponseOutputItem(
            type="message",
            id="msg_002",
            status="completed",
            role="assistant",
            content=[self._make_output_text("The answer is 42.")],
            phase="final_answer",
        )
        assert item.phase == "final_answer"

    def test_generic_response_output_item_phase_defaults_to_none(self):
        from litellm.types.responses.main import GenericResponseOutputItem

        item = GenericResponseOutputItem(
            type="message",
            id="msg_003",
            status="completed",
            role="assistant",
            content=[self._make_output_text("Hello")],
        )
        assert item.phase is None

    def test_output_function_tool_call_accepts_phase(self):
        from litellm.types.responses.main import OutputFunctionToolCall

        item = OutputFunctionToolCall(
            type="function_call",
            id="fc_001",
            arguments='{"query": "test"}',
            call_id="call_001",
            name="search",
            status="completed",
            phase="commentary",
        )
        assert item.phase == "commentary"

    def test_input_passthrough_dict_preserves_phase(self):
        """Dict input items (the normal HTTP flow) must preserve phase verbatim."""
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hi"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Preamble..."}],
                "phase": "commentary",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
                "phase": "final_answer",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Neutral."}],
                "phase": None,
            },
        ]

        result = self.config._validate_input_param(input_items)
        assert isinstance(result, list)

        assert "phase" not in result[0]
        assert result[1]["phase"] == "commentary"
        assert result[2]["phase"] == "final_answer"
        assert result[3]["phase"] is None

    def test_input_passthrough_pydantic_preserves_non_null_phase(self):
        """Pydantic input items must preserve non-null phase values."""
        from litellm.types.responses.main import GenericResponseOutputItem

        item = GenericResponseOutputItem(
            type="message",
            id="msg_010",
            status="completed",
            role="assistant",
            content=[self._make_output_text("commentary")],
            phase="commentary",
        )

        result = self.config._validate_input_param([item])
        assert isinstance(result, list)
        assert result[0]["phase"] == "commentary"

    def test_response_parsing_preserves_phase_on_output(self):
        """Non-streaming response must preserve phase on output items."""
        raw_json = {
            "id": "resp_001",
            "created_at": 1700000000,
            "model": "gpt-5.3-codex",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_001",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "preamble"}],
                    "phase": "commentary",
                },
                {
                    "type": "message",
                    "id": "msg_002",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                    "phase": "final_answer",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }

        response = ResponsesAPIResponse(**raw_json)
        assert len(response.output) == 2

        for idx, output_item in enumerate(response.output):
            if isinstance(output_item, dict):
                phase = output_item.get("phase")
            else:
                phase = getattr(output_item, "phase", None)

            expected = "commentary" if idx == 0 else "final_answer"
            assert (
                phase == expected
            ), f"output[{idx}] phase={phase!r}, expected {expected!r}"

    def test_streaming_output_item_done_preserves_phase(self):
        """OutputItemDoneEvent must preserve phase on its item."""
        from litellm.types.llms.openai import (
            OutputItemDoneEvent,
            ResponsesAPIStreamEvents,
        )

        chunk = {
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "message",
                "id": "msg_100",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
                "phase": "final_answer",
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputItemDoneEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        assert getattr(result.item, "phase", None) == "final_answer"

    def test_streaming_output_item_added_preserves_phase(self):
        """OutputItemAddedEvent must preserve phase on its item."""
        from litellm.types.llms.openai import (
            OutputItemAddedEvent,
            ResponsesAPIStreamEvents,
        )

        chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_200",
                "role": "assistant",
                "phase": "commentary",
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputItemAddedEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert getattr(result.item, "phase", None) == "commentary"

    def test_streaming_response_completed_preserves_phase(self):
        """ResponseCompletedEvent must preserve phase on output items inside the response."""
        completed_chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_300",
                "created_at": 1700000000,
                "model": "gpt-5.3-codex",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_300",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "final"}],
                        "phase": "final_answer",
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 10,
                    "total_tokens": 15,
                },
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model,
            parsed_chunk=completed_chunk,
            logging_obj=self.logging_obj,
        )

        assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        output_item = result.response.output[0]
        if isinstance(output_item, dict):
            assert output_item["phase"] == "final_answer"
        else:
            assert getattr(output_item, "phase", None) == "final_answer"

    def test_phase_roundtrip_output_to_input(self):
        """Simulate full round-trip: parse response output, then send items back as input."""
        raw_json = {
            "id": "resp_rt",
            "created_at": 1700000000,
            "model": "gpt-5.3-codex",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_rt1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "preamble"}],
                    "phase": "commentary",
                },
                {
                    "type": "message",
                    "id": "msg_rt2",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                    "phase": "final_answer",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }

        response = ResponsesAPIResponse(**raw_json)

        input_items = []
        for item in response.output:
            if isinstance(item, dict):
                input_items.append(item)
            else:
                input_items.append(
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                )

        input_items.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "next question"}],
            }
        )

        validated = self.config._validate_input_param(input_items)
        assert isinstance(validated, list)

        assert validated[0]["phase"] == "commentary"
        assert validated[1]["phase"] == "final_answer"
        assert "phase" not in validated[2]


class TestPromptCacheOptionsOnResponsesPath:
    """`prompt_cache_options` and block-level `prompt_cache_breakpoint` survive the Responses transformation (#37509)."""

    OPTIONS = {"mode": "explicit", "ttl": "30m"}

    def test_prompt_cache_options_survives_optional_param_filter(self):
        from litellm.responses.utils import ResponsesAPIRequestUtils

        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            {"prompt_cache_options": dict(self.OPTIONS), "temperature": 0.2, "not_a_responses_param": 1}
        )
        assert result["prompt_cache_options"] == self.OPTIONS
        assert result["temperature"] == 0.2
        assert "not_a_responses_param" not in result

    def test_prompt_cache_options_reaches_transformed_request(self):
        config = OpenAIResponsesAPIConfig()
        mapped = config.map_openai_params(
            response_api_optional_params={"prompt_cache_options": dict(self.OPTIONS)},
            model="gpt-5.6",
            drop_params=False,
        )
        result = config.transform_responses_api_request(
            model="gpt-5.6",
            input="hi",
            response_api_optional_request_params=mapped,
            litellm_params={},
            headers={},
        )
        assert result["prompt_cache_options"] == self.OPTIONS

    def test_prompt_cache_breakpoint_survives_cache_control_strip(self):
        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-5.6",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hi",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            response_api_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert result["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "hi",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }


class TestResponsesSurfaceSharesTheEffortRule:
    """The Responses API reaches the same gpt-5 models over a different wire, and the default
    /v1/messages bridge for openai models routes through it. It carried its own copy of the
    temperature rule, so fixing chat completions alone left this surface still forwarding
    temperature to a model that rejects it.
    """

    @pytest.mark.parametrize(
        "model, effort, temperature_survives",
        [
            ("gpt-5.1", None, True),
            ("gpt-5.4", None, True),
            ("gpt-5.5", None, False),
            ("gpt-5.6-terra", None, False),
            ("gpt-5.6-sol", None, False),
            ("gpt-5.6-terra", "none", True),
            ("gpt-5.6-terra", "medium", False),
            ("gpt-6-astra", None, False),
            ("gpt-6-astra", "low", False),
        ],
    )
    def test_temperature_follows_the_resolved_effort(
        self, local_model_cost_map, model, effort, temperature_survives
    ):
        params = {"temperature": 0}
        if effort is not None:
            params["reasoning"] = {"effort": effort}
        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=params,
            model=model,
            drop_params=True,
        )
        assert ("temperature" in mapped) is temperature_survives

    @pytest.mark.parametrize(
        "model, effort, top_p_survives",
        [
            ("gpt-5.1", None, True),
            ("gpt-5.4", None, True),
            ("gpt-5.5", None, False),
            ("gpt-5.6-terra", None, False),
            ("gpt-5.6-sol", None, False),
            ("gpt-5.6-terra", "none", True),
            ("gpt-5.6-terra", "medium", False),
            ("gpt-6-astra", None, False),
            ("gpt-6-astra", "low", False),
        ],
    )
    def test_top_p_follows_the_resolved_effort(self, local_model_cost_map, model, effort, top_p_survives):
        params = {"top_p": 0.9}
        if effort is not None:
            params["reasoning"] = {"effort": effort}
        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=params,
            model=model,
            drop_params=True,
        )
        assert ("top_p" in mapped) is top_p_survives

    def test_top_p_raises_without_drop_params(self, local_model_cost_map):
        with pytest.raises(litellm.UnsupportedParamsError):
            OpenAIResponsesAPIConfig().map_openai_params(
                response_api_optional_params={"top_p": 0.9},
                model="gpt-5.5",
                drop_params=False,
            )

        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params={"top_p": 0.9, "reasoning": {"effort": "none"}},
            model="gpt-5.6-terra",
            drop_params=False,
        )
        assert mapped["top_p"] == 0.9


class TestFlattenToolSchemaCombinatorsWiring:
    """Regression tests for MCP tools with a top-level anyOf schema (Codex Desktop).

    OpenAI's /v1/responses rejects function tool parameters carrying
    'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level, while the
    ChatGPT backend Codex uses natively accepts them, so those tools 400'd
    through the proxy with "Invalid schema for function ...".
    """

    def _anyof_parameters(self):
        return {
            "type": "object",
            "anyOf": [
                {
                    "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                    "required": ["id", "enabled"],
                },
                {
                    "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                    "required": ["id", "schedule"],
                },
            ],
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }

    def _flat_function_tool(self):
        return {
            "type": "function",
            "name": "mcp__codex_app__automation_update",
            "description": "Update an automation",
            "parameters": self._anyof_parameters(),
            "strict": False,
        }

    def _codex_namespace_tool(self):
        return {
            "type": "namespace",
            "name": "mcp__codex_app",
            "tools": [
                {
                    "name": "automation_update",
                    "description": "Update an automation",
                    "parameters": self._anyof_parameters(),
                    "strict": False,
                }
            ],
        }

    def test_openai_flattens_top_level_anyof_on_flat_function_tool(self):
        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [self._flat_function_tool()]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        parameters = result["tools"][0]["parameters"]
        assert "anyOf" not in parameters
        assert parameters["type"] == "object"
        assert set(parameters["properties"]) == {"id", "enabled", "schedule"}
        assert parameters["required"] == ["id"]
        assert json.loads(json.dumps(result["tools"])) == result["tools"]

    def test_openai_flattens_anyof_inside_codex_namespace_tools(self):
        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [self._codex_namespace_tool()]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        nested_parameters = result["tools"][0]["tools"][0]["parameters"]
        assert "anyOf" not in nested_parameters
        assert set(nested_parameters["properties"]) == {"id", "enabled", "schedule"}
        assert json.loads(json.dumps(result["tools"])) == result["tools"]

    def test_openai_compact_request_flattens_top_level_anyof(self):
        _, data = OpenAIResponsesAPIConfig().transform_compact_response_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [self._flat_function_tool()]},
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "anyOf" not in data["tools"][0]["parameters"]

    def test_openai_leaves_tools_without_rejected_keys_alone(self):
        clean_tool = {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }

        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [clean_tool]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0]["parameters"] == {"type": "object", "properties": {"city": {"type": "string"}}}

    def test_openai_does_not_mutate_caller_tool_dicts(self):
        tool = self._flat_function_tool()

        OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "anyOf" in tool["parameters"]

    def test_non_openai_subclass_does_not_flatten(self):
        from litellm.llms.hosted_vllm.responses.transformation import HostedVLLMResponsesAPIConfig

        result = HostedVLLMResponsesAPIConfig().transform_responses_api_request(
            model="hosted_vllm/qwen",
            input="hi",
            response_api_optional_request_params={"tools": [self._flat_function_tool()]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "anyOf" in result["tools"][0]["parameters"]

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4-turbo",
            "o1",
            "o3-pro",
            "o4-mini",
            "openai/gpt-4o",
            "ft:gpt-4o-2024-08-06:org::abc",
        ],
    )
    def test_openai_flattens_for_models_whose_validator_rejects_combinators(self, model):
        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model=model,
            input="hi",
            response_api_optional_request_params={"tools": [self._flat_function_tool()]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "anyOf" not in result["tools"][0]["parameters"]

    @pytest.mark.parametrize(
        "model", ["gpt-5", "gpt-5-nano", "gpt-5.4-mini", "gpt-5.4-codex", "gpt-5.5", "openai/gpt-5.2"]
    )
    def test_openai_keeps_combinators_for_models_that_accept_them(self, model):
        tool = self._flat_function_tool()

        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model=model,
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0] is tool

    def test_openai_leaves_non_dict_tool_entries_alone(self):
        opaque_tool = SimpleNamespace(type="function", name="automation_update")

        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [opaque_tool, self._flat_function_tool()]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0] is opaque_tool
        assert "anyOf" not in result["tools"][1]["parameters"]


class TestToolSchemaRegexPatternWiring:
    """Claude Code's Artifact tool reaches /v1/responses (the /v1/messages bridge) with an
    ECMA-262 ``pattern``; OpenAI compiles patterns with Python ``re`` and 400s
    "'...' is not a 'regex'" for every model family, so the keyword is dropped.
    """

    def _artifact_tool(self):
        return {
            "type": "function",
            "name": "Artifact",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "pattern": _ARTIFACT_FIELD_PATTERN},
                    "doc_id": {"type": "string", "pattern": r"^(?!\.\.?(?:/|$))[A-Za-z0-9_\-.~:@+]{1,200}$"},
                },
                "required": ["field"],
            },
        }

    @pytest.mark.parametrize("model", ["gpt-5.6", "gpt-4o", "o3"])
    def test_openai_drops_only_the_pattern_python_re_rejects_for_every_family(self, model):
        tool = self._artifact_tool()

        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model=model,
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        properties = result["tools"][0]["parameters"]["properties"]
        assert properties["field"] == {"type": "string"}
        assert properties["doc_id"] == tool["parameters"]["properties"]["doc_id"]
        assert result["tools"][0]["parameters"]["required"] == ["field"]
        assert tool["parameters"]["properties"]["field"]["pattern"] == _ARTIFACT_FIELD_PATTERN
        assert json.loads(json.dumps(result["tools"])) == result["tools"]

    def test_openai_drops_patterns_inside_codex_namespace_tools(self):
        namespace = {"type": "namespace", "name": "mcp__claude", "tools": [self._artifact_tool()]}

        result = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-5.6",
            input="hi",
            response_api_optional_request_params={"tools": [namespace]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0]["tools"][0]["parameters"]["properties"]["field"] == {"type": "string"}

    def test_openai_compact_request_drops_patterns(self):
        _, data = OpenAIResponsesAPIConfig().transform_compact_response_api_request(
            model="gpt-5.6",
            input="hi",
            response_api_optional_request_params={"tools": [self._artifact_tool()]},
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["tools"][0]["parameters"]["properties"]["field"] == {"type": "string"}

    def test_non_openai_subclass_keeps_patterns(self):
        from litellm.llms.hosted_vllm.responses.transformation import HostedVLLMResponsesAPIConfig

        tool = self._artifact_tool()

        result = HostedVLLMResponsesAPIConfig().transform_responses_api_request(
            model="hosted_vllm/qwen",
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0] is tool


class TestReasoningFollowsModelSupport:
    """Responses API clients like Codex send `reasoning` on every request, and OpenAI 400s it
    on non-reasoning models like gpt-4o. drop_params must strip it there, the same way the
    chat completions surface already strips reasoning_effort for those models.
    """

    @pytest.mark.parametrize(
        "model, reasoning_survives",
        [
            ("gpt-4o", False),
            ("gpt-4.1", False),
            ("gpt-4o-mini", False),
            ("ft:gpt-4o-2024-08-06:my-org::abc123", False),
            ("chat-latest", True),
            ("gpt-5.6", True),
            ("o3", True),
            ("o3-deep-research", True),
            ("o4-mini-deep-research", True),
            ("codex-mini-latest", True),
            ("ft:o4-mini-2025-04-16:my-org::abc123", True),
            ("computer-use-preview", True),
        ],
    )
    def test_drop_params_strips_reasoning_by_model(self, local_model_cost_map, model, reasoning_survives):
        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params={"reasoning": {"effort": "medium", "summary": "auto"}},
            model=model,
            drop_params=True,
        )
        assert ("reasoning" in mapped) is reasoning_survives

    @pytest.mark.parametrize(
        "model, reasoning_survives",
        [
            ("gpt-4o", False),
            ("chat-latest", True),
            ("o3", True),
            ("o3-deep-research", True),
        ],
    )
    def test_a_cost_map_older_than_this_release_never_strips_a_known_reasoning_model(
        self, local_model_cost_map, monkeypatch, model, reasoning_survives
    ):
        lagging = {
            name: {field: value for field, value in entry.items() if field != "supports_reasoning"}
            for name, entry in litellm.model_cost.items()
        }
        monkeypatch.setattr(litellm, "model_cost", lagging)
        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params={"reasoning": {"effort": "medium"}},
            model=model,
            drop_params=True,
        )
        assert ("reasoning" in mapped) is reasoning_survives

    def test_without_drop_params_the_error_is_litellms_400(self, local_model_cost_map, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        with pytest.raises(litellm.UnsupportedParamsError) as excinfo:
            OpenAIResponsesAPIConfig().map_openai_params(
                response_api_optional_params={"reasoning": {"effort": "medium"}},
                model="gpt-4o",
                drop_params=False,
            )
        assert excinfo.value.status_code == 400
        assert "reasoning.effort" in str(excinfo.value)
        assert "cost map" in str(excinfo.value)

    @pytest.mark.parametrize("drop_params", [True, False])
    @pytest.mark.parametrize(
        "reasoning",
        [{"summary": "auto"}, {"effort": None, "summary": "auto"}, {}],
    )
    def test_reasoning_without_an_effort_passes_through_on_non_reasoning_models(
        self, local_model_cost_map, monkeypatch, drop_params, reasoning
    ):
        monkeypatch.setattr(litellm, "drop_params", drop_params)
        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params={"reasoning": dict(reasoning)},
            model="gpt-4o",
            drop_params=drop_params,
        )
        assert mapped["reasoning"] == reasoning

    def test_an_explicit_supports_reasoning_false_beats_the_bundled_floor(self, local_model_cost_map, monkeypatch):
        overridden = {
            name: ({**entry, "supports_reasoning": False} if name == "o3" else entry)
            for name, entry in litellm.model_cost.items()
        }
        monkeypatch.setattr(litellm, "model_cost", overridden)
        mapped = OpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params={"reasoning": {"effort": "medium"}},
            model="o3",
            drop_params=True,
        )
        assert "reasoning" not in mapped

    def test_azure_deployments_keep_reasoning_even_on_a_non_reasoning_model_name(self, local_model_cost_map):
        mapped = AzureOpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params={"reasoning": {"effort": "medium"}},
            model="gpt-4o",
            drop_params=True,
        )
        assert mapped["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_openai_responses_litellm_router_no_metadata():
    """
    Test that metadata is not passed through when using the Router for responses API
    """
    mock_response = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "gpt-5.5",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello world!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        # Adding all required fields
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)
            self.headers = httpx.Headers({})

        def json(self):  # Changed from async to sync
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt4o-special-alias",
                    "litellm_params": {
                        "model": "gpt-5.5",
                        "api_key": "fake-key",
                    },
                }
            ]
        )

        # Call the handler with metadata
        await router.aresponses(
            model="gpt4o-special-alias",
            input="Hello, can you tell me a short joke?",
        )

        # Check the request body
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4))

        # Assert metadata is not in the request
        assert (
            "metadata" not in request_body
        ), "metadata should not be in the request body"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_openai_responses_litellm_router_with_metadata():
    """
    Test that metadata is correctly passed through when explicitly provided to the Router for responses API
    """
    test_metadata = {
        "user_id": "123",
        "conversation_id": "abc",
        "custom_field": "test_value",
    }

    mock_response = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "gpt-5.5",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello world!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": test_metadata,  # Include the test metadata in response
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt4o-special-alias",
                    "litellm_params": {
                        "model": "gpt-5.5",
                        "api_key": "fake-key",
                    },
                }
            ]
        )

        # Call the handler with metadata
        await router.aresponses(
            model="gpt4o-special-alias",
            input="Hello, can you tell me a short joke?",
            metadata=test_metadata,
        )

        # Check the request body
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4))

        # Assert metadata matches exactly what was passed
        assert (
            request_body["metadata"] == test_metadata
        ), "metadata in request body should match what was passed"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_openai_responses_litellm_router_with_prompt():
    """Test that prompt object is passed through the Router for responses API"""

    prompt_obj = {
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {"random_variable": "ishaan_from_litellm"},
    }

    mock_response = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "gpt-5.5",
        "output": [],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt4o-special-alias",
                    "litellm_params": {
                        "model": "gpt-5.5",
                        "api_key": "fake-key",
                    },
                }
            ]
        )

        await router.aresponses(
            model="gpt4o-special-alias",
            input="Hello",
            prompt=prompt_obj,
        )

        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["prompt"] == prompt_obj
        mock_post.assert_called_once()


def test_bad_request_bad_param_error():
    """Raise a BadRequestError when an invalid parameter value is provided"""
    try:
        litellm.responses(model="gpt-5.5", input="This should fail", temperature=2000)
        pytest.fail("Expected BadRequestError but no exception was raised")
    except litellm.BadRequestError as e:
        print(f"Exception raised: {e}")
        print(f"Exception type: {type(e)}")
        print(f"Exception args: {e.args}")
        print(f"Exception details: {e.__dict__}")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.asyncio()
async def test_async_bad_request_bad_param_error():
    """Raise a BadRequestError when an invalid parameter value is provided"""
    try:
        await litellm.aresponses(
            model="gpt-5.5", input="This should fail", temperature=2000
        )
        pytest.fail("Expected BadRequestError but no exception was raised")
    except litellm.BadRequestError as e:
        print(f"Exception raised: {e}")
        print(f"Exception type: {type(e)}")
        print(f"Exception args: {e.args}")
        print(f"Exception details: {e.__dict__}")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_o1_pro_response_api(sync_mode):
    """
    Test that LiteLLM correctly handles an incomplete response from OpenAI's o1-pro model
    due to reaching max_output_tokens limit.
    """
    # Mock response from o1-pro
    mock_response = {
        "id": "resp_67dc3dd77b388190822443a85252da5a0e13d8bdc0e28d88",
        "object": "response",
        "created_at": 1742486999,
        "status": "incomplete",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": None,
        "max_output_tokens": 20,
        "model": "o1-pro-2025-03-19",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_67dc3de50f64819097450ed50a33d5f90e13d8bdc0e28d88",
                "summary": [],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 73,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 93,
        },
        "user": None,
        "metadata": {},
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = httpx.Headers({})

        def json(self):  # Changed from async to sync
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        litellm.set_verbose = True

        # Call o1-pro with max_output_tokens=20
        response = await litellm.aresponses(
            model="openai/o1-pro",
            input="Write a detailed essay about artificial intelligence and its impact on society",
            max_output_tokens=20,
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["model"] == "o1-pro"
        assert request_body["max_output_tokens"] == 20

        # Validate the response
        print("Response:", json.dumps(response, indent=4, default=str))

        # Check that the response has the expected structure
        assert response["id"] is not None
        assert response["status"] == "incomplete"
        assert response["incomplete_details"].reason == "max_output_tokens"
        assert response["max_output_tokens"] == 20

        # Validate usage information
        assert response["usage"]["input_tokens"] == 73
        assert response["usage"]["output_tokens"] == 20
        assert response["usage"]["total_tokens"] == 93

        # Validate that the response is properly identified as incomplete
        validate_responses_api_response(response, final_chunk=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_o1_pro_response_api_streaming(sync_mode):
    """
    Test that LiteLLM correctly handles an incomplete response from OpenAI's o1-pro model
    due to reaching max_output_tokens limit in both sync and async streaming modes.
    """
    # Mock response from o1-pro
    mock_response = {
        "id": "resp_67dc3dd77b388190822443a85252da5a0e13d8bdc0e28d88",
        "object": "response",
        "created_at": 1742486999,
        "status": "incomplete",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": None,
        "max_output_tokens": 20,
        "model": "o1-pro-2025-03-19",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_67dc3de50f64819097450ed50a33d5f90e13d8bdc0e28d88",
                "summary": [],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 73,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 93,
        },
        "user": None,
        "metadata": {},
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        litellm.set_verbose = True

        # Verify the request was made correctly
        if sync_mode:
            # For sync mode, we need to patch the sync HTTP handler
            with patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                return_value=MockResponse(mock_response, 200),
            ) as mock_sync_post:
                response = litellm.responses(
                    model="openai/o1-pro",
                    input="Write a detailed essay about artificial intelligence and its impact on society",
                    max_output_tokens=20,
                    stream=True,
                )

                # Process the sync stream
                event_count = 0
                for event in response:
                    print(
                        f"Sync litellm response #{event_count}:",
                        json.dumps(event, indent=4, default=str),
                    )
                    event_count += 1

                # Verify the sync request was made correctly
                mock_sync_post.assert_called_once()
                request_body = mock_sync_post.call_args.kwargs["json"]
                assert request_body["model"] == "o1-pro"
                assert request_body["max_output_tokens"] == 20
                assert "stream" not in request_body
        else:
            # For async mode
            response = await litellm.aresponses(
                model="openai/o1-pro",
                input="Write a detailed essay about artificial intelligence and its impact on society",
                max_output_tokens=20,
                stream=True,
            )

            # Process the async stream
            event_count = 0
            async for event in response:
                print(
                    f"Async litellm response #{event_count}:",
                    json.dumps(event, indent=4, default=str),
                )
                event_count += 1

            # Verify the async request was made correctly
            mock_post.assert_called_once()
            request_body = mock_post.call_args.kwargs["json"]
            assert request_body["model"] == "o1-pro"
            assert request_body["max_output_tokens"] == 20
            assert "stream" not in request_body


def test_basic_computer_use_preview_tool_call():
    """
    Test that LiteLLM correctly handles a computer_use_preview tool call where the environment is set to "linux"

    linux is an unsupported environment for the computer_use_preview tool, but litellm users should still be able to pass it to openai
    """
    # Mock response from OpenAI

    mock_response = {
        "id": "resp_67dc3dd77b388190822443a85252da5a0e13d8bdc0e28d88",
        "object": "response",
        "created_at": 1742486999,
        "status": "incomplete",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": None,
        "max_output_tokens": 20,
        "model": "o1-pro-2025-03-19",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_67dc3de50f64819097450ed50a33d5f90e13d8bdc0e28d88",
                "summary": [],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 73,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 93,
        },
        "user": None,
        "metadata": {},
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=MockResponse(mock_response, 200),
    ) as mock_post:
        litellm.turn_on_debug()
        litellm.set_verbose = True

        # Call the responses API with computer_use_preview tool
        response = litellm.responses(
            model="openai/computer-use-preview",
            tools=[
                {
                    "type": "computer_use_preview",
                    "display_width": 1024,
                    "display_height": 768,
                    "environment": "linux",  # other possible values: "mac", "windows", "ubuntu"
                }
            ],
            input="Check the latest OpenAI news on bing.com.",
            reasoning={"summary": "concise"},
            truncation="auto",
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]

        # Validate the request structure
        assert request_body["model"] == "computer-use-preview"
        assert len(request_body["tools"]) == 1
        assert request_body["tools"][0]["type"] == "computer_use_preview"
        assert request_body["tools"][0]["display_width"] == 1024
        assert request_body["tools"][0]["display_height"] == 768
        assert request_body["tools"][0]["environment"] == "linux"

        # Check that reasoning was passed correctly
        assert request_body["reasoning"]["summary"] == "concise"
        assert request_body["truncation"] == "auto"

        # Validate the input format
        assert isinstance(request_body["input"], str)
        assert request_body["input"] == "Check the latest OpenAI news on bing.com."


@pytest.mark.asyncio
async def test_store_field_transformation():
    """Test store field transformation with mocked API responses"""
    config = OpenAIResponsesAPIConfig()

    # Initialize logging object with required parameters
    logging_obj = LiteLLMLoggingObj(
        model="gpt-5.5",
        messages=[],
        stream=False,
        call_type="aresponses",
        start_time=time.time(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )

    # Base response data with all required fields
    base_response = {
        "id": "test_id",
        "created_at": 1751443898,
        "model": "gpt-5.5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "error": None,
        "incomplete_details": None,
        "instructions": "test instructions",
        "metadata": {},
        "temperature": 0.7,
        "top_p": 1.0,
        "max_output_tokens": 100,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": None,
        "truncation": "auto",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "user": "test_user",
    }

    # Test case 1: API returns store=True
    mock_response_store_true = httpx.Response(
        status_code=200, content=json.dumps({**base_response, "store": True}).encode()
    )

    # Test case 2: API returns store=False
    mock_response_store_false = httpx.Response(
        status_code=200, content=json.dumps({**base_response, "store": False}).encode()
    )

    # Test case 3: API returns store=null
    mock_response_store_null = httpx.Response(
        status_code=200, content=json.dumps({**base_response, "store": None}).encode()
    )

    # Test case 4: API omits store field
    mock_response_no_store = httpx.Response(
        status_code=200, content=json.dumps(base_response).encode()
    )

    # Test when store=True in request
    logging_obj.optional_params = {"store": True}
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_store_true, logging_obj=logging_obj
    )
    assert (
        response.store is True
    ), "store should be True when specified in request and API returns True"

    # Test when store=False in request
    logging_obj.optional_params = {"store": False}
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_store_false, logging_obj=logging_obj
    )
    assert (
        response.store is False
    ), "store should be False when specified in request and API returns False"

    # Test when store not in request but API returns null
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_store_null, logging_obj=logging_obj
    )
    assert (
        response.store is None
    ), "store should be None when not specified in request and API returns null"

    # Test when store not in request and API omits store field
    response = config.transform_response_api_response(
        model="gpt-5.5", raw_response=mock_response_no_store, logging_obj=logging_obj
    )
    assert (
        response.store is None
    ), "store should be None when not specified in request and API omits store"

    # Verify created_at is always converted to integer
    assert isinstance(
        response.created_at, int
    ), "created_at should always be converted to integer"
    assert (
        response.created_at == 1751443898
    ), "created_at should maintain the same value after conversion"


@pytest.mark.asyncio
async def test_aresponses_service_tier_and_safety_identifier():
    """
    Test that service_tier and safety_identifier parameters are correctly sent in the request body
    when using litellm.aresponses.
    """
    mock_response = {
        "id": "resp_01234567890abcdef",
        "object": "response",
        "created_at": 1753060947,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-4o-2024-05-13",
        "output": [
            {
                "type": "text",
                "id": "out_01234567890abcdef",
                "text": "This is a test response with service tier and safety identifier.",
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": None,
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 25,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 40,
        },
        "user": None,
        "metadata": {},
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        litellm.set_verbose = True

        # Call aresponses with service_tier and safety_identifier
        response = await litellm.aresponses(
            model="openai/gpt-5.5",
            input="Test with service tier and safety identifier",
            service_tier="flex",
            safety_identifier="123",
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body=", json.dumps(request_body, indent=4, default=str))

        # Validate that both parameters are present in the request body
        assert (
            request_body["service_tier"] == "flex"
        ), "service_tier should be 'flex' in request body"
        assert (
            request_body["safety_identifier"] == "123"
        ), "safety_identifier should be '123' in request body"
        assert request_body["model"] == "gpt-5.5"
        assert request_body["input"] == "Test with service tier and safety identifier"

        # Validate the response
        print("Response:", json.dumps(response, indent=4, default=str))


@pytest.mark.asyncio
async def test_openai_gpt5_reasoning_effort_parameter():
    """Test that reasoning_effort parameter is properly sent in the HTTP request for GPT-5 models."""

    # Mock response for GPT-5 responses API (correct format)
    mock_response = {
        "id": "resp_01ABC123",
        "object": "response",
        "created_at": 1729621667,
        "status": "completed",
        "model": "gpt-5-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The capital of France is Paris.",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 23,
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": "low", "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()
        litellm.set_verbose = True

        # Call aresponses with reasoning_effort parameter
        response = await litellm.aresponses(
            model="openai/gpt-5-mini",
            input="What is the capital of France?",
            reasoning={"effort": "minimal"},
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body=", json.dumps(request_body, indent=4, default=str))
        print("reasoning=", request_body["reasoning"])
        # Validate that reasoning_effort is present in the request body
        assert (
            "reasoning" in request_body
        ), "reasoning should be present in request body"
        assert (
            request_body["reasoning"]["effort"] == "minimal"
        ), "reasoning_effort should be 'minimal' in request body"
        assert request_body["model"] == "gpt-5-mini"
        assert request_body["input"] == "What is the capital of France?"

        # Validate the response
        print("Response:", json.dumps(response, indent=4, default=str))


class MockResponse:
    def __init__(self, json_data, status_code):
        self._json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
        self.headers = httpx.Headers({})

    def json(self):
        return self._json_data


@pytest.fixture
def extra_body_mock_response_data():
    return {
        "id": "resp_test123",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "gpt-5.5",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "parallel_tool_calls": True,
        "text": {"format": {"type": "text"}},
        "error": None,
        "metadata": {},
        "temperature": 1.0,
        "reasoning": {"effort": None, "summary": None},
    }


@pytest.mark.asyncio
async def test_aresponses_extra_body_params_passed(extra_body_mock_response_data):
    """Test that extra_body parameters are passed in async mode."""
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(extra_body_mock_response_data, 200)

        response = await litellm.aresponses(
            model="gpt-5.5",
            input="Test input",
            max_output_tokens=20,
            extra_body={
                "custom_param_1": "value1",
                "custom_param_2": {"nested": "value2"},
                "experimental_feature": True,
            },
        )

        assert response is not None
        assert response.id is not None

        request_body = mock_post.call_args.kwargs["json"]

        assert "custom_param_1" in request_body
        assert request_body["custom_param_1"] == "value1"
        assert "custom_param_2" in request_body
        assert request_body["custom_param_2"]["nested"] == "value2"
        assert "experimental_feature" in request_body
        assert request_body["experimental_feature"] is True
        assert request_body["model"] == "gpt-5.5"
        assert request_body["input"] == "Test input"


def test_responses_extra_body_params_passed_sync(extra_body_mock_response_data):
    """Test that extra_body parameters are passed in sync mode."""
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=MockResponse(extra_body_mock_response_data, 200),
    ) as mock_post:
        response = litellm.responses(
            model="gpt-5.5",
            input="Sync test",
            max_output_tokens=20,
            extra_body={
                "sync_custom_param": "sync_value",
                "another_param": 42,
            },
        )

        assert response is not None
        assert response.id is not None

        request_body = mock_post.call_args.kwargs["json"]

        assert "sync_custom_param" in request_body
        assert request_body["sync_custom_param"] == "sync_value"
        assert "another_param" in request_body
        assert request_body["another_param"] == 42
        assert request_body["model"] == "gpt-5.5"


@pytest.mark.asyncio
async def test_extra_body_merges_with_request_data(extra_body_mock_response_data):
    """Test that extra_body is merged into the request data."""
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(extra_body_mock_response_data, 200)

        await litellm.aresponses(
            model="gpt-5.5",
            input="Test",
            temperature=1,
            max_output_tokens=20,
            extra_body={
                "custom_field": "custom_value",
            },
        )

        request_body = mock_post.call_args.kwargs["json"]

        assert "temperature" in request_body
        assert "custom_field" in request_body
        assert request_body["custom_field"] == "custom_value"


_MALFORMED_MARKERS: Final = ("yes", 1, {"mode": "bogus"}, {"mode": "explicit", "ttl": "1h"})
_MALFORMED_MARKER_IDS: Final = ("string", "int", "bogus-mode", "bad-ttl")
_VALID_MARKERS: Final = ({"mode": "explicit"}, {"mode": "explicit", "ttl": "30m"})
_VALID_MARKER_IDS: Final = ("explicit", "explicit-30m")
_PARTS: Final[Mapping[str, Mapping[str, object]]] = {
    "input_text": {"type": "input_text", "text": "Say ok"},
    "input_image": {"type": "input_image", "image_url": "https://example.com/a.png"},
    "input_file": {"type": "input_file", "file_id": "file-abc"},
}
_CONFIGS: Final = (OpenAIResponsesAPIConfig(), AzureOpenAIResponsesAPIConfig())
_CONFIG_IDS: Final = ("openai", "azure")


def _marked_message(part_type: str, marker: object) -> list[dict[str, object]]:
    return [{"type": "message", "role": "user", "content": [{**_PARTS[part_type], "prompt_cache_breakpoint": marker}]}]


def _transformed_input(
    config: BaseResponsesAPIConfig, input: str | list[dict[str, object]], litellm_params: GenericLiteLLMParams
) -> object:
    request: Final = config.transform_responses_api_request(
        model="gpt-6.1-sol",
        input=cast("ResponseInputParam", input),  # cast-ok: plain dicts shaped like the SDK's input items
        response_api_optional_request_params={},
        litellm_params=litellm_params,
        headers={},
    )
    return request["input"]


def _single_part(transformed_input: object) -> Mapping[str, object]:
    (item,) = cast(list[Mapping[str, object]], transformed_input)
    (part,) = cast(list[Mapping[str, object]], item["content"])
    return part


@pytest.mark.parametrize("config", _CONFIGS, ids=_CONFIG_IDS)
@pytest.mark.parametrize("part_type", tuple(_PARTS))
@pytest.mark.parametrize("marker", _MALFORMED_MARKERS, ids=_MALFORMED_MARKER_IDS)
def test_malformed_prompt_cache_breakpoint_is_dropped_under_deployment_drop_params(
    monkeypatch: pytest.MonkeyPatch, config: BaseResponsesAPIConfig, part_type: str, marker: object
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    transformed: Final = _transformed_input(
        config, _marked_message(part_type, marker), GenericLiteLLMParams(drop_params=True)
    )
    assert _single_part(transformed) == _PARTS[part_type]


@pytest.mark.parametrize("marker", _MALFORMED_MARKERS, ids=_MALFORMED_MARKER_IDS)
def test_malformed_prompt_cache_breakpoint_is_dropped_under_global_drop_params(
    monkeypatch: pytest.MonkeyPatch, marker: object
) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)
    transformed: Final = _transformed_input(_CONFIGS[0], _marked_message("input_text", marker), GenericLiteLLMParams())
    assert _single_part(transformed) == _PARTS["input_text"]


@pytest.mark.parametrize("drop_params", (False, None), ids=("false", "null"))
@pytest.mark.parametrize("marker", _MALFORMED_MARKERS, ids=_MALFORMED_MARKER_IDS)
def test_malformed_prompt_cache_breakpoint_is_forwarded_verbatim_without_drop_params(
    monkeypatch: pytest.MonkeyPatch, marker: object, drop_params: bool | None
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    input: Final = _marked_message("input_text", marker)
    assert _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params=drop_params)) == input


@pytest.mark.parametrize("marker", _MALFORMED_MARKERS, ids=_MALFORMED_MARKER_IDS)
def test_a_non_flag_deployment_drop_params_keeps_the_marker_like_the_param_mapper_does(
    monkeypatch: pytest.MonkeyPatch, marker: object
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    input: Final = _marked_message("input_text", marker)
    assert _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params="maybe")) == input


@pytest.mark.parametrize("drop_params", (True, False), ids=("drop", "keep"))
@pytest.mark.parametrize("marker", _VALID_MARKERS, ids=_VALID_MARKER_IDS)
def test_valid_prompt_cache_breakpoint_reaches_the_wire_under_both_settings(
    monkeypatch: pytest.MonkeyPatch, marker: Mapping[str, str], drop_params: bool
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    input: Final = _marked_message("input_text", marker)
    assert _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params=drop_params)) == input


def test_prompt_cache_breakpoint_unknown_key_is_trimmed_under_drop_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    input: Final = _marked_message("input_text", {"mode": "explicit", "note": "kept"})
    dropped: Final = _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params=True))
    assert _single_part(dropped)["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params=False)) == input


def test_malformed_prompt_cache_breakpoint_on_a_tool_output_part_is_dropped_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    marked_output: Final = [{"type": "input_text", "text": "42", "prompt_cache_breakpoint": "yes"}]
    input: Final[list[dict[str, object]]] = [{"type": "function_call_output", "call_id": "call_1", "output": marked_output}]
    (item,) = cast(
        list[Mapping[str, object]], _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params=True))
    )
    assert item["output"] == [{"type": "input_text", "text": "42"}]


@pytest.mark.parametrize("input", ("hi", [{"type": "message", "role": "user", "content": "hi"}]), ids=("string", "unmarked"))
def test_input_without_a_marker_is_untouched_under_drop_params(
    monkeypatch: pytest.MonkeyPatch, input: str | list[dict[str, object]]
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    assert _transformed_input(_CONFIGS[0], input, GenericLiteLLMParams(drop_params=True)) == input


@pytest.mark.asyncio
@pytest.mark.usefixtures(httpx_transport.__name__)
async def test_aresponses_forwards_previous_response_id_to_openai() -> None:
    first_input: Final = "remember the first turn"
    second_input: Final = "continue the conversation"
    first_id: Final = "resp_previous_turn"
    second_id: Final = "resp_follow_up"
    response_payloads: Final = (
        {
            "id": first_id,
            "object": "response",
            "created_at": 1734366691,
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_first",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "first answer", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
        {
            "id": second_id,
            "object": "response",
            "created_at": 1734366692,
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_second",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "second answer", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )
    provider_url: Final = "https://api.openai.com/v1/responses"

    with respx.mock() as router:
        route: Final = router.post(provider_url).mock(
            side_effect=[
                httpx.Response(status_code=200, json=response_payloads[0]),
                httpx.Response(status_code=200, json=response_payloads[1]),
            ]
        )
        first_response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input=first_input,
        )
        assert isinstance(first_response, ResponsesAPIResponse)
        second_response: Final = await litellm.aresponses(
            model="openai/gpt-4o",
            api_key="sk-test",
            input=second_input,
            previous_response_id=first_response.id,
        )
        calls: Final = route.calls

    assert isinstance(second_response, ResponsesAPIResponse)
    assert first_response.output[0].content[0].text == "first answer"
    assert second_response.output[0].content[0].text == "second answer"
    assert len(calls) == 2
    request_adapter: Final = TypeAdapter(dict[str, JsonValue])
    request_bodies: Final = tuple(request_adapter.validate_json(call.request.content) for call in calls)
    assert request_bodies[0]["input"] == first_input
    assert request_bodies[1]["input"] == second_input
    assert request_bodies[1]["previous_response_id"] == first_id


def test_dict_responses_input_filters_unset_reasoning_fields() -> None:
    test_input: Final = [
        {"role": "user", "content": "test"},
        {
            "id": "rs_123",
            "summary": [{"text": "test", "type": "summary_text"}],
            "type": "reasoning",
            "content": None,
            "encrypted_content": None,
            "status": None,
        },
        {
            "arguments": "{}",
            "call_id": "call_123",
            "name": "get_today",
            "type": "function_call",
            "id": "fc_123",
            "status": "completed",
        },
    ]

    validated_input: Final = OpenAIResponsesAPIConfig()._validate_input_param(test_input)

    assert len(validated_input) == 3
    reasoning_item: Final = validated_input[1]
    assert reasoning_item["type"] == "reasoning"
    assert "status" not in reasoning_item
    assert "content" not in reasoning_item
    assert "encrypted_content" not in reasoning_item
    assert reasoning_item["id"] == "rs_123"
    assert reasoning_item["summary"] == [{"text": "test", "type": "summary_text"}]

    function_call_item: Final = validated_input[2]
    assert function_call_item["type"] == "function_call"
    assert function_call_item["status"] == "completed"
