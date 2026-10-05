from base_llm_unit_tests import BaseLLMChatTest
import pytest
import os

import litellm
from litellm.types.llms.bedrock import BedrockInvokeNovaRequest


_LITELLM_LOGO_IMAGE_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm@d769e81c90d453240c61fc572cdb27fae06a89d0/"
    "ui/litellm-dashboard/public/assets/logos/litellm_logo.jpg"
)
_AWSMP_LOGO_IMAGE_URL = (
    "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/"
    "c233c9ade2ccb5491072ae232c814942.png"
)


@pytest.mark.flaky(retries=3, delay=5)
class TestBedrockInvokeClaudeJson(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    @pytest.mark.parametrize(
        "image_url, detail",
        [
            (_LITELLM_LOGO_IMAGE_URL, None),
            (_LITELLM_LOGO_IMAGE_URL, "low"),
            (_LITELLM_LOGO_IMAGE_URL, "high"),
            (_AWSMP_LOGO_IMAGE_URL, "low"),
            (_AWSMP_LOGO_IMAGE_URL, "high"),
        ],
    )
    @pytest.mark.flaky(retries=4, delay=2)
    def test_image_url(self, image_url, detail):
        super().test_image_url(detail=detail, image_url=image_url)
    test_content_list_handling = None
    test_image_url_string = None
    test_pdf_handling = None


class TestBedrockInvokeNovaJson(BaseLLMChatTest):
    test_json_response_format = None

    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "bedrock/invoke/us.amazon.nova-micro-v1:0",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    @pytest.fixture(autouse=True)
    def skip_non_json_tests(self, request):
        if not "json" in request.function.__name__.lower():
            pytest.skip(
                f"Skipping non-JSON test: {request.function.__name__} does not contain 'json'"
            )

    def test_json_response_pydantic_obj(self):
        if os.environ.get("LITELLM_RUN_LIVE_BEDROCK_NOVA_JSON_TESTS") != "1":
            pytest.skip("Live Bedrock Nova response-schema E2E tests are opt-in")
        if os.environ.get("CASSETTE_REDIS_URL"):
            pytest.skip(
                "Live Bedrock Nova response-schema E2E tests cannot run under VCR replay"
            )
        super().test_json_response_pydantic_obj()


def test_nova_invoke_remove_empty_system_messages():
    """Test that _remove_empty_system_messages removes empty system list."""
    input_request = BedrockInvokeNovaRequest(
        messages=[{"content": [{"text": "Hello"}], "role": "user"}],
        system=[],
        inferenceConfig={"temperature": 0.7},
    )

    litellm.AmazonInvokeNovaConfig()._remove_empty_system_messages(input_request)

    assert "system" not in input_request
    assert "messages" in input_request
    assert "inferenceConfig" in input_request


def test_nova_invoke_filter_allowed_fields():
    """
    Test that _filter_allowed_fields only keeps fields defined in BedrockInvokeNovaRequest.

    Nova Invoke does not allow `additionalModelRequestFields` and `additionalModelResponseFieldPaths` in the request body.
    This test ensures that these fields are not included in the request body.
    """
    _input_request = {
        "messages": [{"content": [{"text": "Hello"}], "role": "user"}],
        "system": [{"text": "System prompt"}],
        "inferenceConfig": {"temperature": 0.7},
        "additionalModelRequestFields": {"this": "should be removed"},
        "additionalModelResponseFieldPaths": ["this", "should", "be", "removed"],
    }

    input_request = BedrockInvokeNovaRequest(**_input_request)

    result = litellm.AmazonInvokeNovaConfig()._filter_allowed_fields(input_request)

    assert "additionalModelRequestFields" not in result
    assert "additionalModelResponseFieldPaths" not in result
    assert "messages" in result
    assert "system" in result
    assert "inferenceConfig" in result


def test_nova_invoke_streaming_chunk_parsing():
    """
    Test that the AWSEventStreamDecoder correctly handles Nova's /bedrock/invoke/ streaming format
    where content is nested under 'contentBlockDelta'.
    """
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

    # Initialize the decoder with a Nova model
    decoder = AWSEventStreamDecoder(model="bedrock/invoke/us.amazon.nova-micro-v1:0")

    # Test case 1: Text content in contentBlockDelta
    nova_text_chunk = {
        "contentBlockDelta": {
            "delta": {"text": "Hello, how can I help?"},
            "contentBlockIndex": 0,
        }
    }
    result = decoder._chunk_parser(nova_text_chunk)
    assert result.choices[0].delta.content == "Hello, how can I help?"
    assert result.choices[0].index == 0
    assert not result.choices[0].finish_reason
    assert result.choices[0].delta.tool_calls is None

    # Test case 2: Tool use start in contentBlockDelta
    nova_tool_start_chunk = {
        "contentBlockDelta": {
            "start": {"toolUse": {"name": "get_weather", "toolUseId": "tool_1"}},
            "contentBlockIndex": 1,
        }
    }
    result = decoder._chunk_parser(nova_tool_start_chunk)
    assert result.choices[0].delta.content == ""
    assert result.choices[0].index == 0
    assert result.choices[0].delta.tool_calls is not None
    assert result.choices[0].delta.tool_calls[0].type == "function"
    assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"
    assert result.choices[0].delta.tool_calls[0].id == "tool_1"

    # Test case 3: Tool use arguments in contentBlockDelta
    nova_tool_args_chunk = {
        "contentBlockDelta": {
            "delta": {"toolUse": {"input": '{"location": "New York"}'}},
            "contentBlockIndex": 2,
        }
    }
    result = decoder._chunk_parser(nova_tool_args_chunk)
    assert result.choices[0].delta.content == ""
    assert result.choices[0].index == 0
    assert result.choices[0].delta.tool_calls is not None
    assert (
        result.choices[0].delta.tool_calls[0].function.arguments
        == '{"location": "New York"}'
    )

    # Test case 4: Stop reason in contentBlockDelta
    nova_stop_chunk = {
        "contentBlockDelta": {
            "stopReason": "tool_use",
        }
    }
    result = decoder._chunk_parser(nova_stop_chunk)
    print(result)
    assert result.choices[0].finish_reason == "tool_calls"
