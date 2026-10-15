import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Final, Literal, cast

import pytest
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_create_params import ResponseInputParam
from openai.types.responses.response_function_web_search import (
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    ActionSearchSource,
    ResponseFunctionWebSearch,
)
from pydantic import TypeAdapter

import litellm
from litellm.integrations.anthropic_cache_control_hook import MAX_CACHE_CONTROL_BLOCKS, AnthropicCacheControlHook
from litellm.litellm_core_utils.hidden_params import HIDDEN_PARAMS_ATTR
from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_converse_messages_pt,
    anthropic_messages_pt,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    TOOL_CALLS_CACHE,
    ChatCompletionSession,
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.base import HiddenParams
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.responses.main import OutputToolSearchCall, build_web_search_call
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    CompletionTokensDetailsWrapper,
    Function,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestLiteLLMCompletionResponsesConfig:
    def test_transform_input_file_item_to_file_item_with_file_id(self):
        """Test transformation of input_file item with file_id to Chat Completion file format"""
        # Setup
        input_item = {"type": "input_file", "file_id": "file-abc123xyz"}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(input_item)

        # Assert
        expected = {"type": "file", "file": {"file_id": "file-abc123xyz"}}
        assert result == expected
        assert result["type"] == "file"
        assert result["file"]["file_id"] == "file-abc123xyz"

    def test_transform_input_file_item_to_file_item_with_file_data(self):
        """Test transformation of input_file item with file_data to Chat Completion file format"""
        # Setup
        file_data = "base64encodeddata"
        input_item = {"type": "input_file", "file_data": file_data}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(input_item)

        # Assert
        expected = {"type": "file", "file": {"file_data": file_data}}
        assert result == expected
        assert result["type"] == "file"
        assert result["file"]["file_data"] == file_data

    def test_transform_input_file_item_to_file_item_with_both_fields(self):
        """Test transformation of input_file item with both file_id and file_data"""
        # Setup
        input_item = {
            "type": "input_file",
            "file_id": "file-abc123xyz",
            "file_data": "base64encodeddata",
        }

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(input_item)

        # Assert
        expected = {
            "type": "file",
            "file": {"file_id": "file-abc123xyz", "file_data": "base64encodeddata"},
        }
        assert result == expected
        assert result["type"] == "file"
        assert result["file"]["file_id"] == "file-abc123xyz"
        assert result["file"]["file_data"] == "base64encodeddata"

    def test_transform_input_file_item_to_file_item_empty_file_fields(self):
        """Test transformation of input_file item with no file_id or file_data"""
        # Setup
        input_item = {"type": "input_file"}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(input_item)

        # Assert
        expected = {"type": "file", "file": {}}
        assert result == expected
        assert result["type"] == "file"
        assert result["file"] == {}

    def test_transform_input_file_item_to_file_item_ignores_other_fields(self):
        """Test that transformation only includes file_id and file_data, ignoring other fields"""
        # Setup
        input_item = {
            "type": "input_file",
            "file_id": "file-abc123xyz",
            "extra_field": "should_be_ignored",
            "another_field": 123,
        }

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(input_item)

        # Assert
        expected = {"type": "file", "file": {"file_id": "file-abc123xyz"}}
        assert result == expected
        assert "extra_field" not in result["file"]
        assert "another_field" not in result["file"]

    def test_transform_input_file_item_to_file_item_keeps_filename(self):
        """OpenAI rejects file_data with no filename beside it, so dropping it 400s the request"""
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
            {
                "type": "input_file",
                "filename": "report.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0=",
            }
        )
        assert result == {
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,JVBERi0=",
                "filename": "report.pdf",
            },
        }

    def test_transform_input_file_item_to_file_item_with_file_url(self):
        """file_url should be mapped to file_id for downstream URL handling"""
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
            {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
        )
        assert result == {
            "type": "file",
            "file": {"file_id": "https://example.com/doc.pdf"},
        }

    def test_transform_input_file_item_file_id_takes_precedence_over_file_url(self):
        """explicit file_id should not be overwritten by file_url"""
        result = LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
            {
                "type": "input_file",
                "file_id": "file-abc123",
                "file_url": "https://example.com/doc.pdf",
            }
        )
        assert result == {"type": "file", "file": {"file_id": "file-abc123"}}

    def test_transform_input_image_item_to_image_item_with_image_url(self):
        """Test transformation of input_image item with image_url to Chat Completion image format"""
        # Setup
        image_url = "https://example.com/image.png"
        input_item = {"type": "input_image", "image_url": image_url, "detail": "high"}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(input_item)

        # Assert
        expected = {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        }
        assert result == expected
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "high"

    def test_transform_input_image_item_to_image_item_with_image_data(self):
        """Test transformation of input_image item with image_url to Chat Completion image format"""
        # Setup
        image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAMTWlDQ1BJQ0MgUHJvZmlsZQAASImVVwdYU1cbPndkQggQiICMsJcgMgPICGEFkD0EUQlJgDBiTAgqbqRYwbpFBEdFqyCKqwJSXKhVK0XBPYsDFaUWa3Er/wkBtPQfz/89z7n3ve/5znu+77vnjgMAvYsvleaimgDkSfJlsSEBrMnJKSxSDyAAJlADCBjHF8ilnOjoCABt+Px3e30N+kG77KDU+mf/fzUtoUguAACJhjhdKBfkQfwjAHiLQCrLB4Aohbz5rHypEq+DWEcGA4S4RokzVbhFidNV+NKgT3wsF+JHAJDV+XxZJgAafZBnFQgyoQ4dZgucJEKxBGJ/iH3z8mYIIV4EsQ30gXPSlfrs9K90Mv+mmT6iyednjmBVLoNGDhTLpbn8Of9nOf635eUqhuewhk09SxYaq8wZ1u1RzoxwJVaH+K0kPTIKYm0AUFwsHPRXYmaWIjRB5Y/aCORcWDN4pwE6UZ4bxxviY4X8wHCIDSHOkORGRgz5FGWIg5U+sH5ohTifFw+xHsQ1InlQ3JDPCdmM2OF5r2XIuJwh/ilfNhiDUv+zIieBo9LHtLNEvCF9zLEwKz4JYirEgQXixEiINSCOlOfEhQ/5pBZmcSOHfWSKWGUuFhDLRJKQAJU+Vp4hC44d8t+dJx/OHTuRJeZFDuHO/Kz4UFWtsEcC/mD8MBesTyThJAzriOSTI4ZzEYoCg1S542SRJCFOxeN60vyAWNVY3E6aGz3kjweIckOUvBnE8fKCuOGxBflwcar08RJpfnS8Kk68MpsfFq2KB98PIgAXBAIWUMCWDmaAbCBu723shVeqnmDABzKQCUTAYYgZHpE02COBxzhQCH6HSATkI+MCBntFoADyn0axSk48wqmODiBjqE+pkgMeQ5wHwkEuvFYMKklGIkgEjyAj/kdEfNgEMIdc2JT9/54fZr8wHMhEDDGK4RlZ9GFPYhAxkBhKDCba4ga4L+6NR8CjP2zOOBv3HM7jiz/hMaGD8IBwldBFuDldXCQbFeUk0AX1g4fqk/51fXArqOmGB+A+UB0q40zcADjgrnAeDu4HZ3aDLHcobmVVWKO0/5bBV3doyI/iREEpYyj+FJvRIzXsNNxGVJS1/ro+qljTR+rNHekZPT/3q+oL4Tl8tCf2LXYIO4udxM5jLVgjYGHHsSasDTuqxCMr7tHgihueLXYwnhyoM3rNfLmzykrKneqcepw+qvryRbPzlQ8jd4Z0jkycmZXP4sAvhojFkwgcx7GcnZzdAFB+f1Svt1cxg98VhNn2hVvyGwA+xwcGBn76woUdB+CAB3wlHPnC2bDhp0UNgHNHBApZgYrDlQcCfHPQ4dOnD4yBObCB+TgDd+AN/EEQCANRIB4kg2kw+iy4zmVgFpgHFoMSUAZWgfWgEmwF20EN2AsOgkbQAk6Cn8EFcAlcBbfh6ukGz0EfeA0+IAhCQmgIA9FHTBBLxB5xRtiILxKERCCxSDKShmQiEkSBzEOWIGXIGqQS2YbUIgeQI8hJ5DzSgdxE7iM9yJ/IexRD1VEd1Ai1QsejbJSDhqPx6FQ0E52JFqLF6Aq0Aq1G96AN6En0AnoV7UKfo/0YwNQwJmaKOWBsjItFYSlYBibDFmClWDlWjdVjzfA+X8a6sF7sHU7EGTgLd4ArOBRPwAX4THwBvhyvxGvwBvw0fhm/j/fhnwk0giHBnuBF4BEmEzIJswglhHLCTsJhwhn4LHUTXhOJRCbRmugBn8VkYjZxLnE5cTNxH/EEsYP4kNhPIpH0SfYkH1IUiU/KJ5WQNpL2kI6TOkndpLdkNbIJ2ZkcTE4hS8hF5HLybvIxcif5CfkDRZNiSfGiRFGElDmUlZQdlGbKRUo35QNVi2pN9aHGU7Opi6kV1HrqGeod6is1NTUzNU+1GDWx2iK1CrX9aufU7qu9U9dWt1PnqqeqK9RXqO9SP6F+U/0VjUazovnTUmj5tBW0Wtop2j3aWw2GhqMGT0OosVCjSqNBo1PjBZ1Ct6Rz6NPohfRy+iH6RXqvJkXTSpOryddcoFmleUTzuma/FkNrglaUVp7Wcq3dWue1nmqTtK20g7SF2sXa27VPaT9kYAxzBpchYCxh7GCcYXTrEHWsdXg62TplOnt12nX6dLV1XXUTdWfrVuke1e1iYkwrJo+Zy1zJPMi8xnw/xmgMZ4xozLIx9WM6x7zRG6vnryfSK9Xbp3dV770+Sz9IP0d/tX6j/l0D3MDOIMZglsEWgzMGvWN1xnqPFYwtHXtw7C1D1NDOMNZwruF2wzbDfiNjoxAjqdFGo1NGvcZMY3/jbON1xseMe0wYJr4mYpN1JsdNnrF0WRxWLquCdZrVZ2poGmqqMN1m2m76wczaLMGsyGyf2V1zqjnbPMN8nXmreZ+FicUki3kWdRa3LCmWbMssyw2WZy3fWFlbJVkttWq0emqtZ82zLrSus75jQ7Pxs5lpU21zxZZoy7bNsd1se8kOtXOzy7Krsrtoj9q724vtN9t3jCOM8xwnGVc97rqDugPHocChzuG+I9MxwrHIsdHxxXiL8SnjV48/O/6zk5tTrtMOp9sTtCeETSia0DzhT2c7Z4FzlfMVF5pLsMtClyaXl672riLXLa433Bhuk9yWurW6fXL3cJe517v3eFh4pHls8rjO1mFHs5ezz3kSPAM8F3q2eL7zcvfK9zro9Ye3g3eO927vpxOtJ4om7pj40MfMh++zzafLl+Wb5vu9b5efqR/fr9rvgb+5v9B/p/8Tji0nm7OH8yLAKUAWcDjgDdeLO597IhALDAksDWwP0g5KCKoMuhdsFpwZXBfcF+IWMjfkRCghNDx0deh1nhFPwKvl9YV5hM0POx2uHh4XXhn+IMIuQhbRPAmdFDZp7aQ7kZaRksjGKBDFi1obdTfaOnpm9E8xxJjomKqYx7ETYufFno1jxE2P2x33Oj4gfmX87QSbBEVCayI9MTWxNvFNUmDSmqSuyeMnz598IdkgWZzclEJKSUzZmdI/JWjK+indqW6pJanXplpPnT31/DSDabnTjk6nT+dPP5RGSEtK2532kR/Fr+b3p/PSN6X3CbiCDYLnQn/hOmGPyEe0RvQkwydjTcbTTJ/MtZk9WX5Z5Vm9Yq64UvwyOzR7a/abnKicXTkDuUm5+/LIeWl5RyTakhzJ6RnGM2bP6JDaS0ukXTO9Zq6f2ScLl+2UI/Kp8qZ8Hfij36awUXyjuF/gW1BV8HZW4qxDs7VmS2a3zbGbs2zOk8Lgwh/m4nMFc1vnmc5bPO/+fM78bQuQBekLWheaLyxe2L0oZFHNYurinMW/FjkVrSn6a0nSkuZio+JFxQ+/CfmmrkSjRFZyfan30q3f4t+Kv21f5rJs47LPpcLSX8qcysrLPi4XLP/luwnfVXw3sCJjRftK95VbVhFXSVZdW+23umaN1prCNQ/XTlrbsI61rnTdX+unrz9f7lq+dQN1g2JDV0VERdNGi42rNn6szKq8WhVQtW+T4aZlm95sFm7u3OK/pX6r0dayre+/F39/Y1vItoZqq+ry7cTtBdsf70jccfYH9g+1Ow12lu38tEuyq6smtuZ0rUdt7W7D3Svr0DpFXc+e1D2X9gbubap3qN+2j7mvbD/Yr9j/7EDagWsHww+2HmIfqv/R8sdNhxmHSxuQhjkNfY1ZjV1NyU0dR8KOtDZ7Nx/+yfGnXS2mLVVHdY+uPEY9Vnxs4Hjh8f4T0hO9JzNPPmyd3nr71ORTV07HnG4/E37m3M/BP586yzl7/JzPuZbzXueP/ML+pfGC+4WGNre2w7+6/Xq43b294aLHxaZLnpeaOyZ2HOv06zx5OfDyz1d4Vy5cjbzacS3h2o3rqde7bghvPL2Ze/PlrYJbH24vukO4U3pX8275PcN71b/Z/ravy73r6P3A+20P4h7cfih4+PyR/NHH7uLHtMflT0ye1D51ftrSE9xz6dmUZ93Ppc8/9Jb8rvX7phc2L378w/+Ptr7Jfd0vZS8H/lz+Sv/Vrr9c/2rtj+6/9zrv9Yc3pW/139a8Y787+z7p/ZMPsz6SPlZ8sv3U/Dn8852BvIEBKV/GH/wVwIBya5MBwJ+7AKAlA8CA+0bqFNX+cNAQ1Z52EIH/hFV7yEFzB6Ae/tPH9MK/m+sA7N8BgBXUp6cCEE0DIN4ToC4uI214Lze471QaEe4Nvo/8lJ6XDv6NqfakX8U9+gyUqq5g9PlfxcODBS7Lae4AAACKZVhJZk1NACoAAAAIAAQBGgAFAAAAAQAAAD4BGwAFAAAAAQAAAEYBKAADAAAAAQACAACHaQAEAAAAAQAAAE4AAAAAAAAAkAAAAAEAAACQAAAAAQADkoYABwAAABIAAAB4oAIABAAAAAEAAAAOoAMABAAAAAEAAAAOAAAAAEFTQ0lJAAAAU2NyZWVuc2hvdDaPMpgAAAAJcEhZcwAAFiUAABYlAUlSJPAAAAHUaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPGV4aWY6UGl4ZWxZRGltZW5zaW9uPjE0PC9leGlmOlBpeGVsWURpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjE0PC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6VXNlckNvbW1lbnQ+U2NyZWVuc2hvdDwvZXhpZjpVc2VyQ29tbWVudD4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+Cjh4oDkAAAAcaURPVAAAAAIAAAAAAAAABwAAACgAAAAHAAAABwAAAE3Fs0eqAAAAGUlEQVQ4EWLkExT7z0AGYBzViDvUyA4cAAAAAP//YUvjIgAAABZJREFUY+QTFPvPQAZgHNWIO9TIDhwA/sQQ53tmETgAAAAASUVORK5CYII="
        input_item = {"type": "input_image", "image_url": image_url, "detail": "high"}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(input_item)

        # Assert
        expected = {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        }
        assert result == expected
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "high"

    def test_transform_input_image_item_to_image_item_without_detail(self):
        """Test transformation of input_image item with no detail"""
        # Setup
        image_url = "https://example.com/image.png"
        input_item = {"type": "input_image", "image_url": image_url}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(input_item)

        # Assert
        expected = {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "auto"},
        }
        assert result == expected
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "auto"

    def test_transform_input_image_item_to_image_item_empty_image_fields(self):
        """Test transformation of input_image item with no image_url or detail"""
        # Setup
        input_item = {"type": "input_image"}

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(input_item)

        # Assert
        expected = {"type": "image_url", "image_url": {"url": "", "detail": "auto"}}
        assert result == expected
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == ""
        assert result["image_url"]["detail"] == "auto"

    def test_transform_input_image_item_to_image_item_ignores_other_fields(self):
        """Test transformation of input_image item with other fields"""
        # Setup
        input_item = {
            "type": "input_image",
            "image_url": "https://example.com/image.png",
            "extra_field": "should_be_ignored",
            "another_field": 123,
        }

        # Execute
        result = LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(input_item)

        # Assert
        expected = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png", "detail": "auto"},
        }
        assert result == expected
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/image.png"
        assert result["image_url"]["detail"] == "auto"
        assert "extra_field" not in result
        assert "another_field" not in result

    def test_transform_chat_completion_response_with_reasoning_content(self):
        """Test that reasoning content is preserved in the full transformation pipeline"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="The answer is 42.",
                        role="assistant",
                        reasoning_content="Let me think about this step by step. The question asks for the meaning of life, and according to The Hitchhiker's Guide to the Galaxy, the answer is 42.",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="What is the meaning of life?",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        # Assert
        assert hasattr(responses_api_response, "output")
        assert len(responses_api_response.output) >= 2

        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"

        reasoning_item = reasoning_items[0]
        # Note: ID auto-generation was disabled, so reasoning items may not have IDs
        # Only assert ID format if an ID is present
        if hasattr(reasoning_item, "id") and reasoning_item.id:
            assert reasoning_item.id.startswith("rs_"), f"Expected ID to start with 'rs_', got: {reasoning_item.id}"
        assert reasoning_item.status == "completed"
        assert reasoning_item.role == "assistant"
        assert len(reasoning_item.content) == 1
        assert reasoning_item.content[0].type == "output_text"
        assert "step by step" in reasoning_item.content[0].text
        assert "42" in reasoning_item.content[0].text

        message_items = [item for item in responses_api_response.output if item.type == "message"]
        assert len(message_items) == 1, "Should have exactly one message item"

        message_item = message_items[0]
        assert message_item.content[0].text == "The answer is 42."

    def test_transform_chat_completion_response_without_reasoning_content(self):
        """Test that transformation works normally when no reasoning content is present"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Just a regular answer.",
                        role="assistant",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="A simple question?",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        # Assert
        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 0, "Should have no reasoning items"

        message_items = [item for item in responses_api_response.output if item.type == "message"]
        assert len(message_items) == 1, "Should have exactly one message item"
        assert message_items[0].content[0].text == "Just a regular answer."
        assert responses_api_response.object == "response"

    def test_transform_chat_completion_response_multiple_choices_with_reasoning(self):
        """Test that only reasoning from first choice is included when multiple choices exist"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="First answer.",
                        role="assistant",
                        reasoning_content="First reasoning process.",
                    ),
                ),
                Choices(
                    finish_reason="stop",
                    index=1,
                    message=Message(
                        content="Second answer.",
                        role="assistant",
                        reasoning_content="Second reasoning process.",
                    ),
                ),
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="A question with multiple answers?",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        # Assert
        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"
        assert reasoning_items[0].content[0].text == "First reasoning process."

        message_items = [item for item in responses_api_response.output if item.type == "message"]
        assert len(message_items) == 2, "Should have two message items"

    def test_signature_only_thinking_block_still_emits_reasoning_item(self):
        response = ModelResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="10",
                        role="assistant",
                        reasoning_content="",
                        thinking_blocks=[{"type": "thinking", "thinking": "", "signature": "signature-payload"}],
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test input",
                responses_api_request={},
                chat_completion_response=response,
            )
        )

        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 1, "Signature-only thinking should still surface a reasoning item"
        assert reasoning_items[0].content == []
        assert "signature-payload" in reasoning_items[0].encrypted_content

    def test_redacted_thinking_block_preserved_as_encrypted_content(self):
        response = ModelResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="10",
                        role="assistant",
                        thinking_blocks=[{"type": "redacted_thinking", "data": "redacted-payload"}],
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test input",
                responses_api_request={},
                chat_completion_response=response,
            )
        )

        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 1
        assert "redacted-payload" in reasoning_items[0].encrypted_content

    def test_visible_thinking_keeps_text_and_signature(self):
        response = ModelResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="10",
                        role="assistant",
                        reasoning_content="counting the primes",
                        thinking_blocks=[{"type": "thinking", "thinking": "counting the primes", "signature": "sig"}],
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test input",
                responses_api_request={},
                chat_completion_response=response,
            )
        )

        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0].content[0].text == "counting the primes"
        assert "sig" in reasoning_items[0].encrypted_content

    def test_transform_chat_completion_response_status_with_stop(self):
        """
        Test that transforming a chat completion response with 'stop' finish_reason
        results in 'completed' status in the responses API response.

        This is the main test case for GitHub issue #15714.
        """
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-2.5-flash-preview-09-2025",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="That's completely fine! How can I help you with your test?",
                        role="assistant",
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="this is a test",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        assert responses_api_response.status == "completed"
        assert responses_api_response.status in [
            "completed",
            "failed",
            "in_progress",
            "cancelled",
            "queued",
            "incomplete",
        ]

    def test_transform_chat_completion_response_output_item_status(self):
        """
        Test that output items in the transformed response also have valid status values.

        This verifies the fix for GitHub issue #15714.
        """
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-2.5-flash-preview-09-2025",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Test message",
                        role="assistant",
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="this is a test",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        message_items = [item for item in responses_api_response.output if item.type == "message"]
        assert len(message_items) > 0

        for item in message_items:
            assert item.status in [
                "completed",
                "failed",
                "in_progress",
                "cancelled",
                "queued",
                "incomplete",
            ]
            assert item.status != "stop"

    def test_transform_chat_completion_response_status_with_refusal(self):
        """
        `finish_reason=refusal` should map to `status=incomplete` in Responses API.
        """
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-sonnet-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="refusal",
                    index=0,
                    message=Message(
                        content="",
                        role="assistant",
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="this is a test",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        assert responses_api_response.status == "incomplete"

    def test_tool_call_only_response_emits_no_null_text_message_item(self):
        """A tool-calls-only turn (message content None, e.g. from Anthropic)
        must not emit a message output item whose output_text has text null.
        OpenAI rejects such an item on replay with
        "Invalid type for 'input[..].content[..].text': expected a string, but
        got null instead." Native OpenAI tool-only turns carry no message item."""
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-sonnet-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="toolu_01OnlyToolCall",
                                type="function",
                                function=Function(name="get_weather", arguments='{"city": "SF"}'),
                            )
                        ],
                    ),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="what's the weather in SF?",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        output_types = [item.type for item in responses_api_response.output]
        assert "message" not in output_types
        assert "function_call" in output_types

    def test_content_bearing_response_still_emits_message_item(self):
        """Turns with real text content must keep their message output item."""
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-sonnet-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="It is sunny.", role="assistant"),
                )
            ],
        )

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="what's the weather in SF?",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        message_items = [item for item in responses_api_response.output if item.type == "message"]
        assert len(message_items) == 1
        assert message_items[0].content[0].text == "It is sunny."

    def test_transform_chat_completion_response_preserves_hidden_params(self):
        """Test that _hidden_params from chat completion response are preserved in responses API response"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Test response",
                        role="assistant",
                    ),
                )
            ],
        )
        # Set hidden params on the chat completion response
        chat_completion_response._hidden_params = {
            "model_id": "abc123",
            "cache_key": "some-cache-key",
            "custom_llm_provider": "openai",
        }

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        # Assert
        assert hasattr(responses_api_response, "_hidden_params")
        assert responses_api_response._hidden_params == {
            "model_id": "abc123",
            "cache_key": "some-cache-key",
            "custom_llm_provider": "openai",
        }

    @pytest.mark.parametrize(
        "storage",
        [
            HiddenParams(model_id="m1", provider_specific_fields={"k": "v"}),
            {"model_id": "m1", "provider_specific_fields": {"k": "v"}},
        ],
    )
    def test_transform_chat_completion_response_preserves_hidden_params_storage_identity(
        self, storage: dict[str, object] | HiddenParams
    ) -> None:
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Test response", role="assistant"),
                )
            ],
        )
        setattr(chat_completion_response, HIDDEN_PARAMS_ATTR, storage)

        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        assert getattr(responses_api_response, HIDDEN_PARAMS_ATTR) is storage
        assert responses_api_response.hidden_params["model_id"] == "m1"
        assert responses_api_response.provider_specific_fields == {"k": "v"}

    def test_transform_chat_completion_response_handles_missing_hidden_params(self):
        """Test that missing _hidden_params defaults to empty dict"""
        # Setup - no _hidden_params set
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Test response",
                        role="assistant",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test",
                responses_api_request={},
                chat_completion_response=chat_completion_response,
            )
        )

        # Assert - should default to empty dict
        assert hasattr(responses_api_response, "_hidden_params")
        assert responses_api_response._hidden_params == {}

    def test_transform_chat_completion_response_restores_namespace_tool_call(self):
        tool_call_id = "call_namespace_restore"
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-3.1-pro-preview-customtools",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id=tool_call_id,
                                type="function",
                                function=Function(
                                    name="collaboration__spawn_agent",
                                    arguments='{"message":"hello"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        try:
            responses_api_response = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    request_input="Spawn an agent",
                    responses_api_request={
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "collaboration",
                                "tools": [
                                    {
                                        "type": "function",
                                        "name": "spawn_agent",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {},
                                        },
                                    }
                                ],
                            }
                        ]
                    },
                    chat_completion_response=chat_completion_response,
                )
            )
        finally:
            TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

        tool_calls = [item for item in responses_api_response.output if item.type == "function_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "spawn_agent"
        assert tool_calls[0].namespace == "collaboration"
        assert tool_calls[0].arguments == '{"message":"hello"}'

    def test_transform_chat_completion_response_plain_tool_call_has_no_namespace(self):
        """A non-namespace function call must not gain a namespace attribute, matching
        the streaming path which only sets it when a namespace was restored."""
        tool_call_id = "call_plain_no_namespace"
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gpt-4o",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id=tool_call_id,
                                type="function",
                                function=Function(
                                    name="get_weather",
                                    arguments='{"city":"Paris"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        try:
            responses_api_response = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    request_input="What is the weather in Paris?",
                    responses_api_request={
                        "tools": [
                            {
                                "type": "function",
                                "name": "get_weather",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ]
                    },
                    chat_completion_response=chat_completion_response,
                )
            )
        finally:
            TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

        tool_calls = [item for item in responses_api_response.output if item.type == "function_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].namespace is None
        assert "namespace" not in tool_calls[0].model_fields_set

    def test_transform_top_level_function_collision_stays_unnamespaced(self):
        tool_call_id = "call_top_level_collision"
        chat_completion_response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id=tool_call_id,
                                type="function",
                                function=Function(name="run", arguments="{}"),
                            )
                        ],
                    ),
                )
            ]
        )
        responses_api_request = {
            "tools": [
                {"type": "function", "name": "run", "parameters": {"type": "object"}},
                {
                    "type": "namespace",
                    "name": "admin",
                    "tools": [
                        {
                            "type": "function",
                            "name": "run",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            ]
        }

        try:
            response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Run the tool",
                responses_api_request=responses_api_request,
                chat_completion_response=chat_completion_response,
            )
        finally:
            TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

        tool_call = next(item for item in response.output if item.type == "function_call")
        assert tool_call.name == "run"
        assert getattr(tool_call, "namespace", None) is None


class TestFunctionCallTransformation:
    """Test cases for function_call input transformation"""

    def test_function_call_detection(self):
        """Test that function_call items are correctly detected"""
        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "test"}',
            "call_id": "test_id",
        }

        function_call_output_item = {
            "type": "function_call_output",
            "call_id": "test_id",
            "output": "result",
        }

        regular_message = {"type": "message", "role": "user", "content": "Hello"}

        # Test function_call detection
        assert LiteLLMCompletionResponsesConfig._is_input_item_function_call(function_call_item)
        assert not LiteLLMCompletionResponsesConfig._is_input_item_function_call(function_call_output_item)
        assert not LiteLLMCompletionResponsesConfig._is_input_item_function_call(regular_message)

        # Test function_call_output detection (should still work)
        assert LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(function_call_output_item)
        assert not LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(function_call_item)
        assert not LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(regular_message)

    def test_function_call_transformation(self):
        """Test that function_call items are correctly transformed to assistant messages with tool calls"""
        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "São Paulo, Brazil"}',
            "call_id": "call_123",
            "id": "call_123",
            "status": "completed",
        }

        result = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call=function_call_item
        )

        assert len(result) == 1
        message = result[0]

        # Should be an assistant message
        assert message.get("role") == "assistant"
        assert message.get("content") is None  # Function calls don't have content

        # Should have tool calls
        tool_calls = message.get("tool_calls", [])
        assert len(tool_calls) == 1

        tool_call = tool_calls[0]
        assert tool_call.get("id") == "call_123"
        assert tool_call.get("type") == "function"

        function = tool_call.get("function", {})
        assert function.get("name") == "get_weather"
        assert function.get("arguments") == '{"location": "São Paulo, Brazil"}'

    def test_function_call_transformation_normalizes_redacted_arguments(self):
        """Redacted rows hold the bare sentinel in arguments, which is invalid JSON."""
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call={
                "type": "function_call",
                "name": "get_weather",
                "arguments": "redacted-by-litellm",
                "call_id": "call_123",
            }
        )

        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_function_call_transformation_json_encodes_object_arguments(self):
        """A decoded arguments object must be JSON-encoded, not str()'d.

        Clients and providers sometimes send `arguments` as an object rather
        than a JSON string; `str()` on a dict produces a Python repr with
        single quotes, which downstream JSON parsers reject with errors like
        "Expecting ',' delimiter".
        """
        function_call_item = {
            "type": "function_call",
            "name": "shell",
            "arguments": {"command": "ls", "timeout": 30, "flags": ["-l", "-a"]},
            "call_id": "call_123",
            "id": "call_123",
            "status": "completed",
        }

        result = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call=function_call_item
        )

        arguments = result[0].get("tool_calls", [])[0].get("function", {}).get("arguments")
        assert json.loads(arguments) == {"command": "ls", "timeout": 30, "flags": ["-l", "-a"]}
        assert "'" not in arguments

    def test_create_tool_call_chunk_json_encodes_object_arguments(self):
        """Cached tool_call definitions with object arguments stay valid JSON."""
        chunk = LiteLLMCompletionResponsesConfig._create_tool_call_chunk(
            tool_use_definition={
                "id": "call_456",
                "type": "function",
                "function": {"name": "shell", "arguments": {"command": "ls"}},
            },
            tool_call_id="call_456",
            index=0,
        )

        assert json.loads(chunk["function"]["arguments"]) == {"command": "ls"}

    def test_create_tool_call_chunk_keeps_empty_arguments_default(self):
        """Missing arguments still fall back to an empty JSON object."""
        chunk = LiteLLMCompletionResponsesConfig._create_tool_call_chunk(
            tool_use_definition={"id": "call_789", "type": "function", "function": {"name": "shell"}},
            tool_call_id="call_789",
            index=0,
        )

        assert chunk["function"]["arguments"] == "{}"

    def test_complete_input_transformation_with_function_calls(self):
        """Test the complete transformation with the exact input from the issue"""
        test_input = [
            {
                "type": "message",
                "role": "user",
                "content": "How is the weather in São Paulo today ?",
            },
            {
                "type": "function_call",
                "arguments": '{"location": "São Paulo, Brazil"}',
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "name": "get_weather",
                "id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "output": "Rainy",
            },
        ]

        # This should not raise an error (previously would raise "Invalid content type: <class 'NoneType'>")
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=test_input
        )

        assert len(messages) == 3

        # First message: user message
        user_msg = messages[0]
        assert user_msg.get("role") == "user"
        assert user_msg.get("content") == "How is the weather in São Paulo today ?"

        # Second message: assistant message with tool call
        assistant_msg = messages[1]
        assert assistant_msg.get("role") == "assistant"
        assert assistant_msg.get("tool_calls") is not None
        assert len(assistant_msg.get("tool_calls", [])) == 1

        tool_call = assistant_msg.get("tool_calls")[0]
        assert tool_call.get("function", {}).get("name") == "get_weather"

        # Third message: tool output
        tool_msg = messages[2]
        assert tool_msg.get("role") == "tool"
        assert tool_msg.get("content") == "Rainy"
        assert tool_msg.get("tool_call_id") == "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5"

    def test_complete_request_transformation_with_function_calls(self):
        """Test the complete request transformation that would be used by the responses API"""
        test_input = [
            {
                "type": "message",
                "role": "user",
                "content": "How is the weather in São Paulo today ?",
            },
            {
                "type": "function_call",
                "arguments": '{"location": "São Paulo, Brazil"}',
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "name": "get_weather",
                "id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "output": "Rainy",
            },
        ]

        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            }
        ]

        responses_api_request = {"store": False, "tools": tools}

        # This should work without errors for non-OpenAI models
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="gemini/gemini-2.0-flash",
            input=test_input,
            responses_api_request=responses_api_request,
            extra_headers={"X-Test-Header": "test-value"},
        )

        assert "messages" in result
        assert "model" in result
        assert "tools" in result

        messages = result["messages"]
        assert len(messages) == 3
        assert result["model"] == "gemini/gemini-2.0-flash"

        # Verify the structure is correct for chat completion
        user_msg = messages[0]
        assert user_msg["role"] == "user"

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg

        tool_msg = messages[2]
        assert tool_msg["role"] == "tool"

        assert result["extra_headers"] == {"X-Test-Header": "test-value"}

    def test_drops_tool_choice_when_no_tools(self):
        """Chat completions providers reject tool_choice when no tools are present."""
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="azure_ai/grok-4.3",
            input="who are you?",
            responses_api_request={"tool_choice": "auto", "tools": []},
            custom_llm_provider="azure_ai",
        )

        assert "tool_choice" not in result
        assert "tools" not in result

    def test_safety_identifier_forwarded_to_chat_completion_request(self) -> None:
        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="bedrock/global.openai.gpt-5.6-luna",
            input="hi",
            responses_api_request={"safety_identifier": "user-7f3a"},
            custom_llm_provider="bedrock",
        )

        assert result["safety_identifier"] == "user-7f3a"

    def test_parallel_tool_calls_dropped_when_no_chat_tools_remain(self) -> None:
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
        codex_tool_search: Final = {
            "type": "tool_search",
            "execution": "client",
            "description": "Searches over deferred tool metadata with BM25.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        }
        function_tool: Final = {
            "type": "function",
            "name": "get_goal",
            "description": "Returns the current goal.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }

        empty_tools_result: Final = transform(
            model="azure/gpt-5.4-mini",
            input="Reply with just the word pong.",
            responses_api_request={"tools": [], "parallel_tool_calls": True},
            custom_llm_provider="azure",
        )
        hosted_only_result: Final = transform(
            model="azure/gpt-5.4-mini",
            input="Reply with just the word pong.",
            responses_api_request={"tools": [codex_tool_search], "parallel_tool_calls": True},
            custom_llm_provider="azure",
        )
        function_tools_result: Final = transform(
            model="azure/gpt-5.4-mini",
            input="Reply with just the word pong.",
            responses_api_request={"tools": [function_tool], "parallel_tool_calls": True},
            custom_llm_provider="azure",
        )

        assert "parallel_tool_calls" not in empty_tools_result
        assert "parallel_tool_calls" not in hosted_only_result
        assert function_tools_result["parallel_tool_calls"] is True

    def test_function_call_without_call_id_fallback_to_id(self):
        """Test that function_call items can use 'id' field when 'call_id' is missing"""
        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "test"}',
            "id": "fallback_id",  # Only has 'id', not 'call_id'
        }

        result = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call=function_call_item
        )

        assert len(result) == 1
        message = result[0]
        tool_calls = message.get("tool_calls", [])
        assert len(tool_calls) == 1

        tool_call = tool_calls[0]
        assert tool_call.get("id") == "fallback_id"

    def test_ensure_tool_results_preserves_cached_openai_object_tool_call(self):
        """
        Test cached ChatCompletionMessageToolCall objects are normalized correctly.
        """
        tool_call_id = "call_cached_openai_object"
        TOOL_CALLS_CACHE.set_cache(
            key=tool_call_id,
            value=ChatCompletionMessageToolCall(
                id=tool_call_id,
                type="function",
                function=Function(
                    name="search_web",
                    arguments='{"query": "python bugs"}',
                ),
            ),
        )

        messages_missing_tool_calls = [
            {"role": "user", "content": "Search for python bugs"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "tool",
                "content": "Found 5 results",
                "tool_call_id": tool_call_id,
            },
        ]

        try:
            fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
                messages=messages_missing_tool_calls,
                tools=None,
            )
        finally:
            TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

        assistant_msg = fixed_messages[1]
        tool_calls = assistant_msg.get("tool_calls", [])
        assert len(tool_calls) == 1

        tool_call = tool_calls[0]
        function = tool_call.get("function", {})
        assert function.get("name") == "search_web"
        assert function.get("arguments") == '{"query": "python bugs"}'

    def test_ensure_tool_results_preserves_cached_attr_object_tool_call(self):
        """
        Test cached attribute-only tool call objects are normalized correctly.
        """

        class AttrOnlyFunction:
            def __init__(self, name: str, arguments: str):
                self.name = name
                self.arguments = arguments

        class AttrOnlyToolCall:
            def __init__(self, id: str, type: str, function: AttrOnlyFunction):
                self.id = id
                self.type = type
                self.function = function

        tool_call_id = "call_cached_attr_object"
        TOOL_CALLS_CACHE.set_cache(
            key=tool_call_id,
            value=AttrOnlyToolCall(
                id=tool_call_id,
                type="function",
                function=AttrOnlyFunction(
                    name="search_web",
                    arguments='{"query": "attribute objects"}',
                ),
            ),
        )

        messages_missing_tool_calls = [
            {"role": "user", "content": "Search using attr object"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "tool",
                "content": "Found 3 results",
                "tool_call_id": tool_call_id,
            },
        ]

        try:
            fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
                messages=messages_missing_tool_calls,
                tools=None,
            )
        finally:
            TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

        assistant_msg = fixed_messages[1]
        tool_calls = assistant_msg.get("tool_calls", [])
        assert len(tool_calls) == 1

        tool_call = tool_calls[0]
        function = tool_call.get("function", {})
        assert function.get("name") == "search_web"
        assert function.get("arguments") == '{"query": "attribute objects"}'


class TestToolChoiceTransformation:
    """Test the tool_choice transformation fix for Cursor IDE bug"""

    def test_transform_tool_choice_cursor_bug_fix(self):
        """
        Test that {"type": "tool"} is transformed to "required".
        This fixes the Anthropic error: "tool_choice.tool.name: Field required"
        """
        result = LiteLLMCompletionResponsesConfig._transform_tool_choice({"type": "tool"})
        assert result == "required"

    def test_transform_tool_choice_preserves_function_with_name(self):
        """Test that valid OpenAI format with function name passes through unchanged"""
        tool_choice = {"type": "function", "function": {"name": "my_tool"}}
        result = LiteLLMCompletionResponsesConfig._transform_tool_choice(tool_choice)
        assert result == tool_choice

    def test_transform_tool_choice_responses_flat_function_name(self):
        """Responses-API forced-function with a top-level name maps to the nested Chat
        Completions shape instead of degrading to required and dropping the name"""
        result = LiteLLMCompletionResponsesConfig._transform_tool_choice({"type": "function", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_transform_tool_choice_custom_follows_function_downgrade(self):
        """
        This bridge downgrades custom tools to function tools
        (convert_custom_tool_to_function_tool), so a custom tool_choice must become a
        function tool_choice naming the same tool or it references a tool type absent
        from the converted request.
        """
        flat = LiteLLMCompletionResponsesConfig._transform_tool_choice({"type": "custom", "name": "ApplyPatch"})
        assert flat == {"type": "function", "function": {"name": "ApplyPatch"}}

        nested = LiteLLMCompletionResponsesConfig._transform_tool_choice(
            {"type": "custom", "custom": {"name": "ApplyPatch"}}
        )
        assert nested == {"type": "function", "function": {"name": "ApplyPatch"}}

    def test_transform_tool_choice_custom_without_name_falls_back_to_required(self):
        result = LiteLLMCompletionResponsesConfig._transform_tool_choice({"type": "custom"})
        assert result == "required"

    def test_transform_tool_choice_function_without_name_falls_back_to_required(self):
        """A function-type dict with no name still falls back to required"""
        result = LiteLLMCompletionResponsesConfig._transform_tool_choice({"type": "function"})
        assert result == "required"

    def test_transform_tool_choice_function_empty_name_falls_back_to_required(self):
        """An empty top-level name is falsy and must not produce an empty function name"""
        result = LiteLLMCompletionResponsesConfig._transform_tool_choice({"type": "function", "name": ""})
        assert result == "required"

    @pytest.mark.parametrize(
        "request_tool_choice,expected",
        [
            ({"type": "function", "name": "run_command"}, {"type": "function", "name": "run_command"}),
            ({"type": "function", "function": {"name": "run_command"}}, {"type": "function", "name": "run_command"}),
            ({"type": "custom", "name": "ApplyPatch"}, {"type": "custom", "name": "ApplyPatch"}),
            ({"type": "custom", "custom": {"name": "ApplyPatch"}}, {"type": "custom", "name": "ApplyPatch"}),
            ({"type": "function"}, "required"),
            ({"type": "tool"}, "required"),
            ({"type": "auto"}, "auto"),
            ("required", "required"),
            ("none", "none"),
            (None, "auto"),
            ("any", "auto"),
            ("run_command", "auto"),
            ({"name": "run_command"}, "auto"),
        ],
    )
    def test_transform_tool_choice_for_responses_api_response(
        self, request_tool_choice: object, expected: str | dict[str, str]
    ) -> None:
        result: Final = LiteLLMCompletionResponsesConfig.transform_tool_choice_for_responses_api_response(
            request_tool_choice
        )
        assert result == expected

    def test_non_streamed_response_echoes_named_tool_choice_in_responses_api_shape(self) -> None:
        chat_completion_response: Final = ModelResponse(
            id="chatcmpl-named-tool-choice",
            created=1748575031,
            model="claude-haiku-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    index=0,
                    finish_reason="tool_calls",
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_pwd",
                                type="function",
                                function=Function(name="run_command", arguments='{"command":"pwd"}'),
                            )
                        ],
                    ),
                )
            ],
        )

        responses_api_response: Final = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Run the command pwd.",
                responses_api_request={"tool_choice": {"type": "function", "name": "run_command"}},
                chat_completion_response=chat_completion_response,
            )
        )

        assert responses_api_response.tool_choice == {"type": "function", "name": "run_command"}

    def test_non_streamed_response_with_unrecognized_tool_choice_echoes_auto(self) -> None:
        chat_completion_response: Final = ModelResponse(
            id="chatcmpl-unrecognized-tool-choice",
            created=1748575031,
            model="claude-haiku-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    index=0,
                    finish_reason="stop",
                    message=Message(role="assistant", content="/Users/dev"),
                )
            ],
        )

        responses_api_response: Final = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Run the command pwd.",
                responses_api_request={"tool_choice": "any"},
                chat_completion_response=chat_completion_response,
            )
        )

        assert responses_api_response.tool_choice == "auto"


class TestContentTypeTransformation:
    """Test content type transformation from Responses API to Chat Completion format"""

    def test_tool_result_content_type_transformed_to_text(self):
        """
        Test that 'tool_result' content type is transformed to 'text'.
        This fixes: Invalid user message - content type 'tool_result' not valid.
        """
        result = LiteLLMCompletionResponsesConfig._get_chat_completion_request_content_type("tool_result")
        assert result == "text"

    def test_input_text_content_type_transformed_to_text(self):
        """Test that 'input_text' content type is transformed to 'text'"""
        result = LiteLLMCompletionResponsesConfig._get_chat_completion_request_content_type("input_text")
        assert result == "text"

    def test_none_text_blocks_filtered_out(self):
        """
        Test that content blocks with None text are filtered out.
        This fixes: TypeError: object of type 'NoneType' has no len()
        in Anthropic transformation when text is None.
        """
        content = [
            {"type": "text", "text": "valid text"},
            {"type": "text", "text": None},  # Should be filtered out
            {"type": "text", "text": "another valid"},
        ]
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)
        assert len(result) == 2
        assert result[0]["text"] == "valid text"
        assert result[1]["text"] == "another valid"

    def test_encrypted_content_blocks_preserved_as_text(self):
        """
        OpenAI Responses agent messages can include encrypted_content blocks.
        Chat-completions providers need the payload as text instead of silently
        dropping it.
        """
        content = [
            {"type": "input_text", "text": "Payload:\n"},
            {
                "type": "encrypted_content",
                "encrypted_content": "Reply exactly INPUT_AGENT_OK",
            },
        ]

        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)

        assert result == [
            {"type": "text", "text": "Payload:\n"},
            {"type": "text", "text": "Reply exactly INPUT_AGENT_OK"},
        ]


def _deferred_tools_with_server_tool_search() -> list[dict[str, object]]:
    return [
        {"type": "tool_search", "execution": "server"},
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            "defer_loading": True,
        },
        {
            "type": "namespace",
            "name": "finance",
            "tools": [
                {
                    "type": "function",
                    "name": "get_stock_price",
                    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
                    "defer_loading": True,
                }
            ],
        },
    ]


def _chat_response_with_tool_search_call(
    search_arguments: str, search_call_id: str = "srvtoolu_01Search"
) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-tool-search",
        created=1748575031,
        model="claude-sonnet-5-5",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": search_call_id,
                            "type": "function",
                            "function": {"name": "tool_search_tool_regex", "arguments": search_arguments},
                        },
                        {
                            "id": "toolu_01Weather",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        },
                    ],
                },
            }
        ],
    )


class TestToolTransformation:
    """Test cases for tool transformation from Responses API to Chat Completion format"""

    def test_transform_vertex_ai_tools(self):
        """Test that Vertex AI tools are passed through as-is"""
        from litellm.types.llms.vertex_ai import VertexToolName

        # Create a Vertex AI tool using the enum value
        vertex_tool = {VertexToolName.CODE_EXECUTION.value: {}}

        tools = [vertex_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        assert result_tools[0] == vertex_tool
        assert web_search_options is None

    def test_transform_mcp_tools(self):
        """Test that MCP tools are passed through as-is"""
        mcp_tool = {
            "type": "mcp",
            "server_label": "zapier",
            "server_url": "https://mcp.zapier.com/api/mcp/mcp",
            "headers": {"Authorization": "Bearer token123"},
        }

        tools = [mcp_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        assert result_tools[0] == mcp_tool
        assert result_tools[0]["type"] == "mcp"
        assert web_search_options is None

    def test_transform_computer_use_tools(self):
        """Test that computer_use tools are dropped (no Chat Completions equivalent).

        This deliberately reverses the previous pass-through regression guard:
        forwarding computer_use verbatim made Chat Completions providers reject
        the whole request with "'function' is a required property", so the
        bridge now drops such tools (with a warning log) instead.
        """
        computer_use_tool = {
            "type": "computer_use",
            "display_width_px": 1024,
            "display_height_px": 768,
        }

        tools = [computer_use_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert - computer_use has no Chat Completions equivalent, so it is dropped
        assert len(result_tools) == 0
        assert web_search_options is None

    def test_transform_codex_tools_drops_hosted_tool_search(self) -> None:
        codex_tools: Final = [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Runs a command in a PTY.",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
                "strict": True,
            },
            {
                "type": "function",
                "name": "write_stdin",
                "description": "Writes characters to an existing session's stdin.",
                "parameters": {
                    "type": "object",
                    "properties": {"session_id": {"type": "number"}, "chars": {"type": "string"}},
                    "required": ["session_id", "chars"],
                },
                "strict": True,
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "The `apply_patch` tool can be used to edit files.",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\n',
                },
            },
            {
                "type": "tool_search",
                "execution": "client",
                "description": (
                    "# Tool discovery\n\nSearches over deferred tool metadata with BM25 and exposes matching tools "
                    "for the next model call.\n\nYou have access to tools from the following sources:\n"
                    "- Multi-agent tools: Spawn and manage sub-agents.\nSome of the tools may not have been provided "
                    "to you upfront, and you should use this tool (`tool_search`) to search for the required tools. "
                    "For MCP tool discovery, always use `tool_search` instead of `list_mcp_resources` or "
                    "`list_mcp_resource_templates`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "number", "description": "Maximum number of tools to return. Defaults to 8."},
                        "query": {"type": "string", "description": "Search query for deferred tools."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {"type": "web_search", "external_web_access": False, "search_content_types": ["text", "image"]},
        ]
        function_and_custom_count: Final = sum(1 for tool in codex_tools if tool["type"] in ("function", "custom"))

        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=codex_tools)

        assert not any(tool.get("type") == "tool_search" for tool in result_tools)
        assert all(tool.get("type") == "function" for tool in result_tools)
        assert len(result_tools) == function_and_custom_count
        assert web_search_options is not None

    @pytest.mark.parametrize("provider", ["anthropic", "azure_ai"])
    def test_server_tool_search_becomes_anthropic_tool_search_for_claude(self, provider: str) -> None:
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
        result: Final = transform(
            model=f"{provider}/claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
            custom_llm_provider=provider,
        )

        chat_tools: Final = result["tools"]
        assert chat_tools[0] == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
        assert [tool["function"]["name"] for tool in chat_tools[1:]] == ["get_weather", "finance__get_stock_price"]
        assert [tool["defer_loading"] for tool in chat_tools[1:]] == [True, True]

    def test_server_tool_search_dropped_for_models_without_tool_search(self) -> None:
        model: Final = "openai/model-with-no-cost-map-entry"
        assert litellm.supports_tool_search(model) is False

        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
            custom_llm_provider="openai",
        )

        assert [tool["type"] for tool in result["tools"]] == ["function", "function"]

    def test_dropping_the_search_tool_undefers_the_remaining_tools(self) -> None:
        model: Final = "azure_ai/claude-opus-4-1"
        assert litellm.supports_tool_search(model, "azure_ai") is False

        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
            custom_llm_provider="azure_ai",
        )

        assert [tool["type"] for tool in result["tools"]] == ["function", "function"]
        assert all("defer_loading" not in tool for tool in result["tools"])

    @pytest.mark.parametrize(
        "search_tools", [[], [{"type": "tool_search", "execution": "client"}]], ids=["none", "client"]
    )
    @pytest.mark.parametrize(
        ("model", "provider"),
        [
            ("anthropic/claude-sonnet-5-5", "anthropic"),
            ("vertex_ai/claude-sonnet-5-5", "vertex_ai"),
            ("bedrock/invoke/us.anthropic.claude-sonnet-5-5-v1:0", "bedrock"),
            ("azure_ai/claude-sonnet-5-5", "azure_ai"),
        ],
    )
    def test_claude_routes_keep_the_callers_deferral_when_no_server_search_tool_is_sent(
        self, model: str, provider: str, search_tools: list[dict[str, object]]
    ) -> None:
        function_tools: Final = [
            tool for tool in _deferred_tools_with_server_tool_search() if tool["type"] != "tool_search"
        ]

        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": [*search_tools, *function_tools]},
            custom_llm_provider=provider,
        )

        assert [tool["function"]["name"] for tool in result["tools"]] == ["get_weather", "finance__get_stock_price"]
        assert [tool["defer_loading"] for tool in result["tools"]] == [True, True]

    @pytest.mark.parametrize(
        ("model", "provider"),
        [
            ("mistral/mistral-large-latest", "mistral"),
            ("gemini/gemini-2.5-flash", "gemini"),
            ("bedrock/converse/us.anthropic.claude-sonnet-5-5-v1:0", "bedrock"),
        ],
    )
    def test_routes_without_anthropic_deferral_get_deferred_tools_undeferred(self, model: str, provider: str) -> None:
        tools: Final = [tool for tool in _deferred_tools_with_server_tool_search() if tool["type"] != "tool_search"]

        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": tools},
            custom_llm_provider=provider,
        )

        assert [tool["function"]["name"] for tool in result["tools"]] == ["get_weather", "finance__get_stock_price"]
        assert all("defer_loading" not in tool for tool in result["tools"])

    def test_native_bm25_search_tool_keeps_the_other_tools_deferred(self) -> None:
        tools: Final = [
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            *_deferred_tools_with_server_tool_search()[1:],
        ]

        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="anthropic/claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": tools},
            custom_llm_provider="anthropic",
        )

        chat_tools: Final = result["tools"]
        assert chat_tools[0] == {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
        assert [tool["defer_loading"] for tool in chat_tools[1:]] == [True, True]

    @pytest.mark.parametrize(
        "native_search_tool",
        [
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
        ],
    )
    def test_server_tool_search_beside_a_native_search_tool_sends_only_the_native_one(
        self, native_search_tool: Mapping[str, str]
    ) -> None:
        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="anthropic/claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": [native_search_tool, *_deferred_tools_with_server_tool_search()]},
            custom_llm_provider="anthropic",
        )

        chat_tools: Final = result["tools"]
        assert [tool for tool in chat_tools if tool["type"] != "function"] == [native_search_tool]
        assert [tool["defer_loading"] for tool in chat_tools if tool["type"] == "function"] == [True, True]

    def test_native_search_tool_passes_through_routes_without_tool_search(self) -> None:
        model: Final = "azure_ai/claude-opus-4-1"
        assert litellm.supports_tool_search(model, "azure_ai") is False
        tools: Final = [
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
            *_deferred_tools_with_server_tool_search()[1:],
        ]

        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": tools},
            custom_llm_provider="azure_ai",
        )

        chat_tools: Final = result["tools"]
        assert chat_tools[0] == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
        assert [tool["defer_loading"] for tool in chat_tools[1:]] == [True, True]

    def test_server_tool_search_follows_the_bedrock_route(self) -> None:
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
        converse: Final = transform(
            model="bedrock/anthropic.claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
            custom_llm_provider="bedrock",
        )
        invoke: Final = transform(
            model="bedrock/invoke/anthropic.claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
            custom_llm_provider="bedrock",
        )

        assert [tool["type"] for tool in converse["tools"]] == ["function", "function"]
        assert invoke["tools"][0] == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}

    def test_caller_function_named_like_the_search_tool_stays_a_function_call(self) -> None:
        items: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            chat_completion_response=_chat_response_with_tool_search_call(
                '{"pattern": "weather"}', search_call_id="call_01Mine"
            ),
            responses_api_request={
                "tools": [
                    *_deferred_tools_with_server_tool_search(),
                    {"type": "function", "name": "tool_search_tool_regex", "parameters": {"type": "object"}},
                ]
            },
        )

        assert [(item.type, item.name) for item in items] == [
            ("function_call", "tool_search_tool_regex"),
            ("function_call", "get_weather"),
        ]

    def test_defer_loading_must_be_a_boolean(self) -> None:
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools
        namespace_tool: Final = {
            "type": "namespace",
            "name": "finance",
            "tools": [{"type": "function", "name": "get_stock_price", "parameters": {"type": "object"}}],
        }

        with pytest.raises(litellm.BadRequestError, match="defer_loading must be a boolean"):
            transform(tools=[{**namespace_tool, "tools": [{**namespace_tool["tools"][0], "defer_loading": "false"}]}])
        with pytest.raises(litellm.BadRequestError, match="defer_loading must be a boolean"):
            transform(tools=[{"type": "function", "name": "get_weather", "parameters": {}, "defer_loading": "false"}])
        not_deferred, _ = transform(
            tools=[
                {**namespace_tool, "tools": [{**namespace_tool["tools"][0], "defer_loading": False}]},
                {"type": "function", "name": "get_weather", "parameters": {}, "defer_loading": False},
            ]
        )
        assert [tool["function"]["name"] for tool in not_deferred] == ["finance__get_stock_price", "get_weather"]
        assert all("defer_loading" not in tool for tool in not_deferred)

    @pytest.mark.parametrize("global_drop_params", [False, True])
    def test_malformed_tool_flags_answer_400_unless_drop_params_drops_them(
        self, monkeypatch: pytest.MonkeyPatch, global_drop_params: bool
    ) -> None:
        monkeypatch.setattr(litellm, "drop_params", global_drop_params)
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
        tools: Final = [
            {"type": "tool_search", "execution": "server"},
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"},
                "defer_loading": "false",
                "allowed_callers": "direct",
            },
            {
                "type": "namespace",
                "name": "finance",
                "tools": [
                    "not a tool",
                    {
                        "type": "function",
                        "name": "get_stock_price",
                        "parameters": {"type": "object"},
                        "defer_loading": 1,
                        "allowed_callers": "direct",
                    },
                    {
                        "type": "function",
                        "name": "get_fx_rate",
                        "parameters": {"type": "object"},
                        "defer_loading": True,
                    },
                ],
            },
            {"type": "custom", "name": "apply_patch", "defer_loading": "yes"},
        ]

        if not global_drop_params:
            with pytest.raises(litellm.BadRequestError):
                transform(
                    model="anthropic/claude-sonnet-5-5",
                    input="What is the weather in Paris right now?",
                    responses_api_request={"tools": tools},
                    custom_llm_provider="anthropic",
                )
        result: Final = transform(
            model="anthropic/claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": tools},
            custom_llm_provider="anthropic",
            drop_params=None if global_drop_params else True,
        )

        function_tools: Final = [tool for tool in result["tools"] if tool["type"] == "function"]
        assert {tool["function"]["name"]: tool.get("defer_loading", False) for tool in function_tools} == {
            "get_weather": False,
            "finance__get_stock_price": False,
            "finance__get_fx_rate": True,
            "apply_patch": False,
        }
        assert all("allowed_callers" not in tool for tool in function_tools)

    @pytest.mark.parametrize("execution", ["", "bogus", 5, ["server"]])
    def test_malformed_tool_search_execution_answers_400_unless_drop_params_drops_it(self, execution: object) -> None:
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
        tools: Final = [{"type": "tool_search", "execution": execution}, *_deferred_tools_with_server_tool_search()[1:]]

        with pytest.raises(litellm.BadRequestError, match="tool_search execution must be 'server' or 'client'"):
            transform(
                model="anthropic/claude-sonnet-5-5",
                input="What is the weather in Paris right now?",
                responses_api_request={"tools": tools},
                custom_llm_provider="anthropic",
            )
        dropped: Final = transform(
            model="anthropic/claude-sonnet-5-5",
            input="What is the weather in Paris right now?",
            responses_api_request={"tools": tools},
            custom_llm_provider="anthropic",
            drop_params=True,
        )
        items: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            chat_completion_response=_chat_response_with_tool_search_call('{"pattern": "weather"}'),
            responses_api_request={"tools": tools},
        )

        assert dropped["tools"][0] == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
        assert all(tool.get("defer_loading") is True for tool in dropped["tools"][1:])
        assert [item.type for item in items] == ["tool_search_call", "function_call"]

    def test_top_level_function_allowed_callers_must_be_a_list_of_strings(self) -> None:
        transform: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools
        weather_tool: Final = {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}

        with pytest.raises(litellm.BadRequestError, match="allowed_callers must be a list of strings"):
            transform(tools=[{**weather_tool, "allowed_callers": "direct"}])
        converted, _ = transform(tools=[{**weather_tool, "allowed_callers": ["direct"]}])
        assert converted[0]["allowed_callers"] == ["direct"]

    def test_custom_tools_stay_deferred_when_tool_search_goes_out(self) -> None:
        result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="anthropic/claude-sonnet-5-5",
            input="Patch the README",
            responses_api_request={
                "tools": [
                    {"type": "tool_search", "execution": "server"},
                    {"type": "custom", "name": "apply_patch", "defer_loading": True},
                    {
                        "type": "namespace",
                        "name": "editor",
                        "tools": [{"type": "custom", "name": "insert", "defer_loading": True}],
                    },
                    {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                ]
            },
            custom_llm_provider="anthropic",
        )

        function_tools: Final = [tool for tool in result["tools"] if tool["type"] == "function"]
        assert {tool["function"]["name"]: tool.get("defer_loading", False) for tool in function_tools} == {
            "apply_patch": True,
            "insert": True,
            "get_weather": False,
        }

    def test_anthropic_tool_search_call_becomes_tool_search_call_item(self) -> None:
        response: Final = _chat_response_with_tool_search_call('{"pattern": "weather|forecast"}')
        with_tool_search: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            chat_completion_response=response,
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
        )
        without_tool_search: Final = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
                chat_completion_response=response,
                responses_api_request={"tools": _deferred_tools_with_server_tool_search()[1:]},
            )
        )

        assert [item.type for item in with_tool_search] == ["tool_search_call", "function_call"]
        search_call: Final = with_tool_search[0]
        assert isinstance(search_call, OutputToolSearchCall)
        assert search_call.id == "tsc_srvtoolu_01Search"
        assert search_call.call_id == "srvtoolu_01Search"
        assert search_call.arguments == {"pattern": "weather|forecast"}
        assert search_call.execution == "server"
        assert search_call.status == "completed"
        assert with_tool_search[1].name == "get_weather"
        assert [item.type for item in without_tool_search] == ["function_call", "function_call"]

        full_response: Final = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="What is the weather in Paris right now?",
                responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
                chat_completion_response=response,
            )
        )
        assert [item.type for item in full_response.output] == ["tool_search_call", "function_call"]

    def test_tool_search_call_with_unparseable_arguments_is_incomplete(self) -> None:
        items: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            chat_completion_response=_chat_response_with_tool_search_call('{"pattern": '),
            responses_api_request={"tools": _deferred_tools_with_server_tool_search()},
        )

        search_call: Final = items[0]
        assert isinstance(search_call, OutputToolSearchCall)
        assert search_call.status == "incomplete"
        assert search_call.arguments == {}

    def test_tool_search_items_are_not_replayed_to_chat_messages(self) -> None:
        tools: Final = _deferred_tools_with_server_tool_search()
        messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=[
                {"type": "message", "role": "user", "content": "What is the weather in Paris right now?"},
                {
                    "type": "tool_search_call",
                    "id": "tsc_srvtoolu_01Search",
                    "call_id": "srvtoolu_01Search",
                    "arguments": {"pattern": "weather|forecast"},
                    "execution": "server",
                    "status": "completed",
                },
                {
                    "type": "tool_search_output",
                    "id": "tso_srvtoolu_01Search",
                    "call_id": "srvtoolu_01Search",
                    "execution": "server",
                    "status": "completed",
                    "tools": [tools[1]],
                },
                {
                    "type": "function_call",
                    "id": "fc_toolu_01Weather",
                    "call_id": "toolu_01Weather",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                    "status": "completed",
                },
                {"type": "function_call_output", "call_id": "toolu_01Weather", "output": "18C and sunny"},
            ],
            responses_api_request={"tools": tools},
        )

        assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
        assert [call["id"] for call in messages[1]["tool_calls"]] == ["toolu_01Weather"]
        assert "tool_search" not in json.dumps(messages)

    def test_transform_local_shell_tools_dropped(self) -> None:
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[{"type": "local_shell"}]
        )

        assert result_tools == []
        assert web_search_options is None

    def test_transform_custom_tools_to_function_tools(self):
        """Test that custom (freeform/grammar) tools are converted to function tools"""
        custom_tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to files",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: begin_patch hunk+ end_patch",
            },
        }

        tools = [custom_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert - custom tool is converted to a function tool
        assert len(result_tools) == 1
        assert result_tools[0]["type"] == "function"
        assert result_tools[0]["function"]["name"] == "apply_patch"
        assert "content" in result_tools[0]["function"]["parameters"]["properties"]
        assert result_tools[0]["function"]["parameters"]["required"] == ["content"]
        assert "begin_patch" in result_tools[0]["function"]["description"]
        assert web_search_options is None

    def test_transform_custom_tools_without_format(self):
        """Test that custom tools without format info are still converted"""
        custom_tool = {
            "type": "custom",
            "name": "exec",
            "description": "Execute code",
        }

        tools = [custom_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        assert result_tools[0]["type"] == "function"
        assert result_tools[0]["function"]["name"] == "exec"
        assert result_tools[0]["function"]["description"] == "Execute code"

    def test_transform_custom_tools_preserves_allowed_callers(self):
        """allowed_callers on a custom tool gates direct model invocation in the
        Anthropic adapter, so it must survive the custom->function conversion."""
        custom_tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to files",
            "allowed_callers": ["exec"],
        }

        tools = [custom_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        assert result_tools[0]["type"] == "function"
        assert result_tools[0]["allowed_callers"] == ["exec"]

    def test_transform_custom_tools_without_allowed_callers(self):
        """A custom tool without allowed_callers must not synthesize the field."""
        custom_tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to files",
        }

        tools = [custom_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        assert "allowed_callers" not in result_tools[0]

    def test_transform_custom_tools_rejects_invalid_allowed_callers(self):
        """Invalid allowed_callers must raise rather than silently dropping the
        provider-side allowlist."""
        custom_tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to files",
            "allowed_callers": "exec",
        }

        tools = [custom_tool]

        with pytest.raises(litellm.BadRequestError, match="allowed_callers must be a list of strings"):
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

    def test_transform_web_search_tools_to_web_search_options(self):
        """Test that web_search tools are converted to web_search_options"""
        web_search_tool = {
            "type": "web_search_preview",
            "search_context_size": "medium",
            "user_location": {"country": "US"},
        }

        tools = [web_search_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 0  # Web search is not added to tools
        assert web_search_options is not None
        assert web_search_options.get("search_context_size") == "medium"
        assert web_search_options.get("user_location") == {"country": "US"}

    def test_transform_function_tools_with_anthropic_specific_fields(self):
        """Test that Anthropic-specific fields are preserved in function tools"""
        function_tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            "cache_control": {"type": "ephemeral"},
            "defer_loading": True,
            "allowed_callers": ["user"],
            "input_examples": [{"location": "San Francisco"}],
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["type"] == "function"
        assert result_tool["function"]["name"] == "get_weather"
        assert result_tool["function"]["description"] == "Get weather for a location"
        assert result_tool["cache_control"] == {"type": "ephemeral"}
        assert result_tool["defer_loading"] is True
        assert result_tool["allowed_callers"] == ["user"]
        assert result_tool["input_examples"] == [{"location": "San Francisco"}]
        assert web_search_options is None

    def test_transform_function_tools_with_cache_control_only(self):
        """Test that cache_control field is preserved when present"""
        function_tool = {
            "type": "function",
            "name": "search",
            "description": "Search function",
            "parameters": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert "cache_control" in result_tool
        assert result_tool["cache_control"]["type"] == "ephemeral"

    def test_transform_function_tools_without_anthropic_fields(self):
        """Test that function tools work when anthropic-specific fields are not present"""
        function_tool = {
            "type": "function",
            "name": "simple_function",
            "description": "A simple function",
            "parameters": {
                "type": "object",
                "properties": {"param": {"type": "string"}},
            },
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["type"] == "function"
        assert result_tool["function"]["name"] == "simple_function"
        # Anthropic-specific fields should not be present
        assert "cache_control" not in result_tool
        assert "defer_loading" not in result_tool
        assert "allowed_callers" not in result_tool
        assert "input_examples" not in result_tool
        assert "eager_input_streaming" not in result_tool

    @pytest.mark.parametrize("eager_input_streaming", [True, False])
    def test_transform_function_tools_forwards_eager_input_streaming(self, eager_input_streaming: bool) -> None:
        function_tool: Final = {
            "type": "function",
            "name": "write_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            "eager_input_streaming": eager_input_streaming,
        }

        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[function_tool]
        )

        assert result_tools[0]["eager_input_streaming"] is eager_input_streaming

    @pytest.mark.parametrize("eager_input_streaming", [True, False])
    def test_chat_completion_tools_to_responses_tools_keeps_eager_input_streaming(
        self, eager_input_streaming: bool
    ) -> None:
        chat_tool: Final = {
            "type": "function",
            "function": {"name": "write_file", "parameters": {"type": "object"}},
            "eager_input_streaming": eager_input_streaming,
        }

        result_tools: Final = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools([chat_tool])
        )

        assert result_tools[0]["eager_input_streaming"] is eager_input_streaming

    def test_transform_code_execution_tools(self):
        """Test that code_execution tools are passed through as-is"""
        code_execution_tool = {
            "type": "code_execution_20250825",
            "name": "python_code_execution",
        }

        tools = [code_execution_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        assert result_tools[0]["type"] == "code_execution_20250825"

    def test_transform_tool_search_tools(self):
        """Test that tool_search tools are passed through as-is"""
        tool_search_regex = {
            "name": "tool_search_tool_regex",
            "description": "Search tools using regex",
        }

        tool_search_bm25 = {
            "name": "tool_search_tool_bm25",
            "description": "Search tools using BM25",
        }

        tools = [tool_search_regex, tool_search_bm25]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 2
        assert result_tools[0]["name"] == "tool_search_tool_regex"
        assert result_tools[1]["name"] == "tool_search_tool_bm25"

    def test_transform_mixed_tools_list(self):
        """Test transforming a mixed list of different tool types"""
        from litellm.types.llms.vertex_ai import VertexToolName

        tools = [
            # Regular function tool with anthropic fields
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            },
            # MCP tool
            {"type": "mcp", "server_label": "zapier"},
            # Web search tool
            {"type": "web_search_preview", "search_context_size": "high"},
            # Vertex AI tool
            {VertexToolName.CODE_EXECUTION.value: {}},
        ]

        # Execute
        (
            result_tools,
            web_search_options,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 3  # function, mcp, vertex (web_search becomes options)
        assert web_search_options is not None

        # Check function tool
        func_tools = [t for t in result_tools if t.get("type") == "function"]
        assert len(func_tools) == 1
        assert func_tools[0]["cache_control"]["type"] == "ephemeral"

        # Check MCP tool
        mcp_tools = [t for t in result_tools if t.get("type") == "mcp"]
        assert len(mcp_tools) == 1

        # Check web search was converted to options
        assert web_search_options.get("search_context_size") == "high"

    def test_transform_function_tools_parameters_with_missing_type(self):
        """Test that parameters get 'type': 'object' added if missing"""
        function_tool = {
            "type": "function",
            "name": "test_function",
            "description": "Test function",
            "parameters": {"properties": {"arg": {"type": "string"}}},
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"
        assert "properties" in result_tool["function"]["parameters"]

    def test_transform_function_tools_parameters_keep_client_key_order(self):
        tools = [
            {
                "type": "function",
                "name": "a",
                "parameters": {"properties": {"arg": {"type": "string"}}, "required": ["arg"]},
            },
            {"type": "function", "name": "b", "parameters": {"type": "object", "properties": {}}},
        ]

        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )

        assert list(result_tools[0]["function"]["parameters"]) == ["properties", "required", "type"]
        assert list(result_tools[1]["function"]["parameters"]) == ["type", "properties"]

    def test_transform_function_tools_empty_parameters(self):
        """Test that empty parameters get 'type': 'object' added"""
        function_tool = {
            "type": "function",
            "name": "test_function",
            "description": "Test function",
            "parameters": {},
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"

    def test_transform_function_tools_missing_parameters(self):
        """Test that missing parameters get default 'type': 'object' added"""
        function_tool = {
            "type": "function",
            "name": "test_function",
            "description": "Test function",
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"

    def test_transform_function_tools_preserves_existing_type(self):
        """Test that existing 'type': 'object' in parameters is preserved"""
        function_tool = {
            "type": "function",
            "name": "test_function",
            "description": "Test function",
            "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
        }

        tools = [function_tool]

        # Execute
        (
            result_tools,
            _,
        ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"
        assert "properties" in result_tool["function"]["parameters"]
        assert result_tool["function"]["parameters"]["properties"]["arg"]["type"] == "string"

    def test_bedrock_anthropic_drops_derived_web_search_options(self):
        """
        Regression for LIT-3858: a Responses web_search tool becomes a derived
        web_search_options param. Bedrock Anthropic models do not support it, so on the
        chat-completion bridge it must be dropped (so litellm doesn't raise
        UnsupportedParamsError) without the caller setting drop_params.
        """
        responses_api_request = {
            "tools": [{"type": "web_search", "external_web_access": False}],
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="bedrock",
        )

        assert "web_search_options" not in result

    def test_bedrock_converse_provider_drops_derived_web_search_options(self):
        """
        Regression for the ``bedrock_converse`` alias: routing a Bedrock Converse model resolves
        to custom_llm_provider='bedrock_converse' with model='bedrock/converse/...'. The derived
        web_search_options must still be dropped on this route, not just the bare 'bedrock' one.
        """
        responses_api_request = {
            "tools": [{"type": "web_search", "external_web_access": False}],
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="bedrock/converse/us.anthropic.claude-sonnet-4-6",
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="bedrock_converse",
        )

        assert "web_search_options" not in result

    def test_bedrock_nova_keeps_derived_web_search_options(self):
        """Nova models map web_search_options to a nova_grounding systemTool, so keep it."""
        responses_api_request = {
            "tools": [{"type": "web_search", "search_context_size": "high"}],
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="amazon.nova-pro-v1:0",
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="bedrock",
        )

        assert result.get("web_search_options") is not None

    def test_supported_provider_keeps_derived_web_search_options(self):
        """A provider whose config lists web_search_options (e.g. OpenAI) keeps it untouched."""
        responses_api_request = {
            "tools": [{"type": "web_search", "search_context_size": "high"}],
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="gpt-4o",
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="openai",
        )

        assert result.get("web_search_options") is not None

    def test_unsupported_non_bedrock_provider_drops_derived_web_search_options(self):
        """
        The drop is provider-agnostic, not hardcoded to Bedrock: any provider whose config
        does not list web_search_options (e.g. Cohere) drops the derived param. This fails if
        the drop is ever re-scoped to a single provider.
        """
        responses_api_request = {
            "tools": [{"type": "web_search", "search_context_size": "high"}],
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="command-r",
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="cohere",
        )

        assert "web_search_options" not in result

    def test_transform_nested_namespace_tools_to_function_tools(self):
        """Codex Responses namespace tools contain nested functions that chat
        providers need as flattened function names."""
        namespace_tool = {
            "type": "namespace",
            "name": "collaboration",
            "description": "Multi-agent tools",
            "tools": [
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn an agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["task_name", "message"],
                    },
                }
            ],
        }

        result_tools, web_search_options = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                tools=[namespace_tool]
            )
        )

        assert web_search_options is None
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["type"] == "function"
        assert result_tool["function"]["name"] == "collaboration__spawn_agent"
        assert result_tool["function"]["parameters"] == namespace_tool["tools"][0]["parameters"]
        assert result_tool["function"]["description"] == "Multi-agent tools\n\nSpawn an agent"

    def test_transform_namespace_tools_are_json_serializable(self):
        """Outbound chat payloads go through json.dumps, which rejects MappingProxyType."""
        namespace_tool = {
            "type": "namespace",
            "name": "mcp__everything",
            "description": "MCP tools",
            "tools": [
                {
                    "type": "function",
                    "name": "get_sum",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                }
            ],
        }

        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[namespace_tool]
        )

        assert "mcp__everything__get_sum" in json.dumps(result_tools)

    def test_function_call_echo_requalifies_namespace_tool_name(self):
        """Codex echoes restored history items as short name plus namespace; the
        outbound chat tool_call must use the flattened name the provider was given."""
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call={
                "type": "function_call",
                "name": "get_sum",
                "namespace": "mcp__everything",
                "call_id": "call_1",
                "arguments": '{"a": 21, "b": 21}',
            }
        )

        assert messages[0]["tool_calls"][0]["function"]["name"] == "mcp__everything__get_sum"

    def test_custom_tool_call_echo_keeps_short_name(self):
        """Custom tools stay advertised under their short name, so a namespace on
        a custom_tool_call echo is routing metadata and must not be prefixed."""
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call={
                "type": "custom_tool_call",
                "name": "apply_patch",
                "namespace": "mcp__everything",
                "call_id": "call_2",
                "input": "patch body",
            }
        )

        assert messages[0]["tool_calls"][0]["function"]["name"] == "apply_patch"

    @pytest.mark.parametrize("nested", [True, False])
    def test_transform_namespace_tools_preserves_allowed_callers(self, nested):
        function_tool = {
            "type": "function",
            "name": "spawn_agent",
            "parameters": {"type": "object", "properties": {}},
            "allowed_callers": ["code_execution_20250825"],
        }
        namespace_tool = (
            {
                "type": "namespace",
                "name": "collaboration",
                "tools": [function_tool],
            }
            if nested
            else {**function_tool, "type": "namespace"}
        )

        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[namespace_tool]
        )

        assert result_tools[0]["allowed_callers"] == ["code_execution_20250825"]

    @pytest.mark.parametrize("nested", [True, False])
    def test_transform_namespace_tools_rejects_invalid_allowed_callers(self, nested):
        function_tool = {
            "type": "function",
            "name": "spawn_agent",
            "parameters": {"type": "object", "properties": {}},
            "allowed_callers": "code_execution_20250825",
        }
        namespace_tool = (
            {
                "type": "namespace",
                "name": "collaboration",
                "tools": [function_tool],
            }
            if nested
            else {**function_tool, "type": "namespace"}
        )

        with pytest.raises(litellm.BadRequestError, match="allowed_callers must be a list of strings"):
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                tools=[namespace_tool]
            )

    def test_transform_flat_namespace_tools_to_function_tools(self):
        namespace_tool = {
            "type": "namespace",
            "name": "mcp__node_repl",
            "description": "Run JavaScript in the node REPL",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "JavaScript source to evaluate",
                    }
                },
                "required": ["code"],
            },
        }

        result_tools, web_search_options = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                tools=[namespace_tool]
            )
        )

        assert web_search_options is None
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["type"] == "function"
        assert result_tool["function"]["name"] == "mcp__node_repl"
        assert result_tool["function"]["description"] == "Run JavaScript in the node REPL"
        assert result_tool["function"]["parameters"] == namespace_tool["parameters"]

    def test_namespace_tool_name_map_accepts_unique_unqualified_tool_names(self):
        """Some chat providers return the nested tool name without its namespace."""
        namespace_tool = {
            "type": "namespace",
            "name": "collaboration",
            "tools": [
                {
                    "type": "function",
                    "name": "wait_agent",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        result = LiteLLMCompletionResponsesConfig.namespace_tool_name_map([namespace_tool])

        assert result["collaboration__wait_agent"] == ("collaboration", "wait_agent")
        assert result["wait_agent"] == ("collaboration", "wait_agent")

    def test_namespace_tool_name_map_drops_ambiguous_unqualified_names(self):
        tools = [
            {"type": "function", "name": "ordinary"},
            {
                "type": "namespace",
                "name": "alpha",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
            {
                "type": "namespace",
                "name": "beta",
                "tools": [
                    {"type": "namespace", "name": "ignored"},
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            },
            {
                "type": "namespace",
                "name": "gamma",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        ]

        result = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(tools)

        assert result["alpha__run"] == ("alpha", "run")
        assert result["beta__run"] == ("beta", "run")
        assert result["gamma__run"] == ("gamma", "run")
        assert "run" not in result

    def test_namespace_tool_name_map_drops_top_level_function_collision(self):
        tools = [
            {"type": "function", "name": "run", "parameters": {"type": "object"}},
            {
                "type": "namespace",
                "name": "admin",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ]

        result = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(tools)

        assert result["admin__run"] == ("admin", "run")
        assert "run" not in result

    def test_transform_tools_rejects_flattened_name_collision(self):
        tools = [
            {
                "type": "function",
                "name": "admin__run",
                "parameters": {"type": "object"},
            },
            {
                "type": "namespace",
                "name": "admin",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ]

        with pytest.raises(
            ValueError,
            match="Top-level function names conflict with flattened namespace tools: admin__run",
        ):
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools=tools)

    def test_restore_namespace_tool_name_leaves_unknown_tool_unchanged(self):
        tool_name, namespace = LiteLLMCompletionResponsesConfig._restore_namespace_tool_name(
            "mcp__node_repl",
            {},
        )

        assert tool_name == "mcp__node_repl"
        assert namespace is None

    def test_transform_nested_namespace_ignores_non_function_subtools(self):
        namespace_tool = {
            "type": "namespace",
            "name": "collaboration",
            "tools": [
                "ignored",
                {"type": "namespace", "name": "ignored"},
                {"type": "web_search", "name": "ignored"},
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "parameters": {"properties": {"task_name": {"type": "string"}}},
                },
            ],
        }

        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[namespace_tool]
        )

        assert len(result_tools) == 1
        assert result_tools[0]["function"]["name"] == "collaboration__spawn_agent"
        assert result_tools[0]["function"]["parameters"] == {
            "properties": {"task_name": {"type": "string"}},
            "type": "object",
        }

    def test_transform_nested_namespace_custom_tool_becomes_a_content_function_under_its_short_name(self):
        namespace_tool = {
            "type": "namespace",
            "name": "functions",
            "description": "Codex shell tools.",
            "tools": [
                {
                    "type": "custom",
                    "name": "exec",
                    "description": "Runs a shell command.",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                },
            ],
        }

        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=[namespace_tool]
        )

        assert len(result_tools) == 1
        function = result_tools[0]["function"]
        assert function["name"] == "exec"
        assert function["description"].startswith("Codex shell tools.")
        assert "Runs a shell command." in function["description"]
        assert "start: /.+/" in function["description"]
        assert function["parameters"]["required"] == ["content"]
        assert function["parameters"]["properties"]["content"]["type"] == "string"

    @pytest.mark.parametrize(
        "model, custom_llm_provider",
        [
            ("bedrock/converse/global.anthropic.claude-sonnet-5", "bedrock_converse"),
            ("anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock"),
            ("claude-sonnet-5", "vertex_ai"),
            ("gemini-3.1-pro-preview", "vertex_ai"),
            ("moonshotai.kimi-k2-thinking", "bedrock_mantle"),
        ],
    )
    def test_reasoning_summary_still_yields_a_string_reasoning_effort(self, model, custom_llm_provider):
        """
        A Responses request carrying ``reasoning.summary`` must still reach a chat provider as a
        plain ``reasoning_effort`` string. ``summary`` is Responses-only, and forwarding the whole
        object turns reasoning off: Bedrock Converse and Vertex silently discard a non-string
        ``reasoning_effort``, and Bedrock Mantle rejects the request outright.
        """
        responses_api_request = {"reasoning": {"effort": "medium", "summary": "auto"}}

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider=custom_llm_provider,
        )

        assert result["reasoning_effort"] == "medium"

    @pytest.mark.parametrize(
        "model, custom_llm_provider",
        [
            ("gpt-5.4-pro", "azure_ai"),
            ("gpt-5", "openai"),
            ("gpt-5.1", "openai"),
            ("gpt-5", "azure"),
        ],
    )
    def test_bridged_model_carries_the_summary_as_an_alias(self, model, custom_llm_provider):
        """
        ``summary`` reaches a bridged model through the ``reasoning_summary`` alias, never smuggled
        inside ``reasoning_effort``. ``litellm.completion`` reads that alias back with
        ``peek_reasoning_summary_aliases`` and reassembles ``{effort, summary}``, so the far end
        gets the same object it always did while no chat provider ever sees a non-string effort.
        """
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="hi",
            responses_api_request={"reasoning": {"effort": "medium", "summary": "auto"}},
            custom_llm_provider=custom_llm_provider,
        )

        assert result["reasoning_effort"] == "medium"
        assert result["reasoning_summary"] == "auto"

    @pytest.mark.parametrize(
        "model, custom_llm_provider",
        [
            ("gpt-5", "openai"),
            ("gpt-5.1", "openai"),
            ("gpt-5", "azure"),
        ],
    )
    def test_gpt_5_summary_survives_the_bridge_it_claims_to_take(self, model, custom_llm_provider):
        """
        Regression for the probe disagreeing with the real decision. The transform asked
        ``responses_api_bridge_check`` with ``reasoning_summary`` taken straight off the Responses
        object, but ``litellm.completion`` reads it from ``optional_params`` via
        ``peek_reasoning_summary_aliases``, which the bridged request never populated. So these
        models answered "bridging" to the probe and "not bridging" for real, and the object landed
        on Chat Completions, which only takes a string. Emitting the alias makes the two agree.
        """
        from litellm.main import responses_api_bridge_check
        from litellm.utils import get_optional_params, peek_reasoning_summary_aliases

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="hi",
            responses_api_request={"reasoning": {"effort": "medium", "summary": "auto"}},
            custom_llm_provider=custom_llm_provider,
        )
        optional_params = get_optional_params(
            model=model,
            custom_llm_provider=custom_llm_provider,
            reasoning_effort=result["reasoning_effort"],
            reasoning_summary=result["reasoning_summary"],
        )
        model_info, _ = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            reasoning_effort=result["reasoning_effort"],
            reasoning_summary=peek_reasoning_summary_aliases(optional_params),
        )

        assert model_info.get("mode") == "responses"

    def test_a_failing_bridge_probe_falls_back_to_the_string_effort(self, monkeypatch):
        """
        The probe is a capability question, so a model-info lookup blowing up must not fail the
        request. It degrades to the chat-safe form: a string effort and no alias.
        """
        import litellm.main

        def _boom(**_kwargs):
            raise RuntimeError("model info unavailable")

        monkeypatch.setattr(litellm.main, "responses_api_bridge_check", _boom)

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="gpt-5.4-pro",
            input="hi",
            responses_api_request={"reasoning": {"effort": "medium", "summary": "auto"}},
            custom_llm_provider="azure_ai",
        )

        assert result["reasoning_effort"] == "medium"
        assert "reasoning_summary" not in result

    @pytest.mark.parametrize(
        "reasoning, expected",
        [
            ({"effort": "high"}, "high"),
            ("low", "low"),
            ({"summary": "auto"}, None),
            ({}, None),
            (None, None),
        ],
    )
    def test_reasoning_param_shapes_map_to_reasoning_effort(self, reasoning, expected):
        """
        An object without ``effort`` carries nothing Chat Completions can use, so no
        ``reasoning_effort`` is sent at all (the bridge drops None-valued params).
        """
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            input="hi",
            responses_api_request={"reasoning": reasoning},
            custom_llm_provider="bedrock",
        )

        assert result.get("reasoning_effort") == expected
        assert ("reasoning_effort" in result) is (expected is not None)

    @pytest.mark.parametrize(
        "model, expected_thinking",
        [
            ("global.anthropic.claude-sonnet-5", {"type": "adaptive"}),
            (
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                {"type": "enabled", "budget_tokens": 2048},
            ),
        ],
    )
    def test_reasoning_summary_still_enables_thinking_on_bedrock(self, model, expected_thinking):
        """
        End to end through Bedrock Converse's own param mapping: the effort a Responses request asks
        for must survive into ``thinking``, whether the model takes an adaptive effort or a legacy
        token budget. Forwarding the object instead leaves ``thinking`` unset and the model never
        reasons, which is the failure this guards.
        """
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        bridged = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="hi",
            responses_api_request={"reasoning": {"effort": "medium", "summary": "auto"}},
            custom_llm_provider="bedrock",
        )

        mapped = AmazonConverseConfig().map_openai_params(
            {"reasoning_effort": bridged["reasoning_effort"]}, {}, model, True
        )

        assert expected_thinking.items() <= mapped["thinking"].items()

    def test_bedrock_anthropic_responses_tools_yield_only_function_toolspec(self):
        """
        End-to-end (no network) of the LIT-3858 acceptance criterion: the mixed tools array
        is transformed for a Bedrock Anthropic model, then fed through the Bedrock tool layer.
        The derived web_search_options is dropped, and toolConfig contains only the function
        tool, never the web_search/image_generation/namespace built-ins as junk toolSpecs.
        """
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_tools_pt,
        )

        model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
        responses_api_request = {
            "tools": [
                {
                    "type": "function",
                    "name": "noop",
                    "description": "x",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": "web_search", "external_web_access": False},
                {"type": "image_generation", "output_format": "png"},
                {"type": "namespace", "name": "grp", "description": "g", "tools": []},
            ],
        }

        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="bedrock",
        )

        assert "web_search_options" not in result

        bedrock_tool_blocks = _bedrock_tools_pt(tools=result["tools"], model=model)
        names = [block["toolSpec"]["name"] for block in bedrock_tool_blocks if "toolSpec" in block]
        assert names == ["noop"]
        assert not any(name.startswith("litellm_unnamed_tool_") for name in names)


class TestUsageTransformation:
    """Test cases for usage transformation from Chat Completion to Responses API format"""

    def test_transform_usage_with_cached_tokens_anthropic(self):
        """Test that cached_tokens from Anthropic are properly transformed to input_tokens_details"""
        # Setup: Simulate Anthropic usage with cache_read_input_tokens
        usage = Usage(
            prompt_tokens=13,
            completion_tokens=27,
            total_tokens=40,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=5,  # From Anthropic cache_read_input_tokens
                text_tokens=8,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-sonnet-4",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.input_tokens == 13
        assert response_usage.output_tokens == 27
        assert response_usage.total_tokens == 40
        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 5
        assert response_usage.input_tokens_details.text_tokens == 8
        assert "cache_write_tokens" not in response_usage.input_tokens_details.model_dump()

    def test_transform_usage_with_cached_tokens_gemini(self):
        """Test that cached_tokens from Gemini are properly transformed to input_tokens_details"""
        # Setup: Simulate Gemini usage with cachedContentTokenCount
        usage = Usage(
            prompt_tokens=9,
            completion_tokens=27,
            total_tokens=36,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=3,  # From Gemini cachedContentTokenCount
                text_tokens=6,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-2.0-flash",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.input_tokens == 9
        assert response_usage.output_tokens == 27
        assert response_usage.total_tokens == 36
        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 3
        assert response_usage.input_tokens_details.text_tokens == 6

    def test_transform_usage_preserves_cache_write_tokens(self):
        """Regression for #34801: the chat-completions to Responses bridge dropped
        cache-write tokens, so cache-creation billing disappeared on that route."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=10,
            total_tokens=1010,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=100,
                cache_write_tokens=800,
            ),
        )

        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=usage
        )

        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 100
        assert getattr(response_usage.input_tokens_details, "cache_write_tokens", None) == 800
        assert response_usage.input_tokens_details.model_dump()["cache_write_tokens"] == 800

    def test_transform_usage_preserves_input_modality_tokens(self):
        """Regression: the bridge dropped image and video input tokens.

        Vertex reports prompt tokens split by modality, so a Live session that sends
        camera frames arrives with image_tokens set. InputTokensDetails declared only
        audio/cached/text, so those tokens were folded into text and lost their
        attribution, and any per-modality rate could never apply to them.
        """
        usage = Usage(
            prompt_tokens=300,
            completion_tokens=10,
            total_tokens=310,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=20, audio_tokens=80, image_tokens=150, video_tokens=50, cached_tokens=0
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=10),
        )

        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=usage
        )
        details = response_usage.input_tokens_details
        assert details is not None
        assert getattr(details, "image_tokens", None) == 150
        assert getattr(details, "video_tokens", None) == 50
        assert getattr(details, "audio_tokens", None) == 80

        from litellm.responses.utils import ResponseAPILoggingUtils

        back = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(response_usage.model_dump())
        assert back.prompt_tokens_details.image_tokens == 150
        assert back.prompt_tokens_details.video_tokens == 50

    def test_transform_usage_with_reasoning_tokens_gemini(self):
        """Test that reasoning_tokens from Gemini are properly transformed to output_tokens_details"""
        # Setup: Simulate Gemini usage with thoughtsTokenCount
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=100,
            total_tokens=110,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=50,  # From Gemini thoughtsTokenCount
                text_tokens=50,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-2.0-flash",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.output_tokens == 100
        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 50
        assert response_usage.output_tokens_details.text_tokens == 50

    def test_transform_usage_with_cached_and_reasoning_tokens(self):
        """Test transformation with both cached tokens (input) and reasoning tokens (output)"""
        # Setup: Combined Anthropic cached tokens and Gemini reasoning tokens
        usage = Usage(
            prompt_tokens=13,
            completion_tokens=100,
            total_tokens=113,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=5,  # Anthropic cache_read_input_tokens
                text_tokens=8,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=50,  # Gemini thoughtsTokenCount
                text_tokens=50,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-sonnet-4",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.input_tokens == 13
        assert response_usage.output_tokens == 100
        assert response_usage.total_tokens == 113

        # Verify input_tokens_details
        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 5
        assert response_usage.input_tokens_details.text_tokens == 8

        # Verify output_tokens_details
        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 50
        assert response_usage.output_tokens_details.text_tokens == 50

    def test_transform_usage_with_zero_cached_tokens(self):
        """Test that cached_tokens=0 is properly handled (no cached tokens used)"""
        # Setup: Usage with cached_tokens=0 (no cache hit)
        usage = Usage(
            prompt_tokens=9,
            completion_tokens=27,
            total_tokens=36,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=0,  # No cache hit
                text_tokens=9,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-sonnet-4",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert: Should still include cached_tokens=0 in input_tokens_details
        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 0
        assert response_usage.input_tokens_details.text_tokens == 9

    def test_transform_usage_without_details(self):
        """Test transformation when prompt_tokens_details and completion_tokens_details are None"""
        # Setup: Usage without details (basic usage only)
        usage = Usage(
            prompt_tokens=9,
            completion_tokens=27,
            total_tokens=36,
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gpt-4o",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert: Basic usage should still be transformed, but details should be None
        assert response_usage.input_tokens == 9
        assert response_usage.output_tokens == 27
        assert response_usage.total_tokens == 36
        assert response_usage.input_tokens_details is None
        assert response_usage.output_tokens_details is None

    def test_transform_usage_with_image_tokens(self):
        """Test that image_tokens from Vertex AI/Gemini are properly transformed to output_tokens_details"""
        # Setup: Simulate Vertex AI/Gemini usage with image_tokens in completion_tokens_details
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=150,
            total_tokens=160,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=0,
                text_tokens=50,
                image_tokens=100,  # From Vertex AI candidatesTokensDetails with modality="IMAGE"
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-2.0-flash",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Here is the generated image.", role="assistant"),
                )
            ],
        )

        # Execute
        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.output_tokens == 150
        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 0
        assert response_usage.output_tokens_details.text_tokens == 50
        assert response_usage.output_tokens_details.image_tokens == 100

    def test_reasoning_tokens_fall_back_to_zero_when_absent(self):
        # The OpenAI SDK's ResponseUsage requires output_tokens_details.reasoning_tokens
        # as an int, so an absent count degrades to 0 on the responses wire instead of
        # dropping output_tokens_details and breaking SDK clients.
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                text_tokens=50,
                # reasoning_tokens intentionally absent -> None
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-haiku-4-5",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 0
        assert response_usage.output_tokens_details.text_tokens == 50

    def test_reasoning_tokens_preserved_when_thinking_occurred(self):
        # Regression: reasoning_tokens must survive the chat->responses translation
        # when the provider actually did thinking.
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=612,
            total_tokens=712,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=512,
                text_tokens=100,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="claude-haiku-4-5",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 512

    def test_reasoning_tokens_explicit_zero_preserved(self):
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=0,
                text_tokens=50,
            ),
        )

        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gpt-5.6",
            object="chat.completion",
            usage=usage,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                )
            ],
        )

        response_usage = LiteLLMCompletionResponsesConfig.transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 0


class TestStreamingIDConsistency:
    """Test cases for consistent IDs across streaming events (issue #14962)"""

    def test_streaming_iterator_uses_consistent_item_ids(self):
        """
        Test that all streaming events use the same item_id throughout the stream.
        This fixes the issue where text-start, text-delta, and text-end events
        had different IDs, breaking SDK text accumulation.

        Reproduces: https://github.com/BerriAI/litellm/issues/14962
        """
        from unittest.mock import Mock

        import litellm
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        # Create a mock stream wrapper
        mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_logging_obj = Mock()
        mock_stream_wrapper.logging_obj = mock_logging_obj

        # Create the streaming iterator
        iterator = LiteLLMCompletionStreamingIterator(
            model="gemini/gemini-2.5-flash-lite",
            litellm_custom_stream_wrapper=mock_stream_wrapper,
            request_input="Say Hello World",
            responses_api_request={},
            custom_llm_provider="gemini",
        )

        # Simulate streaming chunks with different IDs (as Gemini does)
        chunk1 = ModelResponseStream(
            id="chatcmpl-first-id",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Hello", role="assistant"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gemini-2.5-flash-lite",
            object="chat.completion.chunk",
        )

        chunk2 = ModelResponseStream(
            id="chatcmpl-second-id",  # Different ID from chunk1
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=" World", role=None),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gemini-2.5-flash-lite",
            object="chat.completion.chunk",
        )

        chunk3 = ModelResponseStream(
            id="chatcmpl-third-id",  # Different ID from chunk1 and chunk2
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="", role=None),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gemini-2.5-flash-lite",
            object="chat.completion.chunk",
        )

        # Transform chunks to response API events
        event1 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk1)
        event2 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk2)
        iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk3)

        # Assert: All events should use the same item_id (from the first chunk)
        assert event1 is not None, "First event should not be None"
        assert event2 is not None, "Second event should not be None"

        # Extract item_ids from events
        item_id_1 = getattr(event1, "item_id", None)
        item_id_2 = getattr(event2, "item_id", None)

        assert item_id_1 is not None, "First event should have an item_id"
        assert item_id_2 is not None, "Second event should have an item_id"

        # The critical assertion: IDs should match across all events
        assert item_id_1 == item_id_2, (
            f"Item IDs should be consistent across streaming events. "
            f"Got {item_id_1} and {item_id_2}. "
            f"This breaks SDK text accumulation (issue #14962)."
        )

        # Verify the cached ID is set and matches
        assert iterator._cached_item_id is not None, "Iterator should cache the item_id"
        assert iterator._cached_item_id == item_id_1, "Cached ID should match event IDs"
        assert iterator._cached_item_id.startswith("msg_"), (
            "Message item IDs must use the Responses API msg_ prefix (issue #27333)"
        )

    def test_streaming_iterator_initial_events_use_cached_id(self):
        """
        Test that initial events (output_item_added, content_part_added) also use the cached ID.
        """
        from unittest.mock import Mock

        import litellm
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        # Create a mock stream wrapper
        mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_logging_obj = Mock()
        mock_stream_wrapper.logging_obj = mock_logging_obj

        # Create the streaming iterator
        iterator = LiteLLMCompletionStreamingIterator(
            model="gemini/gemini-2.5-flash-lite",
            litellm_custom_stream_wrapper=mock_stream_wrapper,
            request_input="Test",
            responses_api_request={},
        )

        # Create initial events
        output_item_event = iterator.create_output_item_added_event()
        content_part_event = iterator.create_content_part_added_event()

        # Extract IDs
        output_item_id = getattr(output_item_event.item, "id", None)
        content_part_id = getattr(content_part_event, "item_id", None)

        # Assert: Both should use the same cached ID
        assert output_item_id is not None, "Output item should have an ID"
        assert content_part_id is not None, "Content part should have an item_id"
        assert output_item_id == content_part_id, (
            f"Initial events should use consistent IDs. "
            f"Got output_item_id={output_item_id}, content_part_id={content_part_id}"
        )

        # Verify it matches the cached ID
        assert iterator._cached_item_id is not None
        assert iterator._cached_item_id == output_item_id

    def test_streaming_iterator_done_events_use_cached_id(self):
        """
        Test that done events (output_text_done, content_part_done, output_item_done) use the cached ID.
        """
        from unittest.mock import Mock

        import litellm
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )
        from litellm.types.utils import Choices, Message, ModelResponse

        # Create a mock stream wrapper
        mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_logging_obj = Mock()
        mock_stream_wrapper.logging_obj = mock_logging_obj
        mock_logging_obj.response_cost_calculator = Mock(return_value=0.001)

        # Create the streaming iterator
        iterator = LiteLLMCompletionStreamingIterator(
            model="gemini/gemini-2.5-flash-lite",
            litellm_custom_stream_wrapper=mock_stream_wrapper,
            request_input="Test",
            responses_api_request={},
        )

        # Set up a complete model response
        complete_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="gemini-2.5-flash-lite",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello World", role="assistant"),
                )
            ],
        )
        iterator.litellm_model_response = complete_response

        # Create done events
        text_done_event = iterator.create_output_text_done_event(complete_response)
        content_done_event = iterator.create_output_content_part_done_event(complete_response)
        item_done_event = iterator.create_output_item_done_event(complete_response)

        # Extract IDs
        text_done_id = getattr(text_done_event, "item_id", None)
        content_done_id = getattr(content_done_event, "item_id", None)
        item_done_id = getattr(item_done_event.item, "id", None)

        # Assert: All done events should use the same cached ID
        assert text_done_id is not None, "Text done event should have an item_id"
        assert content_done_id is not None, "Content done event should have an item_id"
        assert item_done_id is not None, "Item done event should have an id"

        assert text_done_id == content_done_id == item_done_id, (
            f"All done events should use consistent IDs. "
            f"Got text_done={text_done_id}, content_done={content_done_id}, item_done={item_done_id}"
        )

        # Verify it matches the cached ID
        assert iterator._cached_item_id is not None
        assert iterator._cached_item_id == text_done_id

    def test_parallel_tool_calls_merged_into_single_assistant_message(self):
        """
        Regression test: multi-turn parallel tool calls via the Responses API must
        produce a single assistant message with all tool_calls, not one assistant
        message per function_call item.

        When the model responds with two parallel tool calls (e.g. get_weather for
        SF and NYC), the next Responses API request includes two consecutive
        function_call items followed by two function_call_output items.

        Without the fix each function_call becomes its own assistant message,
        producing back-to-back assistant messages that Anthropic/Vertex AI rejects:
        "tool_use ids were found without tool_result blocks immediately after".
        """
        input_items = [
            {"type": "message", "role": "user", "content": "Weather in SF and NYC?"},
            # Two parallel tool calls from the previous assistant response
            {
                "type": "function_call",
                "call_id": "toolu_01",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {
                "type": "function_call",
                "call_id": "toolu_02",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            # Tool results
            {"type": "function_call_output", "call_id": "toolu_01", "output": "72°F"},
            {"type": "function_call_output", "call_id": "toolu_02", "output": "55°F"},
        ]

        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=input_items
        )

        roles = [m.get("role") if isinstance(m, dict) else getattr(m, "role", None) for m in messages]

        # Must not have two consecutive assistant messages
        for i in range(len(roles) - 1):
            assert not (roles[i] == "assistant" and roles[i + 1] == "assistant"), (
                f"Consecutive assistant messages at indices {i} and {i + 1}: {roles}"
            )

        # The single assistant message must contain BOTH tool_calls
        assistant_messages = [
            m for m in messages if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"
        ]
        assert len(assistant_messages) == 1, f"Expected 1 assistant message, got {len(assistant_messages)}"

        assistant_msg = assistant_messages[0]
        tool_calls = (
            assistant_msg.get("tool_calls")
            if isinstance(assistant_msg, dict)
            else getattr(assistant_msg, "tool_calls", None)
        )
        assert tool_calls is not None and len(tool_calls) == 2, (
            f"Expected 2 tool_calls in the merged assistant message, got: {tool_calls}"
        )

        call_ids = [(tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)) for tc in tool_calls]
        assert "toolu_01" in call_ids, f"toolu_01 missing from tool_calls: {call_ids}"
        assert "toolu_02" in call_ids, f"toolu_02 missing from tool_calls: {call_ids}"

        # Both tool messages must be present
        tool_messages = [
            m for m in messages if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "tool"
        ]
        assert len(tool_messages) == 2, f"Expected 2 tool messages, got {len(tool_messages)}"

    def test_single_tool_call_still_works_after_merge_fix(self):
        """
        Ensure the parallel-tool-call merging fix does not break the existing
        single-tool-call path.
        """
        input_items = [
            {"type": "message", "role": "user", "content": "Weather in SF?"},
            {
                "type": "function_call",
                "call_id": "toolu_01",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {"type": "function_call_output", "call_id": "toolu_01", "output": "72°F"},
        ]

        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=input_items
        )

        roles = [m.get("role") if isinstance(m, dict) else getattr(m, "role", None) for m in messages]

        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

        assistant_messages = [
            m for m in messages if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"
        ]
        assert len(assistant_messages) == 1

        tool_calls = (
            assistant_messages[0].get("tool_calls")
            if isinstance(assistant_messages[0], dict)
            else getattr(assistant_messages[0], "tool_calls", None)
        )
        assert tool_calls is not None and len(tool_calls) == 1


class TestCompletedResponseLatchedOnStreamEnd:
    """Regression: LiteLLMCompletionStreamingIterator (the Chat Completions
    bridge path) never set ``self.completed_response`` because it overrides
    __anext__ and bypasses the base class's _process_chunk where that
    attribute is normally latched. FallbackResponsesStreamWrapper reads
    ``completed_response`` via getattr to record container ownership; when
    it stays None the proxy logs a "Container ownership recording skipped"
    warning and follow-up /v1/containers/<id>/files calls 403 for non-admin
    keys. Codex CLI's apply_patch tool also surfaces as "aborted" because
    the terminal response.completed event never propagates correctly."""

    def _make_iterator_with_stop(self, model_response):
        """Build a LiteLLMCompletionStreamingIterator whose underlying
        CustomStreamWrapper raises StopAsyncIteration immediately (simulating
        a stream that already delivered all content chunks)."""
        from unittest.mock import Mock

        import litellm
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        mock_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_wrapper.logging_obj = Mock()
        mock_wrapper.logging_obj.response_cost_calculator = Mock(return_value=0.0)
        mock_wrapper.__aiter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__anext__ = Mock(side_effect=StopAsyncIteration)

        iterator = LiteLLMCompletionStreamingIterator(
            model="deepseek/deepseek-chat",
            litellm_custom_stream_wrapper=mock_wrapper,
            request_input="test",
            responses_api_request={},
        )
        iterator.litellm_model_response = model_response
        return iterator

    def test_completed_response_set_after_common_done_event_logic(self):
        """common_done_event_logic builds a ResponseCompletedEvent and must
        latch it onto self.completed_response so downstream wrappers can
        read it. Before the fix the event was returned but
        completed_response stayed None."""
        from litellm.types.utils import Choices, Message, ModelResponse

        complete_response = ModelResponse(
            id="resp_test",
            created=1234567890,
            model="deepseek-chat",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="hello", role="assistant"),
                )
            ],
        )
        iterator = self._make_iterator_with_stop(complete_response)

        import asyncio

        async def drain():
            results = []
            try:
                async for chunk in iterator:
                    results.append(chunk)
            except StopAsyncIteration:
                pass
            return results

        results = asyncio.run(drain())
        assert len(results) > 0
        assert iterator.completed_response is not None, (
            "LiteLLMCompletionStreamingIterator.completed_response is still None "
            "after common_done_event_logic ran — downstream wrappers and the "
            "proxy container-ownership hook will see no terminal event"
        )
        assert iterator.completed_response.type == "response.completed"


class TestFallbackWrapperStopAsyncIterationFallback:
    """Regression: FallbackResponsesStreamWrapper.__anext__ only sniffed
    terminal events off forwarded chunks. When the inner generator raises
    StopAsyncIteration without a sniffable chunk (the bridge path ends this
    way), the wrapper re-raised without checking source_iterator for a
    latched completed_response, leaving its own completed_response None."""

    def test_falls_back_to_source_completed_response_on_stop(self):
        """When the inner async generator raises StopAsyncIteration and the
        wrapper never sniffed a terminal chunk, it must copy
        source_iterator.completed_response so the proxy ownership hook
        still sees the terminal event."""
        import asyncio
        from types import SimpleNamespace

        from litellm.router import Router

        source = SimpleNamespace(
            response=None,
            model="deepseek/deepseek-chat",
            logging_obj=None,
            responses_api_provider_config=None,
            start_time=None,
            litellm_metadata=None,
            custom_llm_provider="deepseek",
            request_data={},
            call_type="aresponses",
            _hidden_params={},
            completed_response=SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_src", output=[], container=None),
            ),
        )

        async def empty_gen():
            return
            yield  # pragma: no cover

        async def _drive():
            router = Router(
                model_list=[
                    {
                        "model_name": "deepseek/deepseek-chat",
                        "litellm_params": {"model": "deepseek/deepseek-chat", "api_key": "sk-test"},
                    }
                ]
            )
            wrapper = await router._aresponses_streaming_iterator(
                response=source,  # type: ignore[arg-type]
                initial_kwargs={},
            )
            wrapper._async_generator = empty_gen()
            out = []
            try:
                async for chunk in wrapper:
                    out.append(chunk)
            except StopAsyncIteration:
                pass
            return wrapper, out

        wrapper, _ = asyncio.run(_drive())
        assert wrapper.completed_response is not None, (
            "FallbackResponsesStreamWrapper.completed_response is None after "
            "StopAsyncIteration even though source_iterator had one — the "
            "proxy ownership hook will log a spurious warning"
        )
        assert wrapper.completed_response.type == "response.completed"


class TestEnsureOutputItemContentPartAdded:
    """Test that _ensure_output_item_for_chunk emits content_part.added after
    output_item.added for message items."""

    def _make_iterator(self):
        """Create a minimal LiteLLMCompletionStreamingIterator for testing."""
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        iterator = LiteLLMCompletionStreamingIterator.__new__(LiteLLMCompletionStreamingIterator)
        iterator.sent_output_item_added_event = False
        iterator.sent_content_part_added_event = False
        iterator._sequence_number = 0
        iterator._cached_item_id = None
        iterator._cached_reasoning_item_id = None
        iterator._reasoning_active = False
        iterator._pending_response_events = []
        iterator._pending_tool_events = []
        iterator._tool_output_index_by_call_id = {}
        iterator._tool_args_by_call_id = {}
        iterator._tool_item_id_by_call_id = {}
        iterator._tool_call_id_by_index = {}
        iterator._ambiguous_tool_call_indexes = set()
        iterator._next_output_index = 0
        iterator._message_output_index = None
        iterator._reasoning_output_index = None
        iterator._final_tool_events_queued = False
        iterator._custom_tool_names = set()
        iterator.responses_api_request = {}
        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(None)
        iterator._web_search_calls = {}
        iterator._queued_web_search_call_ids = set()
        iterator._tool_search_requested = False
        iterator._tool_search_call_ids = set()
        return iterator

    def _make_text_chunk(self):
        """Create a mock ModelResponseStream with a text delta."""
        from unittest.mock import MagicMock

        chunk = MagicMock()
        delta = MagicMock()
        delta.reasoning_content = None
        delta.tool_calls = None
        chunk.choices = [MagicMock(delta=delta)]
        return chunk

    def _make_reasoning_chunk(self):
        """Create a mock ModelResponseStream with a reasoning delta."""
        from unittest.mock import MagicMock

        chunk = MagicMock()
        delta = MagicMock()
        delta.reasoning_content = "thinking..."
        delta.tool_calls = None
        chunk.choices = [MagicMock(delta=delta)]
        return chunk

    def test_message_item_emits_content_part_added(self):
        """content_part.added must follow output_item.added for message items."""
        from litellm.types.llms.openai import (
            ContentPartAddedEvent,
            OutputItemAddedEvent,
            ResponsesAPIStreamEvents,
        )

        iterator = self._make_iterator()
        chunk = self._make_text_chunk()

        iterator._ensure_output_item_for_chunk(chunk)

        events = iterator._pending_response_events
        assert len(events) == 2
        assert isinstance(events[0], OutputItemAddedEvent)
        assert events[0].type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert isinstance(events[1], ContentPartAddedEvent)
        assert events[1].type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
        assert events[1].part.type == "output_text"
        assert iterator.sent_content_part_added_event is True

    def test_streaming_namespace_tool_calls_restore_responses_namespace(self):
        """Flattened chat-completion namespace tool calls must stream back as
        Responses function calls with name + namespace split."""
        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            ]
        }

        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )

        iterator._queue_tool_call_delta_events(
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "collaboration__spawn_agent",
                        "arguments": '{"task_name":"input_test"}',
                    },
                }
            ]
        )

        added = iterator._pending_tool_events[0]
        assert added.item.name == "spawn_agent"
        assert added.item.namespace == "collaboration"

    def test_streaming_nested_custom_tool_call_comes_back_as_custom_tool_call(self):
        from litellm.responses.litellm_completion_transformation.custom_tools import extract_custom_tool_names

        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "exec",
                            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                        }
                    ],
                }
            ]
        }
        iterator._custom_tool_names = extract_custom_tool_names(iterator.responses_api_request.get("tools"))
        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )

        iterator._queue_tool_call_delta_events(
            [{"index": 0, "id": "call_exec", "function": {"name": "exec", "arguments": '{"content":"ls"}'}}]
        )
        iterator._queue_final_tool_call_done_events(
            ModelResponse(
                id="chatcmpl-exec",
                created=1,
                model="us.openai.gpt-5.6",
                object="chat.completion",
                choices=[
                    Choices(
                        finish_reason="tool_calls",
                        index=0,
                        message=Message(
                            content=None,
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_exec",
                                    type="function",
                                    function=Function(name="exec", arguments='{"content":"ls"}'),
                                )
                            ],
                        ),
                    )
                ],
            )
        )

        added = iterator._pending_tool_events[0]
        assert added.item.type == "custom_tool_call"
        assert added.item.name == "exec"
        done = iterator._pending_tool_events[-1]
        assert done.item.type == "custom_tool_call"
        assert done.item.input == "ls"

    def test_streaming_namespaced_function_sharing_a_nested_custom_short_name_stays_a_function_call(self):
        from litellm.responses.litellm_completion_transformation.custom_tools import extract_custom_tool_names

        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "alpha",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "run",
                            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                        }
                    ],
                },
                {
                    "type": "namespace",
                    "name": "beta",
                    "tools": [
                        {
                            "type": "function",
                            "name": "run",
                            "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}},
                        }
                    ],
                },
            ]
        }
        iterator._custom_tool_names = extract_custom_tool_names(iterator.responses_api_request.get("tools"))
        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )
        function_call = {"id": "call_fn", "function": {"name": "beta__run", "arguments": '{"job_id":"42"}'}}
        custom_call = {"id": "call_custom", "function": {"name": "run", "arguments": '{"content":"echo hi"}'}}

        iterator._queue_tool_call_delta_events([{"index": 0, **function_call}, {"index": 1, **custom_call}])
        iterator._queue_final_tool_call_done_events(
            ModelResponse(
                id="chatcmpl-run",
                created=1,
                model="us.openai.gpt-5.6",
                object="chat.completion",
                choices=[
                    Choices(
                        finish_reason="tool_calls",
                        index=0,
                        message=Message(
                            content=None,
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id=call["id"], type="function", function=Function(**call["function"])
                                )
                                for call in (function_call, custom_call)
                            ],
                        ),
                    )
                ],
            )
        )

        items = [
            event.item
            for event in iterator._pending_tool_events
            if event.type in ("response.output_item.added", "response.output_item.done")
        ]
        function_items = [item for item in items if item.call_id == "call_fn"]
        custom_items = [item for item in items if item.call_id == "call_custom"]
        assert len(function_items) == 2 and len(custom_items) == 2
        assert all(
            (item.type, item.name, item.namespace) == ("function_call", "run", "beta") for item in function_items
        )
        assert function_items[-1].arguments == '{"job_id":"42"}'
        assert all(item.type == "custom_tool_call" and item.name == "run" for item in custom_items)
        assert all(getattr(item, "namespace", None) is None for item in custom_items)
        assert custom_items[-1].input == "echo hi"

    def test_streaming_unqualified_namespace_tool_calls_restore_namespace(self):
        """A unique nested tool name without the namespace still maps back."""
        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "wait_agent",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            ]
        }

        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )

        iterator._queue_tool_call_delta_events(
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "wait_agent",
                        "arguments": "{}",
                    },
                }
            ]
        )

        added = iterator._pending_tool_events[0]
        assert added.item.name == "wait_agent"
        assert added.item.namespace == "collaboration"

    def test_streaming_flat_namespace_tool_call_keeps_flat_name(self):
        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__node_repl",
                    "description": "Run JavaScript",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                }
            ]
        }

        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )

        iterator._queue_tool_call_delta_events(
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "mcp__node_repl",
                        "arguments": '{"code":"1+1"}',
                    },
                }
            ]
        )

        chat_completion_response = ModelResponse(
            id="chatcmpl-test",
            created=1234567890,
            model="gemini-3.1-pro-preview-customtools",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_1",
                                type="function",
                                function=Function(
                                    name="mcp__node_repl",
                                    arguments='{"code":"1+1"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        iterator._queue_final_tool_call_done_events(chat_completion_response)

        added = iterator._pending_tool_events[0]
        done = iterator._pending_tool_events[-1]
        assert added.item.name == "mcp__node_repl"
        assert getattr(added.item, "namespace", None) is None
        assert done.item.name == "mcp__node_repl"
        assert getattr(done.item, "namespace", None) is None

    def test_streaming_final_only_namespace_tool_call_restores_namespace(self):
        from unittest.mock import MagicMock

        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            ]
        }
        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )

        message = MagicMock()
        message.tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "collaboration__spawn_agent",
                    "arguments": '{"message":"hello world"}',
                },
            }
        ]
        complete_response = MagicMock()
        complete_response.choices = [MagicMock(message=message)]

        iterator._queue_final_tool_call_done_events(complete_response)

        added = iterator._pending_tool_events[0]
        delta_events = iterator._pending_tool_events[1:-2]
        done = iterator._pending_tool_events[-1]
        assert added.item.name == "spawn_agent"
        assert added.item.namespace == "collaboration"
        assert "".join(event.delta for event in delta_events) == '{"message":"hello world"}'
        assert done.item.name == "spawn_agent"
        assert done.item.namespace == "collaboration"

    def test_streaming_top_level_function_collision_stays_unnamespaced(self):
        iterator = self._make_iterator()
        iterator.responses_api_request = {
            "tools": [
                {"type": "function", "name": "run", "parameters": {"type": "object"}},
                {
                    "type": "namespace",
                    "name": "admin",
                    "tools": [
                        {
                            "type": "function",
                            "name": "run",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            ]
        }

        iterator._namespace_tool_names = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
            iterator.responses_api_request.get("tools")
        )

        iterator._queue_tool_call_delta_events(
            [
                {
                    "index": 0,
                    "id": "call_top_level",
                    "function": {"name": "run", "arguments": "{}"},
                }
            ]
        )

        added = iterator._pending_tool_events[0]
        assert added.item.name == "run"
        assert getattr(added.item, "namespace", None) is None

    def test_streaming_namespace_map_is_built_once(self):
        from unittest.mock import MagicMock, patch

        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        mock_stream_wrapper = MagicMock()
        mock_stream_wrapper.logging_obj = MagicMock()
        request = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "admin",
                    "tools": [
                        {
                            "type": "function",
                            "name": "run",
                            "parameters": {"type": "object"},
                        }
                    ],
                }
            ]
        }

        with patch.object(
            LiteLLMCompletionResponsesConfig,
            "namespace_tool_name_map",
            wraps=LiteLLMCompletionResponsesConfig.namespace_tool_name_map,
        ) as namespace_map:
            iterator = LiteLLMCompletionStreamingIterator(
                model="test-model",
                litellm_custom_stream_wrapper=mock_stream_wrapper,
                request_input="test",
                responses_api_request=request,
            )
            iterator._responses_namespace_tool_call_fields("admin__run")
            iterator._responses_namespace_tool_call_fields("admin__run")

        namespace_map.assert_called_once_with(request["tools"])

    def test_emit_terminal_event_uses_stream_finish_reason(self):
        """
        The terminal stream event type must track the assembled response
        status: finish_reason="content_filter" yields response.incomplete,
        finish_reason="stop" yields response.completed.
        """
        from unittest.mock import Mock

        import litellm
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )
        from litellm.types.llms.openai import ResponsesAPIStreamEvents

        mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_stream_wrapper.logging_obj = Mock()

        iterator = LiteLLMCompletionStreamingIterator(
            model="anthropic/claude-sonnet-4-6",
            litellm_custom_stream_wrapper=mock_stream_wrapper,
            request_input="test",
            responses_api_request={},
            custom_llm_provider="anthropic",
        )

        def _response(finish_reason: str) -> ModelResponse:
            return ModelResponse(
                id="chatcmpl-test",
                created=1234567890,
                model="anthropic/claude-sonnet-4-6",
                object="chat.completion",
                choices=[
                    Choices(
                        finish_reason=finish_reason,
                        index=0,
                        message=Message(content="Partial answer", role="assistant"),
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
            )

        incomplete_event = iterator._emit_terminal_response_event(_response("content_filter"))
        assert incomplete_event is not None
        assert incomplete_event.type == ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
        assert incomplete_event.response.status == "incomplete"
        assert incomplete_event.response.output[0].status == "incomplete"

        completed_event = iterator._emit_terminal_response_event(_response("stop"))
        assert completed_event is not None
        assert completed_event.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        assert completed_event.response.status == "completed"

    def test_reasoning_item_does_not_emit_content_part_added(self):
        """Reasoning items should not get a content_part.added event."""
        from litellm.types.llms.openai import OutputItemAddedEvent

        iterator = self._make_iterator()
        chunk = self._make_reasoning_chunk()

        iterator._ensure_output_item_for_chunk(chunk)

        events = iterator._pending_response_events
        assert len(events) == 1
        assert isinstance(events[0], OutputItemAddedEvent)
        assert iterator.sent_content_part_added_event is False

    def test_only_emits_once(self):
        """Calling _ensure_output_item_for_chunk twice should not duplicate events."""
        iterator = self._make_iterator()
        chunk = self._make_text_chunk()

        iterator._ensure_output_item_for_chunk(chunk)
        iterator._ensure_output_item_for_chunk(chunk)

        events = iterator._pending_response_events
        assert len(events) == 2


class TestCacheControlPreservation:
    def test_cache_control_preserved_in_content_transformation(self):
        """cache_control injected by AnthropicCacheControlHook must survive
        the Responses API -> Chat Completion content transformation."""
        content = [
            {
                "type": "text",
                "text": "hello",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_content_without_cache_control_unaffected(self):
        """Content blocks that don't have cache_control should be unaffected."""
        content = [{"type": "text", "text": "hello"}]
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "cache_control" not in result[0]

    def test_cache_control_preserved_in_input_item_transformation(self):
        """cache_control survives the full input-item -> messages transformation."""
        input_item = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "long context",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item
        )
        assert len(messages) == 1
        msg_content = (
            messages[0].get("content") if isinstance(messages[0], dict) else getattr(messages[0], "content", None)
        )
        assert isinstance(msg_content, list)
        assert msg_content[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_preserved_for_input_file_block(self):
        content = [
            {
                "type": "input_file",
                "file_id": "file-abc123",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_preserved_for_input_image_block(self):
        content = [
            {
                "type": "input_image",
                "image_url": "https://example.com/img.png",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_function_call_tool_id_falls_back_to_unique_id_for_degenerate_call_id():
    """Bedrock Mantle returns a non-unique, index-based ``call_id`` (``call_0`` that
    resets every response) alongside a unique ``id`` (``fc_...``). For that degenerate
    form the converter must expose the unique ``id``; otherwise every tool call across
    an agent's turns collapses to the same id, the agent cannot correlate its tool
    results, and it loops re-issuing the same call. A normal (unique) ``call_id`` must
    be preserved, since it is the canonical Responses API correlation key. Regression
    for the bedrock-mantle gpt-5.5 non-streaming path."""
    from types import SimpleNamespace

    convert = LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call

    mantle = SimpleNamespace(id="fc_unique_abc123", call_id="call_0", name="get_weather", arguments="{}")
    assert convert(mantle)["id"] == "fc_unique_abc123"

    openai = SimpleNamespace(id="fc_2", call_id="call_tokyo", name="get_weather", arguments="{}")
    assert convert(openai)["id"] == "call_tokyo"


class TestHostedWebSearchReplay:
    def test_emitted_hosted_search_output_round_trips_with_client_tool_result(self) -> None:
        search_result: Final = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_round_trip_search",
            "content": [{"type": "web_search_result", "url": "https://example.com/forecast"}],
        }
        search: Final = build_web_search_call(
            tool_id="srvtoolu_round_trip_search", tool_input={"query": "Paris forecast"}, result=search_result
        )
        message: Final = Message(
            role="assistant",
            content="I found a forecast source.",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="srvtoolu_round_trip_search",
                    type="function",
                    function=Function(name="web_search", arguments='{"query":"Paris forecast"}'),
                ),
                ChatCompletionMessageToolCall(
                    id="call_round_trip_weather",
                    type="function",
                    function=Function(name="get_weather", arguments='{"city":"Paris"}'),
                ),
            ],
            provider_specific_fields={"web_search_calls": [search], "web_search_results": [search_result]},
        )
        response: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Find a forecast source and check the weather in Paris.",
            responses_api_request={
                "tools": [
                    {"type": "web_search"},
                    {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                ]
            },
            chat_completion_response=_bridged_chat_completion_response(
                choices=[Choices(index=0, finish_reason="tool_calls", message=message)]
            ),
        )
        assert [item for item in response.output if item.type == "web_search_call"] == [search]
        assert [item.call_id for item in response.output if item.type == "function_call"] == ["call_round_trip_weather"]
        history: Final = [
            {"role": "user", "content": "Find a forecast source and check the weather in Paris."},
            *(item.model_dump(exclude_none=True) for item in response.output),
            {"type": "function_call_output", "call_id": "call_round_trip_weather", "output": "Paris is sunny."},
        ]
        original: Final = deepcopy(history)

        messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=history, responses_api_request={}
        )

        assert [item.get("role") for item in messages] == ["user", "assistant", "tool"]
        assistant: Final = messages[1]
        assert [call["id"] for call in assistant["tool_calls"]] == ["call_round_trip_weather"]
        assert [call["function"]["name"] for call in assistant["tool_calls"]] == ["get_weather"]
        content: Final = assistant["content"]
        assert isinstance(content, list)
        text_parts: Final = tuple(block["text"] for block in content if block.get("type") == "text")
        assert text_parts[0] == "I found a forecast source."
        replayed_searches: Final = tuple(
            ResponseFunctionWebSearch.model_validate_json(text[text.index("{") :])
            for text in text_parts
            if "web_search_call" in text
        )
        assert replayed_searches == (search,)
        assert messages[2]["tool_call_id"] == "call_round_trip_weather"
        assert messages[2]["content"] == "Paris is sunny."
        assert history == original

    @pytest.mark.parametrize(
        "action",
        (
            ActionSearch(
                type="search",
                query="hosted search history",
                queries=["hosted search history", "search replay"],
                sources=[ActionSearchSource(type="url", url="https://example.com/search-result")],
            ),
            ActionOpenPage(type="open_page", url="https://example.com/opened-page"),
            ActionFind(type="find_in_page", url="https://example.com/find-page", pattern="search history"),
        ),
        ids=("search", "open_page", "find"),
    )
    @pytest.mark.parametrize("status", ("completed", "failed"))
    def test_replays_typed_search_action_without_client_tool_call(
        self,
        action: ActionSearch | ActionOpenPage | ActionFind,
        status: Literal["completed", "failed"],
    ) -> None:
        search: Final = ResponseFunctionWebSearch(
            id="ws_replayed_search", type="web_search_call", status=status, action=action
        )
        input_item: Final = search.model_dump(exclude_none=True)
        original: Final = deepcopy(input_item)

        messages: Final = (
            LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
                input_item=input_item
            )
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert not messages[0].get("tool_calls")
        content: Final = messages[0].get("content")
        assert isinstance(content, str)
        replayed: Final = ResponseFunctionWebSearch.model_validate_json(content[content.index("{") :])
        assert replayed == search
        assert input_item == original

    @pytest.mark.parametrize("order", ((0, 1, 2, 3), (1, 0, 3, 2), (1, 3, 0, 2)))
    @pytest.mark.parametrize("modify_params", (False, True))
    @pytest.mark.parametrize("structured_content", (False, True))
    def test_search_replay_preserves_client_tool_result_adjacency(
        self,
        monkeypatch: pytest.MonkeyPatch,
        order: tuple[int, int, int, int],
        modify_params: bool,
        structured_content: bool,
    ) -> None:
        monkeypatch.setattr(litellm, "modify_params", modify_params)
        searches: Final = tuple(
            ResponseFunctionWebSearch(
                id=f"ws_search_{index}",
                type="web_search_call",
                status="completed",
                action=ActionSearch(
                    type="search",
                    query=f"search query {index}",
                    queries=[f"search query {index}"],
                    sources=[ActionSearchSource(type="url", url=f"https://example.com/result-{index}")],
                ),
            )
            for index in (1, 2)
        )
        replay_items: Final = (
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_weather",
                "arguments": '{"city":"Paris"}',
            },
            searches[0].model_dump(exclude_none=True),
            {"type": "function_call", "name": "get_time", "call_id": "call_time", "arguments": "{}"},
            searches[1].model_dump(exclude_none=True),
        )
        history: Final = [
            {"role": "user", "content": "Research the forecast and call get_weather."},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will check the forecast."}]
                if structured_content
                else "I will check the forecast.",
            },
            *(replay_items[index] for index in order),
            {"role": "assistant", "content": [{"type": "output_text", "text": "I found two sources."}]},
            {"type": "function_call_output", "call_id": "call_weather", "output": "Paris is sunny."},
            {"type": "function_call_output", "call_id": "call_time", "output": "12:00"},
        ]
        original: Final = deepcopy(history)

        messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=history, responses_api_request={}
        )

        assert [message.get("role") for message in messages] == ["user", "assistant", "tool", "tool"]
        assistant: Final = messages[1]
        assert [call["id"] for call in assistant["tool_calls"]] == ["call_weather", "call_time"]
        assert [call["function"]["name"] for call in assistant["tool_calls"]] == ["get_weather", "get_time"]
        assert messages[2]["tool_call_id"] == "call_weather"
        assert messages[2]["content"] == "Paris is sunny."
        assert messages[3]["tool_call_id"] == "call_time"
        assert messages[3]["content"] == "12:00"
        content: Final = assistant["content"]
        assert isinstance(content, list)
        text_parts: Final = tuple(block["text"] for block in content if block.get("type") == "text")
        assert text_parts[0] == "I will check the forecast."
        assert text_parts[-1] == "I found two sources."
        replayed_searches: Final = tuple(
            ResponseFunctionWebSearch.model_validate_json(text[text.index("{") :])
            for text in text_parts
            if "web_search_call" in text
        )
        assert replayed_searches == searches
        assert history == original

        provider_messages: Final = anthropic_messages_pt(
            messages=messages, model="claude-fable-5-1", llm_provider="anthropic"
        )

        assert [message["role"] for message in provider_messages] == ["user", "assistant", "user"]
        assistant_blocks: Final = provider_messages[1]["content"]
        result_blocks: Final = provider_messages[2]["content"]
        assert [block["id"] for block in assistant_blocks if block.get("type") == "tool_use"] == [
            "call_weather",
            "call_time",
        ]
        assert [block["tool_use_id"] for block in result_blocks if block.get("type") == "tool_result"] == [
            "call_weather",
            "call_time",
        ]
        assert [block["content"] for block in result_blocks if block.get("type") == "tool_result"] == [
            "Paris is sunny.",
            "12:00",
        ]
        assert [block["text"] for block in assistant_blocks if block.get("type") == "text"] == list(text_parts)
        assert history == original


BRIDGED_CHAT_COMPLETION_ID = "chatcmpl-dfa2da3a-1586-4ff7-b64e-f59c692a5d11"


def _bridged_chat_completion_response(**overrides):
    defaults = dict(
        id=BRIDGED_CHAT_COMPLETION_ID,
        created=1717000000,
        model="claude-sonnet-4-5",
        object="chat.completion",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="apple"),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    defaults.update(overrides)
    return ModelResponse(**defaults)


def _bridged_output_items(response, item_type):
    return [item for item in response.output if getattr(item, "type", None) == item_type]


class TestBridgedOutputItemIdPrefixes:
    """Bridged output items must carry Responses API ID prefixes (issue #27333).

    Native OpenAI Responses rejects a replayed history whose message item ID does not
    begin with "msg", so leaking the upstream chatcmpl-* ID makes the conversation
    impossible to hand off from a bridged provider to OpenAI.
    """

    def _transform(self, chat_completion_response):
        return LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Say the single word: apple",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

    @pytest.mark.parametrize(
        "tool_type,result_kind,expected_sources",
        [
            (
                "web_search",
                "valid",
                {"srvtoolu_01Search": ["https://example.com/one"], "srvtoolu_02Search": ["https://example.com/two"]},
            ),
            (
                "web_search_preview",
                "valid",
                {"srvtoolu_01Search": ["https://example.com/one"], "srvtoolu_02Search": ["https://example.com/two"]},
            ),
            ("function", "valid", {}),
            ("web_search", "unpaired", {"srvtoolu_01Search": ["https://example.com/one"]}),
            ("web_search", "web_fetch", {"srvtoolu_02Search": ["https://example.com/two"]}),
            ("web_search", "error", {"srvtoolu_01Search": [], "srvtoolu_02Search": ["https://example.com/two"]}),
        ],
    )
    def test_anthropic_web_search_output_mapping(self, tool_type, result_kind, expected_sources):
        call_ids: Final = ("srvtoolu_01Search", "srvtoolu_02Search")
        valid_results: Final = (
            {
                "type": "web_search_tool_result",
                "tool_use_id": call_ids[0],
                "content": [{"type": "web_search_result", "url": "https://example.com/one"}],
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": call_ids[1],
                "content": [{"type": "web_search_result", "url": "https://example.com/two"}],
            },
        )
        first_result: Final = (
            {**valid_results[0], "type": "web_fetch_tool_result"}
            if result_kind == "web_fetch"
            else {**valid_results[0], "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}}
            if result_kind == "error"
            else valid_results[0]
        )
        results: Final = (first_result,) if result_kind == "unpaired" else (first_result, valid_results[1])
        message: Final = Message(
            role="assistant",
            content="answer",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id=call_id,
                    type="function",
                    function=Function(name="web_search", arguments=json.dumps({"query": query})),
                )
                for call_id, query in zip(call_ids, ("one", "two"), strict=True)
            ]
            + [
                ChatCompletionMessageToolCall(
                    id="toolu_regular",
                    type="function",
                    function=Function(name="get_weather", arguments='{"city":"Paris"}'),
                )
            ],
            provider_specific_fields={
                "web_search_results": results,
                "web_search_calls": [
                    build_web_search_call(
                        tool_id=result["tool_use_id"],
                        tool_input={"query": "one" if result["tool_use_id"].endswith("01Search") else "two"},
                        result=result,
                    )
                    for result in results
                    if tool_type != "function" and result["type"] == "web_search_tool_result"
                ],
            },
        )
        request_tools: Final = (
            [{"type": "function", "name": "web_search", "parameters": {"type": "object"}}]
            if tool_type == "function"
            else [{"type": tool_type}]
        )
        response: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="search",
            responses_api_request={"tools": request_tools},
            chat_completion_response=_bridged_chat_completion_response(
                choices=[Choices(index=0, finish_reason="stop", message=message)]
            ),
        )
        search_items: Final = {
            item.id.removeprefix("ws_"): item for item in response.output if item.type == "web_search_call"
        }
        function_ids: Final = {item.call_id for item in response.output if item.type == "function_call"}

        assert set(search_items) == set(expected_sources)
        assert function_ids == set(call_ids).difference(expected_sources) | {"toolu_regular"}
        assert [item.content[0].text for item in response.output if item.type == "message"] == ["answer"]
        for call_id, item in search_items.items():
            assert item.status == ("failed" if result_kind == "error" and call_id.endswith("01Search") else "completed")
            assert item.action.type == "search"
            assert item.action.query == ("one" if call_id.endswith("01Search") else "two")
            assert item.action.queries == [item.action.query]
            assert [source.url for source in item.action.sources] == expected_sources[call_id]

    def test_message_item_id_uses_msg_prefix(self):
        response = self._transform(_bridged_chat_completion_response())

        message_items = _bridged_output_items(response, "message")
        assert len(message_items) == 1
        assert message_items[0].id.startswith("msg_")

    def test_message_item_id_does_not_leak_chat_completion_id(self):
        response = self._transform(_bridged_chat_completion_response())

        for item in _bridged_output_items(response, "message"):
            assert item.id != BRIDGED_CHAT_COMPLETION_ID
            assert not item.id.startswith("chatcmpl-")

    def test_message_item_ids_are_unique_across_responses(self):
        first = self._transform(_bridged_chat_completion_response())
        second = self._transform(_bridged_chat_completion_response())

        first_id = _bridged_output_items(first, "message")[0].id
        second_id = _bridged_output_items(second, "message")[0].id
        assert first_id != second_id

    def _reasoning_items(self):
        message = Message(role="assistant", content="apple")
        message.reasoning_content = "thinking about fruit"
        choice = Choices(index=0, finish_reason="stop", message=message)
        return LiteLLMCompletionResponsesConfig._extract_reasoning_output_items(
            chat_completion_response=_bridged_chat_completion_response(),
            choices=[choice],
        )

    def test_reasoning_item_id_uses_rs_prefix(self):
        items = self._reasoning_items()

        assert len(items) == 1
        assert items[0].id.startswith("rs_")

    def test_reasoning_item_id_is_not_a_salted_hash(self):
        """Python's hash() is salted per process, so the old rs_{hash(...)} ID for the
        same reasoning text differed between workers and across restarts."""
        suffix = self._reasoning_items()[0].id.removeprefix("rs_")

        assert not suffix.lstrip("-").isdigit()
        assert not suffix.startswith("-")


class TestStreamingSnapshotItemIds:
    """The response.completed snapshot must reuse the streamed item ID (issue #27333).

    The incremental events already minted msg_* IDs while the final snapshot went back
    through the non-streaming transform, so a streaming client replaying the snapshot
    sent back an ID it had never been shown.
    """

    def _make_iterator(self):
        from unittest.mock import Mock

        import litellm
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_stream_wrapper.logging_obj = Mock()
        return LiteLLMCompletionStreamingIterator(
            model="anthropic/claude-sonnet-4-5",
            litellm_custom_stream_wrapper=mock_stream_wrapper,
            request_input="Say the single word: apple",
            responses_api_request={},
            custom_llm_provider="anthropic",
        )

    def _make_chunk(self, content):
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        return ModelResponseStream(
            id=BRIDGED_CHAT_COMPLETION_ID,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=content, role="assistant"),
                    finish_reason=None,
                )
            ],
            created=1717000000,
            model="claude-sonnet-4-5",
            object="chat.completion.chunk",
        )

    def test_incremental_item_id_uses_msg_prefix(self):
        iterator = self._make_iterator()

        event = iterator._transform_chat_completion_chunk_to_response_api_chunk(self._make_chunk("apple"))

        assert event is not None
        assert event.item_id.startswith("msg_")
        assert event.item_id != BRIDGED_CHAT_COMPLETION_ID

    def test_completed_snapshot_reuses_streamed_item_id(self):
        iterator = self._make_iterator()

        streamed_event = iterator._transform_chat_completion_chunk_to_response_api_chunk(self._make_chunk("apple"))
        assert streamed_event is not None

        completed_event = iterator._emit_terminal_response_event(_bridged_chat_completion_response())

        assert completed_event is not None
        message_items = _bridged_output_items(completed_event.response, "message")
        assert len(message_items) == 1
        assert message_items[0].id == streamed_event.item_id

    def test_completed_snapshot_item_id_is_replayable(self):
        iterator = self._make_iterator()
        iterator._transform_chat_completion_chunk_to_response_api_chunk(self._make_chunk("apple"))

        completed_event = iterator._emit_terminal_response_event(_bridged_chat_completion_response())

        assert completed_event is not None
        for item in _bridged_output_items(completed_event.response, "message"):
            assert item.id.startswith("msg_")
            assert not item.id.startswith("chatcmpl-")

    def _make_reasoning_chunk(self, reasoning_content):
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        return ModelResponseStream(
            id=BRIDGED_CHAT_COMPLETION_ID,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", reasoning_content=reasoning_content),
                    finish_reason=None,
                )
            ],
            created=1717000000,
            model="claude-sonnet-4-5",
            object="chat.completion.chunk",
        )

    def _reasoning_chat_completion_response(self):
        message = Message(role="assistant", content="apple")
        message.reasoning_content = "thinking about fruit"
        return _bridged_chat_completion_response(choices=[Choices(index=0, finish_reason="stop", message=message)])

    def test_reasoning_delta_events_share_one_item_id(self):
        """The old rs_{hash(text)} ID changed with every delta, so a client accumulating
        reasoning by item ID saw a new item per chunk."""
        iterator = self._make_iterator()

        first = iterator._transform_chat_completion_chunk_to_response_api_chunk(self._make_reasoning_chunk("thinking "))
        second = iterator._transform_chat_completion_chunk_to_response_api_chunk(
            self._make_reasoning_chunk("about fruit")
        )

        assert first is not None and second is not None
        assert first.item_id.startswith("rs_")
        assert first.item_id == second.item_id

    def test_completed_snapshot_reuses_streamed_reasoning_item_id(self):
        iterator = self._make_iterator()

        streamed_event = iterator._transform_chat_completion_chunk_to_response_api_chunk(
            self._make_reasoning_chunk("thinking about fruit")
        )
        assert streamed_event is not None

        completed_event = iterator._emit_terminal_response_event(self._reasoning_chat_completion_response())

        assert completed_event is not None
        reasoning_items = _bridged_output_items(completed_event.response, "reasoning")
        assert len(reasoning_items) == 1
        assert reasoning_items[0].id == streamed_event.item_id


def test_transform_chat_completion_response_incomplete_details():
    from litellm.types.llms.openai import IncompleteDetails

    resp_length = ModelResponse(
        id="resp-length",
        choices=[Choices(index=0, finish_reason="length", message=Message(content="cutoff", role="assistant"))],
        model="gpt-4o",
    )
    result_length = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={},
        chat_completion_response=resp_length,
    )
    assert result_length.status == "incomplete"
    assert result_length.incomplete_details is not None
    assert result_length.incomplete_details.reason == "max_output_tokens"

    resp_filter = ModelResponse(
        id="resp-filter",
        choices=[Choices(index=0, finish_reason="content_filter", message=Message(content=None, role="assistant"))],
        model="gpt-4o",
    )
    result_filter = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={},
        chat_completion_response=resp_filter,
    )
    assert result_filter.status == "incomplete"
    assert result_filter.incomplete_details is not None
    assert result_filter.incomplete_details.reason == "content_filter"

    resp_refusal = ModelResponse(
        id="resp-refusal",
        choices=[Choices(index=0, finish_reason="refusal", message=Message(content=None, role="assistant"))],
        model="gpt-4o",
    )
    result_refusal = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={},
        chat_completion_response=resp_refusal,
    )
    assert result_refusal.status == "incomplete"
    assert result_refusal.incomplete_details is not None
    assert result_refusal.incomplete_details.reason == "content_filter"

    existing_details = IncompleteDetails(reason="content_filter")
    resp_existing = ModelResponse(
        id="resp-existing",
        choices=[Choices(index=0, finish_reason="length", message=Message(content="cutoff", role="assistant"))],
        model="gpt-4o",
    )
    resp_existing.incomplete_details = existing_details
    result_existing = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={},
        chat_completion_response=resp_existing,
    )
    assert result_existing.status == "incomplete"
    assert result_existing.incomplete_details == existing_details


def test_transform_chat_completion_response_null_finish_reason_is_incomplete():
    """
    A choice with no finish signal (e.g. Gemini thinking consuming the whole
    output budget, or a provider that never sent one) must not report
    status "completed". Output at the requested cap reads as
    max_output_tokens; below it, no reason can be claimed.
    """
    choice_at_cap = Choices.model_construct(
        index=0,
        finish_reason=None,
        message=Message(content="", role="assistant"),
    )
    resp_at_cap = ModelResponse(
        id="resp-null-cap",
        choices=[choice_at_cap],
        model="gemini-2.5-pro",
        usage=Usage(prompt_tokens=10, completion_tokens=60, total_tokens=70),
    )
    result_at_cap = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={"max_output_tokens": 60},
        chat_completion_response=resp_at_cap,
    )
    assert result_at_cap.status == "incomplete"
    assert result_at_cap.incomplete_details is not None
    assert result_at_cap.incomplete_details.reason == "max_output_tokens"

    choice_under_cap = Choices.model_construct(
        index=0,
        finish_reason=None,
        message=Message(content="", role="assistant"),
    )
    resp_under_cap = ModelResponse(
        id="resp-null-under",
        choices=[choice_under_cap],
        model="gemini-2.5-pro",
        usage=Usage(prompt_tokens=10, completion_tokens=30, total_tokens=40),
    )
    result_under_cap = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={"max_output_tokens": 60},
        chat_completion_response=resp_under_cap,
    )
    assert result_under_cap.status == "incomplete"
    assert result_under_cap.incomplete_details is None

    resp_empty = ModelResponse(id="resp-empty", choices=[], model="gemini-2.5-pro")
    result_empty = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={},
        chat_completion_response=resp_empty,
    )
    assert result_empty.status == "incomplete"


def test_transform_chat_completion_response_echoes_request_params():
    """
    A chat completion response never carries the caller's request fields, so
    the bridged response echoes them from the request. Before this fix
    temperature reported 0 and the rest reported None or empty defaults.
    """
    resp = ModelResponse(
        id="resp-echo",
        choices=[Choices(index=0, finish_reason="stop", message=Message(content="hi", role="assistant"))],
        model="gpt-4o",
    )
    result = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 128,
            "instructions": "be terse",
            "metadata": {"trace": "abc"},
            "user": "user-123",
            "truncation": "disabled",
            "store": True,
            "parallel_tool_calls": True,
        },
        chat_completion_response=resp,
    )
    assert result.temperature == 0.7
    assert result.top_p == 0.9
    assert result.max_output_tokens == 128
    assert result.instructions == "be terse"
    assert result.metadata == {"trace": "abc"}
    assert result.user == "user-123"
    assert result.truncation == "disabled"
    assert result.store is True
    assert result.parallel_tool_calls is True

    # Fields the request never set fall back to the response defaults.
    result_unset = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request={},
        chat_completion_response=resp,
    )
    assert result_unset.temperature is None
    assert result_unset.metadata == {}
    assert result_unset.parallel_tool_calls is False
    assert result_unset.tools == []
    assert result_unset.text == {}


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected"),
    [
        ("user", 123, None),
        ("instructions", 5, None),
        ("text", "plain", {}),
        ("truncation", "bogus", None),
        ("parallel_tool_calls", "maybe", False),
    ],
)
def test_transform_drops_request_params_the_response_cannot_echo(
    field: str, invalid_value: object, expected: object
) -> None:
    request: Final[dict[str, object]] = {"temperature": 0.7, field: invalid_value}
    response: Final = ModelResponse(
        id="resp-invalid-echo",
        choices=[Choices(index=0, finish_reason="stop", message=Message(content="hi", role="assistant"))],
        model="gpt-4o",
    )

    result: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test prompt",
        responses_api_request=request,
        chat_completion_response=response,
    )

    assert getattr(result, field) == expected, f"invalid {field} value {invalid_value!r}"
    assert result.temperature == 0.7, f"valid temperature was dropped with invalid {field}"


@pytest.mark.parametrize("stream", [True, False])
async def test_bridge_rejects_untranslatable_tool_choice_with_a_400(stream: bool):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.aresponses(
            model="anthropic/claude-haiku-4-5",
            input="Which fruit is red?",
            tools=[{"type": "function", "name": "lookup_fruit", "parameters": {"type": "object"}}],
            tool_choice={"type": "file_search"},
            stream=stream,
            api_key="sk-unused",
        )
    assert exc_info.value.status_code == 400
    assert "tool_choice={'type': 'file_search'}" in str(exc_info.value)


def test_cache_control_on_user_message():
    """cache_control on a user message should be passed through to GenericChatCompletionMessage"""
    input_item = {
        "role": "user",
        "content": "Hello, world!",
        "cache_control": {"type": "ephemeral"},
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "user"
    assert msg.get("cache_control") == {"type": "ephemeral"}


def test_cache_control_on_assistant_message():
    """cache_control on an assistant message should be passed through"""
    input_item = {
        "role": "assistant",
        "content": "I can help with that.",
        "cache_control": {"type": "ephemeral"},
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert msg.get("cache_control") == {"type": "ephemeral"}


def test_no_cache_control_when_absent():
    """Messages without cache_control should not have the field set"""
    input_item = {
        "role": "user",
        "content": "Hello",
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    assert len(result) == 1
    msg = result[0]
    assert "cache_control" not in msg


def test_cache_control_on_function_call_output():
    """cache_control on a function_call_output should be passed through to tool message"""
    TOOL_CALLS_CACHE.set_cache(
        key="call_123",
        value={
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        },
    )

    input_item = {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": '{"temp": 72}',
        "cache_control": {"type": "ephemeral"},
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    # Should have assistant message + tool message
    assert len(result) == 2
    tool_msg = result[1]
    assert tool_msg["role"] == "tool"
    assert tool_msg.get("cache_control") == {"type": "ephemeral"}


def test_cache_control_on_function_call():
    """cache_control on a function_call input should land on the tool call it builds"""
    input_item = {
        "type": "function_call",
        "call_id": "call_456",
        "name": "get_weather",
        "arguments": '{"city": "NYC"}',
        "cache_control": {"type": "ephemeral"},
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert "cache_control" not in msg
    assert msg["tool_calls"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_full_input_transform():
    """cache_control should survive the full transform_responses_api_input_to_messages path"""
    input_items = [
        {
            "role": "user",
            "content": "What is the weather?",
        },
        {
            "role": "assistant",
            "content": "Let me check.",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "user",
            "content": "Thanks!",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    result = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
        item_cache_control=True,
    )

    # 3 messages, no system
    assert len(result) == 3
    assert "cache_control" not in result[0]
    assert result[1].get("cache_control") == {"type": "ephemeral"}
    assert result[2].get("cache_control") == {"type": "ephemeral"}


def test_cache_control_propagated_to_last_content_block_for_list_content():
    """
    For Anthropic/Bedrock, an item-level cache_control over list content activates
    prompt caching only as a mark on the last content block, so the bridge moves
    it there instead of keeping a message-level copy the hook's census would also count.
    """
    input_item = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "first block"},
            {"type": "input_text", "text": "last block"},
        ],
        "cache_control": {"type": "ephemeral"},
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    assert len(result) == 1
    msg = result[0]
    assert "cache_control" not in msg
    assert isinstance(msg["content"], list)
    assert "cache_control" not in msg["content"][0]
    assert msg["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert AnthropicCacheControlHook.count_request_cache_breakpoints(result) == 1


def test_cache_control_does_not_overwrite_existing_block_level_cache_control():
    """
    If the last content block already carries its own cache_control
    (preserved by _transform_responses_api_content_to_chat_completion_content),
    the message-level cache_control must not silently overwrite it.
    """
    input_item = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "last block",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        "cache_control": {"type": "ephemeral"},
    }

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=input_item
    )

    assert len(result) == 1
    msg = result[0]
    assert "cache_control" not in msg
    assert msg["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert AnthropicCacheControlHook.count_request_cache_breakpoints(result) == 1


def test_function_call_cache_control_reaches_the_provider_tool_use_block():
    cache_control: Final = {"type": "ephemeral"}
    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=[
            {"role": "user", "content": "Summarize the gateway latency report."},
            {
                "type": "function_call",
                "call_id": "call_cached_tool_use",
                "name": "get_latency_report",
                "arguments": '{"service": "gateway"}',
                "cache_control": cache_control,
            },
            {"type": "function_call_output", "call_id": "call_cached_tool_use", "output": "p50 120ms"},
        ],
        responses_api_request={},
        item_cache_control=True,
    )

    anthropic_turns: Final = anthropic_messages_pt(
        messages=messages, model="claude-sonnet-4-6", llm_provider="anthropic"
    )
    tool_use_block: Final = anthropic_turns[1]["content"][0]
    assert tool_use_block["type"] == "tool_use"
    assert tool_use_block["cache_control"] == cache_control

    bedrock_blocks: Final = _bedrock_converse_messages_pt(
        messages=messages, model="us.anthropic.claude-sonnet-4-6", llm_provider="bedrock"
    )[1]["content"]
    tool_use_index: Final = next(index for index, block in enumerate(bedrock_blocks) if "toolUse" in block)
    assert "cachePoint" in bedrock_blocks[tool_use_index + 1]


def test_assistant_text_merged_into_a_function_call_turn_keeps_its_cache_control():
    cache_control: Final = {"type": "ephemeral"}
    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=[
            {"role": "user", "content": "Summarize the gateway latency report."},
            {
                "type": "function_call",
                "call_id": "call_merged_text",
                "name": "get_latency_report",
                "arguments": '{"service": "gateway"}',
            },
            {"role": "assistant", "content": "Looking that up now.", "cache_control": cache_control},
            {"type": "function_call_output", "call_id": "call_merged_text", "output": "p50 120ms"},
        ],
        responses_api_request={},
        item_cache_control=True,
    )

    assert len(messages) == 3
    assert messages[1]["cache_control"] == cache_control

    anthropic_turns: Final = anthropic_messages_pt(
        messages=messages, model="claude-sonnet-4-6", llm_provider="anthropic"
    )
    text_block: Final = anthropic_turns[1]["content"][0]
    assert text_block["type"] == "text"
    assert text_block["cache_control"] == cache_control

    bedrock_blocks: Final = _bedrock_converse_messages_pt(
        messages=messages, model="us.anthropic.claude-sonnet-4-6", llm_provider="bedrock"
    )[1]["content"]
    assert any("cachePoint" in block for block in bedrock_blocks)


def test_assistant_text_merged_behind_earlier_text_puts_cache_control_on_the_last_block():
    cache_control: Final = {"type": "ephemeral"}
    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=[
            {"role": "user", "content": "Summarize the gateway latency report."},
            {
                "type": "function_call",
                "call_id": "call_two_texts",
                "name": "get_latency_report",
                "arguments": '{"service": "gateway"}',
            },
            {"role": "assistant", "content": "Looking that up now."},
            {"role": "assistant", "content": "One moment.", "cache_control": cache_control},
            {"type": "function_call_output", "call_id": "call_two_texts", "output": "p50 120ms"},
        ],
        responses_api_request={},
        item_cache_control=True,
    )

    merged_turn: Final = messages[1]
    assert "cache_control" not in merged_turn
    assert "cache_control" not in merged_turn["content"][0]
    assert merged_turn["content"][-1]["cache_control"] == cache_control
    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == 1


def test_marked_assistant_text_merged_ahead_of_later_text_keeps_its_cache_control_on_its_block():
    cache_control: Final = {"type": "ephemeral"}
    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=[
            {"role": "user", "content": "Summarize the gateway latency report."},
            {
                "type": "function_call",
                "call_id": "call_marked_first_text",
                "name": "get_latency_report",
                "arguments": '{"service": "gateway"}',
            },
            {"role": "assistant", "content": "Looking that up now.", "cache_control": cache_control},
            {"role": "assistant", "content": "One moment."},
            {"type": "function_call_output", "call_id": "call_marked_first_text", "output": "p50 120ms"},
        ],
        responses_api_request={},
        item_cache_control=True,
    )

    merged_turn: Final = messages[1]
    assert "cache_control" not in merged_turn
    assert merged_turn["content"][0]["cache_control"] == cache_control
    assert "cache_control" not in merged_turn["content"][-1]
    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == 1

    anthropic_turns: Final = anthropic_messages_pt(
        messages=messages, model="claude-sonnet-4-6", llm_provider="anthropic"
    )
    blocks: Final = TypeAdapter(list[dict[str, object]]).validate_python(anthropic_turns[1]["content"])
    marked_blocks: Final = [block for block in blocks if block.get("cache_control")]
    assert len(marked_blocks) == 1
    assert marked_blocks[0]["text"] == "Looking that up now."


_CLAUDE_ON_BEDROCK: Final = "us.anthropic.claude-sonnet-4-6"
_EPHEMERAL: Final = {"type": "ephemeral"}


def _replayed_messages_from(
    turns: Sequence[Mapping[str, object]],
    current: Mapping[str, object],
    *,
    model: str = _CLAUDE_ON_BEDROCK,
    tools: Sequence[Mapping[str, object]] = (),
    **request_kwargs: object,
) -> Sequence[Mapping[str, object]]:
    async def session_lookup(previous_response_id: str) -> ChatCompletionSession:
        assert previous_response_id == "resp_marked_history"
        return ChatCompletionSession(messages=list(turns), litellm_session_id=None)

    request: Final = asyncio.run(
        LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id="resp_marked_history",
            litellm_completion_request={
                "model": model,
                "custom_llm_provider": "bedrock",
                "messages": [current],
                **({"tools": list(tools)} if tools else {}),
            },
            instructions=None,
            session_lookup=session_lookup,
            request_kwargs=request_kwargs,
        )
    )
    return TypeAdapter(list[dict[str, object]]).validate_python(request["messages"])


def test_a_replayed_turn_over_the_remaining_budget_yields_its_marks_while_older_turns_keep_theirs():
    three_marked_blocks: Final = [
        {"type": "text", "text": f"block {index}", "cache_control": _EPHEMERAL} for index in range(3)
    ]
    replayed_turns: Final = [
        {"role": "user", "content": "turn 0", "cache_control": _EPHEMERAL},
        {"role": "user", "content": "turn 1", "cache_control": _EPHEMERAL},
        {"role": "user", "content": three_marked_blocks},
        {"role": "user", "content": "turn 3", "cache_control": _EPHEMERAL},
    ]

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest", "cache_control": _EPHEMERAL}
    )

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == MAX_CACHE_CONTROL_BLOCKS
    assert [AnthropicCacheControlHook._count_cache_control_blocks(message) for message in messages] == [1, 1, 0, 1, 1]
    assert messages[2] == {
        "role": "user",
        "content": [{"type": "text", "text": f"block {index}"} for index in range(3)],
    }


def test_replayed_session_turns_yield_their_cache_control_to_the_newest_markers():
    replayed_turns: Final = [
        {"role": "user", "content": f"turn {index}", "cache_control": _EPHEMERAL} for index in range(5)
    ]

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest", "cache_control": _EPHEMERAL}
    )
    assert ["cache_control" in message for message in messages] == [False, False, True, True, True, True]
    assert messages[0] == {"role": "user", "content": "turn 0"}
    assert messages[-1] == {"role": "user", "content": "newest", "cache_control": _EPHEMERAL}


def test_a_replayed_claude_search_call_is_dropped_off_anthropic_while_the_turns_keep_their_cache_marks():
    search_call: Final = {
        "id": "srvtoolu_search",
        "type": "function",
        "function": {"name": "tool_search_tool_regex", "arguments": '{"pattern": "weather"}'},
    }
    weather_call: Final = {
        "id": "toolu_weather",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
    }
    replayed_turns: Final = [
        {"role": "user", "content": "turn 0", "cache_control": _EPHEMERAL},
        {"role": "assistant", "content": None, "tool_calls": [search_call, weather_call]},
        {"role": "tool", "tool_call_id": "srvtoolu_search", "content": "get_weather"},
        {"role": "tool", "tool_call_id": "toolu_weather", "content": "18C"},
        *({"role": "user", "content": f"turn {index}", "cache_control": _EPHEMERAL} for index in range(1, 4)),
    ]

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest", "cache_control": _EPHEMERAL}
    )

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user", "user", "user", "user"]
    assert messages[1]["tool_calls"] == [weather_call]
    assert messages[2]["tool_call_id"] == "toolu_weather"
    assert ["cache_control" in message for message in messages] == [False, False, False, True, True, True, True]


def test_a_marked_tool_takes_a_slot_from_the_replayed_turns():
    replayed_turns: Final = [
        {"role": "user", "content": f"turn {index}", "cache_control": _EPHEMERAL} for index in range(3)
    ]
    marked_tool: Final = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object"}},
        "cache_control": _EPHEMERAL,
    }

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest", "cache_control": _EPHEMERAL}, tools=[marked_tool]
    )

    assert ["cache_control" in message for message in messages] == [False, True, True, True]


@pytest.mark.parametrize(
    "request_kwargs",
    [{"cache_control": _EPHEMERAL}, {"extra_body": {"cache_control": _EPHEMERAL}}],
    ids=["top_level", "extra_body"],
)
def test_a_top_level_cache_control_takes_a_slot_from_the_replayed_turns(request_kwargs: Mapping[str, object]):
    replayed_turns: Final = [
        {"role": "user", "content": f"turn {index}", "cache_control": _EPHEMERAL} for index in range(3)
    ]

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest", "cache_control": _EPHEMERAL}, **request_kwargs
    )

    assert ["cache_control" in message for message in messages] == [False, True, True, True]


def test_a_null_cache_control_on_the_current_turn_takes_no_slot_from_the_replayed_turns():
    current: Final = {"role": "user", "content": [{"type": "text", "text": "newest", "cache_control": None}]}

    messages: Final = _replayed_messages_from(list(_marked_turns(4)), current)

    assert ["cache_control" in message for message in messages[:4]] == [True, True, True, True]


@pytest.mark.parametrize("drop_params", [False, True])
def test_replayed_marks_a_provider_would_refuse_drop_with_or_without_drop_params(drop_params: bool):
    replayed_turns: Final = [{"role": "user", "content": "turn 0", "cache_control": {"type": "forever"}}]
    current: Final = {"role": "user", "content": "newest"}

    assert _replayed_messages_from(replayed_turns, current, drop_params=drop_params)[0] == {
        "role": "user",
        "content": "turn 0",
    }


def test_replayed_tool_marks_drop_for_nova_without_drop_params():
    replayed_turns: Final = [
        {"role": "user", "content": "Summarize the gateway latency report."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_latency_report", "arguments": "{}"},
                    "cache_control": _EPHEMERAL,
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "p50 12 ms", "cache_control": _EPHEMERAL},
    ]

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest"}, model="us.amazon.nova-2-lite-v1:0"
    )

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == 0
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "p50 12 ms"}


def test_replayed_marks_stay_untouched_for_a_target_without_cache_points():
    replayed_turns: Final = [
        {"role": "user", "content": f"turn {index}", "cache_control": _EPHEMERAL} for index in range(5)
    ]

    messages: Final = _replayed_messages_from(
        replayed_turns, {"role": "user", "content": "newest"}, model="meta.llama3-70b-instruct-v1:0"
    )

    assert messages[:5] == replayed_turns


def _bridged_request(
    *items: Mapping[str, object],
    model: str = f"bedrock/{_CLAUDE_ON_BEDROCK}",
    custom_llm_provider: str = "bedrock",
    tools: Sequence[Mapping[str, object]] = (),
    **kwargs: object,
) -> Sequence[Mapping[str, object]]:
    request: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=model,
        input=cast(ResponseInputParam, list(items)),  # cast-ok: cache_control is not a key of the input TypedDicts
        responses_api_request=cast(  # cast-ok: cache_control is not a key of the tool TypedDicts
            ResponsesAPIOptionalRequestParams, {"instructions": "Be brief.", "tools": list(tools)}
        ),
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )
    return TypeAdapter(list[dict[str, object]]).validate_python(request["messages"])


def _marked_turns(count: int) -> tuple[Mapping[str, object], ...]:
    return tuple({"role": "user", "content": f"turn {index}", "cache_control": _EPHEMERAL} for index in range(count))


def test_marks_past_the_provider_cap_reach_it_as_sent_without_drop_params():
    messages: Final = _bridged_request(*_marked_turns(5))

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == MAX_CACHE_CONTROL_BLOCKS + 1


@pytest.mark.parametrize("drop_params", [{"drop_params": True}, {"drop_params": "true"}], ids=["bool", "string"])
def test_marks_past_the_provider_cap_keep_the_newest_under_drop_params(drop_params: Mapping[str, object]):
    messages: Final = _bridged_request(*_marked_turns(5), **drop_params)

    assert ["cache_control" in message for message in messages] == [False, False, True, True, True, True]
    assert messages[1] == {"role": "user", "content": "turn 0"}


def test_global_drop_params_keeps_the_newest_marks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "drop_params", True)

    messages: Final = _bridged_request(*_marked_turns(5))

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == MAX_CACHE_CONTROL_BLOCKS


def test_a_marked_tool_takes_a_slot_from_the_request_marks_under_drop_params():
    marked_tool: Final = {
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object"},
        "cache_control": _EPHEMERAL,
    }

    messages: Final = _bridged_request(*_marked_turns(4), tools=[marked_tool], drop_params=True)

    assert ["cache_control" in message for message in messages] == [False, False, True, True, True]


@pytest.mark.parametrize(
    "request_kwargs",
    [{"cache_control": _EPHEMERAL}, {"extra_body": {"cache_control": _EPHEMERAL}}],
    ids=["top_level", "extra_body"],
)
def test_a_top_level_cache_control_takes_a_slot_from_the_request_marks_under_drop_params(
    request_kwargs: Mapping[str, object],
):
    messages: Final = _bridged_request(*_marked_turns(4), drop_params=True, **request_kwargs)

    assert ["cache_control" in message for message in messages] == [False, False, True, True, True]


def test_an_item_mark_replaces_a_null_mark_on_its_last_block():
    item: Final = {
        "role": "user",
        "content": [{"type": "input_text", "text": "x", "cache_control": None}],
        "cache_control": _EPHEMERAL,
    }

    assert _bridged_request(item)[1]["content"] == [{"type": "text", "text": "x", "cache_control": _EPHEMERAL}]


_MARKED_TOOL_ROUND_TRIP: Final = (
    {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}", "cache_control": _EPHEMERAL},
    {"type": "function_call_output", "call_id": "call_1", "output": "42", "cache_control": _EPHEMERAL},
    {"role": "user", "content": "thanks", "cache_control": _EPHEMERAL},
)


def _marked_tool_round_trip_marks(messages: Sequence[Mapping[str, object]]) -> tuple[bool, bool, bool]:
    tool_calls: Final = TypeAdapter(list[dict[str, object]]).validate_python(messages[1]["tool_calls"])
    return ("cache_control" in tool_calls[0], "cache_control" in messages[2], "cache_control" in messages[3])


def test_nova_gets_no_marks_on_tool_calls_and_results_under_drop_params():
    nova: Final = "bedrock/us.amazon.nova-2-lite-v1:0"

    assert _marked_tool_round_trip_marks(_bridged_request(*_MARKED_TOOL_ROUND_TRIP, model=nova)) == (True, True, True)
    assert _marked_tool_round_trip_marks(_bridged_request(*_MARKED_TOOL_ROUND_TRIP, model=nova, drop_params=True)) == (
        False,
        False,
        True,
    )


def test_claude_keeps_marks_on_tool_calls_and_results_under_drop_params():
    assert _marked_tool_round_trip_marks(_bridged_request(*_MARKED_TOOL_ROUND_TRIP, drop_params=True)) == (
        True,
        True,
        True,
    )


@pytest.mark.parametrize("value", [1, "ephemeral", ["ephemeral"], {"type": "forever"}])
def test_a_cache_control_value_the_provider_would_refuse_reaches_it_as_sent_without_drop_params(value: object):
    assert _bridged_request({"role": "user", "content": "hi", "cache_control": value})[1] == {
        "role": "user",
        "content": "hi",
        "cache_control": value,
    }


@pytest.mark.parametrize("value", [1, "ephemeral", ["ephemeral"], {"type": "forever"}])
def test_a_cache_control_value_the_provider_would_refuse_is_dropped_under_drop_params(value: object):
    assert _bridged_request({"role": "user", "content": "hi", "cache_control": value}, drop_params=True)[1] == {
        "role": "user",
        "content": "hi",
    }


def test_drop_params_keeps_a_valid_mark_with_its_ttl_and_without_unknown_keys():
    marked: Final = {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h", "extra": 1}}

    assert _bridged_request(marked, drop_params=True)[1] == {
        "role": "user",
        "content": "hi",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }


def test_a_marked_developer_item_reaches_a_claude_target_as_a_marked_system_message():
    developer: Final = {
        "role": "developer",
        "content": "Answer in French.",
        "name": "policy",
        "cache_control": _EPHEMERAL,
    }

    assert _bridged_request(developer)[1] == {
        "role": "system",
        "content": "Answer in French.",
        "cache_control": _EPHEMERAL,
    }


@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    [
        ("gemini/gemini-2.5-flash", "gemini"),
        ("mistral/mistral-small-latest", "mistral"),
        ("bedrock/meta.llama3-70b-instruct-v1:0", "bedrock"),
        ("openrouter/openai/gpt-5", "openrouter"),
        ("databricks/databricks-meta-llama-3-3-70b-instruct", "databricks"),
        ("snowflake/llama-claude-distill", "snowflake"),
    ],
)
def test_item_marks_stay_off_a_target_without_cache_points(model: str, custom_llm_provider: str):
    messages: Final = _bridged_request(
        {"role": "system", "content": "House rules", "cache_control": _EPHEMERAL},
        {"role": "developer", "content": "Answer in French.", "cache_control": _EPHEMERAL},
        {"role": "user", "content": "hi"},
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == 0
    assert messages[2] == {"role": "developer", "content": "Answer in French."}


@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    [
        ("databricks/databricks-claude-sonnet-4", "databricks"),
        ("dashscope/qwen-plus", "dashscope"),
        ("minimax/MiniMax-M2", "minimax"),
        ("zai/glm-4.6", "zai"),
        ("sap/anthropic--claude-4-sonnet", "sap"),
        ("openrouter/anthropic/claude-sonnet-4.5", "openrouter"),
        ("openrouter/google/gemini-2.5-flash", "openrouter"),
        ("snowflake/claude-sonnet-4-5", "snowflake"),
        ("anthropic/my-gateway-model", "anthropic"),
    ],
)
def test_item_marks_reach_a_target_whose_chat_route_keeps_them(model: str, custom_llm_provider: str):
    messages: Final = _bridged_request(
        {"role": "system", "content": "House rules", "cache_control": _EPHEMERAL},
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    assert messages[1] == {"role": "system", "content": "House rules", "cache_control": _EPHEMERAL}


def test_a_block_mark_reaches_a_target_without_cache_points_as_before():
    block: Final = {"type": "input_text", "text": "House rules", "cache_control": _EPHEMERAL}

    messages: Final = _bridged_request(
        {"role": "system", "content": [block]}, model="gemini/gemini-2.5-flash", custom_llm_provider="gemini"
    )

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == 1


_APPLICATION_INFERENCE_PROFILE: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"


def test_an_application_inference_profile_keeps_its_item_marks_without_drop_params():
    messages: Final = _bridged_request(
        {"role": "system", "content": "House rules", "cache_control": _EPHEMERAL},
        model=f"bedrock/{_APPLICATION_INFERENCE_PROFILE}",
    )

    assert messages[1] == {"role": "system", "content": "House rules", "cache_control": _EPHEMERAL}


def test_an_application_inference_profile_gets_no_item_marks_under_drop_params():
    messages: Final = _bridged_request(
        {"role": "system", "content": "House rules", "cache_control": _EPHEMERAL},
        {"role": "user", "content": "Summarize the gateway latency report."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_latency_report",
            "arguments": "{}",
            "cache_control": _EPHEMERAL,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "p50 12 ms", "cache_control": _EPHEMERAL},
        model=f"bedrock/{_APPLICATION_INFERENCE_PROFILE}",
        drop_params=True,
    )

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == 0


def test_the_shared_input_transform_keeps_every_item_mark_for_a_request_it_sends():
    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=cast(ResponseInputParam, list(_marked_turns(5))),  # cast-ok: cache_control is not an input TypedDict key
        responses_api_request={},
        item_cache_control=True,
    )

    assert AnthropicCacheControlHook.count_request_cache_breakpoints(messages) == MAX_CACHE_CONTROL_BLOCKS + 1


def test_guardrails_and_token_counting_read_the_input_as_if_it_carried_no_item_marks():
    marked_turns: Final = list(_marked_turns(5))

    inspected: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=cast(ResponseInputParam, marked_turns),  # cast-ok: cache_control is not an input TypedDict key
        responses_api_request={},
    )
    unmarked: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=cast(ResponseInputParam, [{"role": "user", "content": turn["content"]} for turn in marked_turns]),
        responses_api_request={},
    )

    assert inspected == unmarked


def test_a_single_item_input_reads_the_same_whether_or_not_item_marks_are_kept():
    single_item: Final = cast(  # cast-ok: a bare item where the input list belongs, as some clients send it
        ResponseInputParam, {"type": "message", "role": "user", "content": "hi"}
    )

    inspected: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=single_item, responses_api_request={}
    )
    sent: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=single_item, responses_api_request={}, item_cache_control=True
    )

    assert inspected == sent


@pytest.mark.parametrize("index", [0, 1, 2])
def test_convert_response_function_tool_call_keeps_the_index_it_is_given(index: int) -> None:
    item: Final = ResponseFunctionToolCall(
        id="fc_x",
        type="function_call",
        status="completed",
        arguments='{"city":"Paris"}',
        call_id="call_x",
        name="get_weather",
    )

    tool_call: Final = (
        LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
            tool_call_item=item, index=index
        )
    )

    assert tool_call["index"] == index


def test_transform_request_reads_service_tier_from_responses_api_request() -> None:
    result: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="databricks/dbrx",
        input="hello",
        responses_api_request={"service_tier": "priority"},
        custom_llm_provider="databricks",
    )

    assert result["service_tier"] == "priority"
