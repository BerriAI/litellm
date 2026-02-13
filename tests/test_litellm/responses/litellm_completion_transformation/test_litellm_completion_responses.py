import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    TOOL_CALLS_CACHE,
)
from litellm.types.llms.openai import (
    ChatCompletionResponseMessage,
    ChatCompletionToolMessage,
)
from litellm.types.utils import (
    Choices,
    CompletionTokensDetailsWrapper,
    Message,
    ModelResponse,
    Function,
    ChatCompletionMessageToolCall,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestLiteLLMCompletionResponsesConfig:
    def test_transform_input_file_item_to_file_item_with_file_id(self):
        """Test transformation of input_file item with file_id to Chat Completion file format"""
        # Setup
        input_item = {"type": "input_file", "file_id": "file-abc123xyz"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "file", "file": {"file_id": "file-abc123xyz"}}
        assert result == expected
        assert "extra_field" not in result["file"]
        assert "another_field" not in result["file"]

    def test_transform_input_image_item_to_image_item_with_image_url(self):
        """Test transformation of input_image item with image_url to Chat Completion image format"""
        # Setup
        image_url = "https://example.com/image.png"
        input_item = {"type": "input_image", "image_url": image_url, "detail": "high"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}
        assert result == expected
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "auto"

    def test_transform_input_image_item_to_image_item_empty_image_fields(self):
        """Test transformation of input_image item with no image_url or detail"""
        # Setup
        input_item = {"type": "input_image"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

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
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "auto"}}
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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="What is the meaning of life?",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        assert hasattr(responses_api_response, "output")
        assert (
            len(responses_api_response.output) >= 2
        )

        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"

        reasoning_item = reasoning_items[0]
        # Note: ID auto-generation was disabled, so reasoning items may not have IDs
        # Only assert ID format if an ID is present
        if hasattr(reasoning_item, 'id') and reasoning_item.id:
            assert reasoning_item.id.startswith("rs_"), f"Expected ID to start with 'rs_', got: {reasoning_item.id}"
        assert reasoning_item.status == "completed"
        assert reasoning_item.role == "assistant"
        assert len(reasoning_item.content) == 1
        assert reasoning_item.content[0].type == "output_text"
        assert "step by step" in reasoning_item.content[0].text
        assert "42" in reasoning_item.content[0].text

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="A simple question?",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 0, "Should have no reasoning items"

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
        assert len(message_items) == 1, "Should have exactly one message item"
        assert message_items[0].content[0].text == "Just a regular answer."

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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="A question with multiple answers?",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"
        assert reasoning_items[0].content[0].text == "First reasoning process."

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
        assert len(message_items) == 2, "Should have two message items"

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

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Test",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        assert hasattr(responses_api_response, "_hidden_params")
        assert responses_api_response._hidden_params == {
            "model_id": "abc123",
            "cache_key": "some-cache-key",
            "custom_llm_provider": "openai",
        }

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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Test",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert - should default to empty dict
        assert hasattr(responses_api_response, "_hidden_params")
        assert responses_api_response._hidden_params == {}

class TestFunctionCallTransformation:
    """Test cases for function_call input transformation"""

    def test_function_call_detection(self):
        """Test that function_call items are correctly detected"""
        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "test"}',
            "call_id": "test_id"
        }
        
        function_call_output_item = {
            "type": "function_call_output",
            "call_id": "test_id",
            "output": "result"
        }
        
        regular_message = {
            "type": "message",
            "role": "user",
            "content": "Hello"
        }
        
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
            "status": "completed"
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

    def test_complete_input_transformation_with_function_calls(self):
        """Test the complete transformation with the exact input from the issue"""
        test_input = [
            {
                "type": "message",
                "role": "user",
                "content": "How is the weather in São Paulo today ?"
            },
            {
                "type": "function_call",
                "arguments": '{"location": "São Paulo, Brazil"}',
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "name": "get_weather",
                "id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "status": "completed"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "output": "Rainy"
            }
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
                "content": "How is the weather in São Paulo today ?"
            },
            {
                "type": "function_call",
                "arguments": '{"location": "São Paulo, Brazil"}',
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "name": "get_weather",
                "id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "status": "completed"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "output": "Rainy"
            }
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
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False
                }
            }
        ]
        
        responses_api_request = {
            "store": False,
            "tools": tools
        }
        
        # This should work without errors for non-OpenAI models
        result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="gemini/gemini-2.0-flash",
            input=test_input,
            responses_api_request=responses_api_request,
            extra_headers={"X-Test-Header": "test-value"}
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

    def test_function_call_without_call_id_fallback_to_id(self):
        """Test that function_call items can use 'id' field when 'call_id' is missing"""
        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "test"}',
            "id": "fallback_id"  # Only has 'id', not 'call_id'
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
            {"role": "tool", "content": "Found 5 results", "tool_call_id": tool_call_id},
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
            {"role": "tool", "content": "Found 3 results", "tool_call_id": tool_call_id},
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


class TestToolTransformation:
    """Test cases for tool transformation from Responses API to Chat Completion format"""

    def test_transform_vertex_ai_tools(self):
        """Test that Vertex AI tools are passed through as-is"""
        from litellm.types.llms.vertex_ai import VertexToolName

        # Create a Vertex AI tool using the enum value
        vertex_tool = {VertexToolName.CODE_EXECUTION.value: {}}
        
        tools = [vertex_tool]
        
        # Execute
        result_tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
            "headers": {
                "Authorization": "Bearer token123"
            },
        }
        
        tools = [mcp_tool]
        
        # Execute
        result_tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
        # Assert
        assert len(result_tools) == 1
        assert result_tools[0] == mcp_tool
        assert result_tools[0]["type"] == "mcp"
        assert web_search_options is None

    def test_transform_computer_use_tools(self):
        """Test that computer_use tools are passed through as-is"""
        computer_use_tool = {
            "type": "computer_use",
            "display_width_px": 1024,
            "display_height_px": 768
        }
        
        tools = [computer_use_tool]
        
        # Execute
        result_tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
        # Assert
        assert len(result_tools) == 1
        assert result_tools[0] == computer_use_tool
        assert result_tools[0]["type"] == "computer_use"
        assert web_search_options is None

    def test_transform_web_search_tools_to_web_search_options(self):
        """Test that web_search tools are converted to web_search_options"""
        web_search_tool = {
            "type": "web_search_preview",
            "search_context_size": "medium",
            "user_location": {"country": "US"}
        }
        
        tools = [web_search_tool]
        
        # Execute
        result_tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            },
            "cache_control": {"type": "ephemeral"},
            "defer_loading": True,
            "allowed_callers": ["user"],
            "input_examples": [{"location": "San Francisco"}]
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
            "cache_control": {"type": "ephemeral"}
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
                "properties": {
                    "param": {"type": "string"}
                }
            }
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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

    def test_transform_code_execution_tools(self):
        """Test that code_execution tools are passed through as-is"""
        code_execution_tool = {
            "type": "code_execution_20250825",
            "name": "python_code_execution"
        }
        
        tools = [code_execution_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
        # Assert
        assert len(result_tools) == 1
        assert result_tools[0]["type"] == "code_execution_20250825"

    def test_transform_tool_search_tools(self):
        """Test that tool_search tools are passed through as-is"""
        tool_search_regex = {
            "name": "tool_search_tool_regex",
            "description": "Search tools using regex"
        }
        
        tool_search_bm25 = {
            "name": "tool_search_tool_bm25",
            "description": "Search tools using BM25"
        }
        
        tools = [tool_search_regex, tool_search_bm25]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
                "cache_control": {"type": "ephemeral"}
            },
            # MCP tool
            {
                "type": "mcp",
                "server_label": "zapier"
            },
            # Web search tool
            {
                "type": "web_search_preview",
                "search_context_size": "high"
            },
            # Vertex AI tool
            {VertexToolName.CODE_EXECUTION.value: {}}
        ]
        
        # Execute
        result_tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
            "parameters": {
                "properties": {
                    "arg": {"type": "string"}
                }
            }
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"
        assert "properties" in result_tool["function"]["parameters"]

    def test_transform_function_tools_empty_parameters(self):
        """Test that empty parameters get 'type': 'object' added"""
        function_tool = {
            "type": "function",
            "name": "test_function",
            "description": "Test function",
            "parameters": {}
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"

    def test_transform_function_tools_missing_parameters(self):
        """Test that missing parameters get default 'type': 'object' added"""
        function_tool = {
            "type": "function",
            "name": "test_function",
            "description": "Test function"
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
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
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            }
        }
        
        tools = [function_tool]
        
        # Execute
        result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools=tools
        )
        
        # Assert
        assert len(result_tools) == 1
        result_tool = result_tools[0]
        assert result_tool["function"]["parameters"]["type"] == "object"
        assert "properties" in result_tool["function"]["parameters"]
        assert result_tool["function"]["parameters"]["properties"]["arg"]["type"] == "string"


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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.input_tokens == 13
        assert response_usage.output_tokens == 27
        assert response_usage.total_tokens == 40
        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 5
        assert response_usage.input_tokens_details.text_tokens == 8

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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.input_tokens == 9
        assert response_usage.output_tokens == 27
        assert response_usage.total_tokens == 36
        assert response_usage.input_tokens_details is not None
        assert response_usage.input_tokens_details.cached_tokens == 3
        assert response_usage.input_tokens_details.text_tokens == 6

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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
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
        response_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            chat_completion_response=chat_completion_response
        )

        # Assert
        assert response_usage.output_tokens == 150
        assert response_usage.output_tokens_details is not None
        assert response_usage.output_tokens_details.reasoning_tokens == 0
        assert response_usage.output_tokens_details.text_tokens == 50
        assert response_usage.output_tokens_details.image_tokens == 100


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
        event3 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk3)

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
        assert iterator._cached_item_id == "chatcmpl-first-id", "Should use the first chunk's ID"

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
        mock_logging_obj._response_cost_calculator = Mock(return_value=0.001)

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
