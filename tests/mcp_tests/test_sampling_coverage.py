from mcp.types import TextContent, ImageContent
from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _convert_single_content,
)


def test_convert_single_content_coverage():
    # text content
    txt = TextContent(type="text", text="hello")
    res = _convert_single_content(txt)
    assert res == {"type": "text", "text": "hello"}

    # image content
    img = ImageContent(type="image", data="base64", mimeType="image/png")
    res_img = _convert_single_content(img)
    assert res_img == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,base64"},
    }


def test_convert_mcp_tool_choice_coverage():
    from litellm.proxy._experimental.mcp_server.sampling_handler import (
        _convert_mcp_tool_choice_to_openai,
    )

    class MockToolChoice:
        def __init__(self, mode):
            self.mode = mode

    assert _convert_mcp_tool_choice_to_openai(MockToolChoice("auto")) == "auto"
    assert _convert_mcp_tool_choice_to_openai(MockToolChoice("required")) == "required"
    assert _convert_mcp_tool_choice_to_openai(MockToolChoice("none")) == "none"


def test_convert_openai_response_to_mcp_result_coverage():
    from litellm.proxy._experimental.mcp_server.sampling_handler import (
        _convert_openai_response_to_mcp_result,
    )
    from litellm import ModelResponse, Message, Choices

    # Text response
    mock_resp = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                message=Message(content="hello", role="assistant"), finish_reason="stop"
            )
        ],
        model="gpt-4o",
    )
    mcp_res = _convert_openai_response_to_mcp_result(mock_resp, "gpt-4o")
    assert mcp_res.role == "assistant"
    assert mcp_res.content.type == "text"
    assert mcp_res.content.text == "hello"
    assert mcp_res.model == "gpt-4o"
    assert mcp_res.stopReason == "endTurn"
