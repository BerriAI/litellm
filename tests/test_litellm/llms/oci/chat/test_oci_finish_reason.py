import datetime

import httpx

from litellm import ModelResponse
from litellm.llms.oci.chat.transformation import OCIChatConfig


def _transform(model: str, body: dict) -> ModelResponse:
    response = httpx.Response(
        status_code=200, json=body, headers={"Content-Type": "application/json"}
    )
    return OCIChatConfig().transform_response(
        model=model,
        raw_response=response,
        model_response=ModelResponse(),
        logging_obj={},  # type: ignore
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding={},
    )


def _cohere_body(with_tool_calls: bool) -> dict:
    chat_response: dict = {
        "apiFormat": "COHERE",
        "text": "I will look up the weather in Tokyo.",
        "finishReason": "COMPLETE",
        "usage": {"promptTokens": 26, "completionTokens": 22, "totalTokens": 48},
    }
    if with_tool_calls:
        chat_response["toolCalls"] = [
            {"name": "get_weather", "parameters": {"location": "Tokyo"}}
        ]
    return {
        "modelId": "cohere.command-latest",
        "modelVersion": "1.0",
        "chatResponse": chat_response,
    }


def _generic_body_with_tool_calls() -> dict:
    created = (
        datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    return {
        "modelId": "meta.llama-3.3-70b-instruct",
        "modelVersion": "1.0",
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": "Calling a tool."}],
                        "toolCalls": [
                            {
                                "id": "call_0",
                                "type": "FUNCTION",
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            }
                        ],
                    },
                    # OCI GENERIC may report STOP even when tool calls are present.
                    "finishReason": "STOP",
                }
            ],
            "timeCreated": created,
            "usage": {"promptTokens": 5, "completionTokens": 10, "totalTokens": 15},
        },
    }


def test_cohere_tool_calls_report_finish_reason_tool_calls():
    # OCI Cohere returns finishReason="COMPLETE" for tool calls; finish_reason
    # must still be "tool_calls" when the message carries them.
    result = _transform("cohere.command-latest", _cohere_body(with_tool_calls=True))
    assert result.choices[0].message.tool_calls is not None
    assert result.choices[0].finish_reason == "tool_calls"


def test_cohere_without_tool_calls_finish_reason_stop():
    # Sanity: a plain COMPLETE response still maps to "stop".
    result = _transform("cohere.command-latest", _cohere_body(with_tool_calls=False))
    assert result.choices[0].message.tool_calls is None
    assert result.choices[0].finish_reason == "stop"


def test_generic_tool_calls_report_finish_reason_tool_calls():
    result = _transform("meta.llama-3.3-70b-instruct", _generic_body_with_tool_calls())
    assert result.choices[0].message.tool_calls is not None
    assert result.choices[0].finish_reason == "tool_calls"
