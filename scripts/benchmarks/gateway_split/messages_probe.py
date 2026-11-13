from typing import Final

from anthropic import Anthropic


def verify_sdk(port: int, delegate: bool) -> None:
    with Anthropic(
        base_url=f"http://127.0.0.1:{port}",
        api_key="fixture-key",
        max_retries=0,
        timeout=5,
        default_headers={"x-bench-python": "1"} if delegate else {},
    ) as client:
        response: Final = client.messages.create(
            model="fixture-model",
            max_tokens=32,
            system=[{"type": "text", "text": "Use the provided tool result", "cache_control": {"type": "ephemeral"}}],
            tools=[{"name": "lookup", "description": "Look up a value", "input_schema": {"type": "object"}}],
            messages=[
                {"role": "user", "content": "Find the answer"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tool_fixture", "name": "lookup", "input": {}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_fixture", "content": "42"}]},
            ],
        )
        assert response.type == "message"
        assert response.stop_reason == "end_turn"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "x" * 1024
        with client.messages.stream(
            model="fixture-model", max_tokens=32, messages=[{"role": "user", "content": "hello"}]
        ) as stream:
            text: Final = "".join(stream.text_stream)
            final: Final = stream.get_final_message()
            assert text == response.content[0].text
            assert final.stop_reason == response.stop_reason
            assert final.usage == response.usage
