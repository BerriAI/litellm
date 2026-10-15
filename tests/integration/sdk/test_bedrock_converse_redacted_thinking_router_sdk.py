import base64
import json
from typing import Final

import pytest
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

from litellm import Router

APPLICATION_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
REDACTED: Final = base64.b64encode(b"EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP").decode()
FIRST_ANSWER: Final = "I looked into it."
FINAL_ANSWER: Final = "redacted thinking router control"


class _ConverseMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: tuple[dict[str, JsonValue], ...]


class _ConverseRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[_ConverseMessage, ...]


def _respond(_: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": FINAL_ANSWER}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


@pytest.mark.parametrize(
    "model_info, replays_redacted",
    [
        pytest.param({"base_model": "bedrock/us.anthropic.claude-sonnet-4-6"}, True, id="claude-base-model"),
        pytest.param({}, False, id="no-base-model"),
    ],
)
def test_sync_router_reads_base_model_to_replay_redacted_thinking_to_an_application_profile(
    model_info: dict[str, JsonValue], replays_redacted: bool
) -> None:
    """An application inference profile ARN hides its model, so the Router deployment's model_info.base_model is
    the only thing that tells the sync SDK path the target is Claude and can take redactedContent back"""
    with wire_server(_respond) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "claude-profile",
                    "litellm_params": {
                        "model": f"bedrock/converse/{APPLICATION_PROFILE_ARN}",
                        "api_base": wire.url,
                        "api_key": "synthetic-bedrock-bearer",
                        "aws_region_name": "us-east-1",
                    },
                    "model_info": model_info,
                }
            ],
            num_retries=0,
        )

        response: Final = router.completion(
            model="claude-profile",
            max_tokens=256,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": FIRST_ANSWER,
                    "thinking_blocks": [{"type": "redacted_thinking", "data": REDACTED}],
                },
                {"role": "user", "content": "Reply with the word ok."},
            ],
        )

        assert response.choices[0].message.content == FINAL_ANSWER
        received: Final = tuple(_ConverseRequest.model_validate_json(request.body) for request in wire.drain())
        assistant: Final = next(message for message in received[-1].messages if message.role == "assistant")
        redacted: Final = ({"reasoningContent": {"redactedContent": REDACTED}},) if replays_redacted else ()
        assert assistant.content == (*redacted, {"text": FIRST_ANSWER})
