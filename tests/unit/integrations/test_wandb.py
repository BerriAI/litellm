from typing import Final
from unittest.mock import MagicMock

import pytest
import respx

import litellm


def test_wandb_logging(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    wandb: Final = pytest.importorskip("wandb")
    run: Final = MagicMock()
    monkeypatch.setattr(wandb, "init", lambda: run)
    monkeypatch.setattr(litellm, "success_callback", ["wandb"])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        json={
            "id": "unit-test-response",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "unit-test response"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    response: Final = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "unit-test prompt"}],
        api_key="unit-test-api-key",
    )

    assert response.choices[0].message.content == "unit-test response"
    assert route.call_count == 1
    run.log.assert_called_once()
    run.finish.assert_called_once()
