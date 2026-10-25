import asyncio
import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.types.router import GenericLiteLLMParams


def _write_auth_record(path: Path, token: str, account_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "account_id": account_id,
                "expires_at": 4102444800,
            }
        )
    )


@pytest.fixture
def auth_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr("time.time", lambda: 1700000000)
    default_dir: Final = tmp_path / "default"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(default_dir))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    _write_auth_record(default_dir / "auth.json", "token-default", "acct-default")
    file_a: Final = tmp_path / "account-a" / "auth.json"
    file_b: Final = tmp_path / "account-b" / "auth.json"
    _write_auth_record(file_a, "token-a", "acct-a")
    _write_auth_record(file_b, "token-b", "acct-b")
    return file_a, file_b


class TestChatGPTMultiAccountChat:
    def test_validate_environment_uses_per_model_auth_file(self, auth_files: tuple[Path, Path]) -> None:
        file_a, file_b = auth_files
        config: Final = ChatGPTConfig()

        headers_a: Final = config.validate_environment(
            headers={},
            model="gpt-6.1-sol",
            messages=[],
            optional_params={},
            litellm_params={"chatgpt_auth_file": str(file_a)},
            api_key="token-default",
        )
        headers_b: Final = config.validate_environment(
            headers={},
            model="gpt-6.1-sol",
            messages=[],
            optional_params={},
            litellm_params={"chatgpt_auth_file": str(file_b)},
            api_key="token-default",
        )

        assert headers_a["Authorization"] == "Bearer token-a"
        assert headers_a["ChatGPT-Account-Id"] == "acct-a"
        assert headers_b["Authorization"] == "Bearer token-b"
        assert headers_b["ChatGPT-Account-Id"] == "acct-b"

    def test_validate_environment_defaults_to_global_auth_file(self, auth_files: tuple[Path, Path]) -> None:
        config: Final = ChatGPTConfig()

        headers: Final = config.validate_environment(
            headers={},
            model="gpt-6.1-sol",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="token-default",
        )

        assert headers["Authorization"] == "Bearer token-default"
        assert headers["ChatGPT-Account-Id"] == "acct-default"

    def test_provider_info_uses_per_model_auth_file(self, auth_files: tuple[Path, Path]) -> None:
        file_a, _ = auth_files
        config: Final = ChatGPTConfig()

        _, per_model_key, _ = config._get_openai_compatible_provider_info(
            "gpt-6.1-sol", None, None, "chatgpt", {"chatgpt_auth_file": str(file_a)}
        )
        _, default_key, _ = config._get_openai_compatible_provider_info("gpt-6.1-sol", None, None, "chatgpt")

        assert per_model_key == "token-a"
        assert default_key == "token-default"

    def test_get_llm_provider_threads_auth_file(self, auth_files: tuple[Path, Path]) -> None:
        file_a, _ = auth_files

        model, provider, dynamic_api_key, _ = get_llm_provider(
            model="chatgpt/gpt-6.1-sol",
            litellm_params=GenericLiteLLMParams(chatgpt_auth_file=str(file_a)),
        )

        assert model == "gpt-6.1-sol"
        assert provider == "chatgpt"
        assert dynamic_api_key == "token-a"


def _subscription_response(request: httpx.Request) -> httpx.Response:
    account: Final = request.headers["ChatGPT-Account-Id"]
    assert request.headers["Authorization"] == f"Bearer token-{account.removeprefix('acct-')}"
    assert "chatgpt_auth_file" not in request.content.decode()
    message: Final = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": account, "annotations": []}],
    }
    response: Final = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "gpt-6.1-sol",
        "output": [message],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    events: Final = (
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "delta": account,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": response},
    )
    return httpx.Response(
        200,
        text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("through_router", [False, True])
@pytest.mark.asyncio
async def test_requests_use_selected_subscription(
    auth_files: tuple[Path, Path], endpoint: str, through_router: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(auth_files[0].parent / "missing-default"))
    router: Final = (
        litellm.Router(
            model_list=[
                {
                    "model_name": f"account-{index}",
                    "litellm_params": {"model": "chatgpt/gpt-6.1-sol", "chatgpt_auth_file": str(path)},
                }
                for index, path in enumerate(auth_files)
            ]
        )
        if through_router
        else None
    )

    async def request(index: int, path: Path) -> str:
        params: Final = (
            {"model": f"account-{index}"}
            if router
            else {"model": "chatgpt/gpt-6.1-sol", "chatgpt_auth_file": str(path)}
        )
        if endpoint == "chat":
            result: Final = await (router.acompletion if router else litellm.acompletion)(
                **params, messages=[{"role": "user", "content": "Hello"}]
            )
            return result.choices[0].message.content
        response: Final = await (router.aresponses if router else litellm.aresponses)(
            **params, input=[{"role": "user", "content": "Hello"}]
        )
        return response.output[0].content[0].text

    with respx.mock as transport:
        transport.post("https://chatgpt.com/backend-api/codex/responses").mock(side_effect=_subscription_response)
        results: Final = await asyncio.gather(*(request(index, path) for index, path in enumerate(auth_files)))
        assert results == ["acct-a", "acct-b"]
