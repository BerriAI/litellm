from unittest.mock import MagicMock

import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.common_utils import GetAccessTokenError

GATEWAY_BASE = "https://chatgpt.example/backend-api/codex"


def _gateway_authenticator() -> MagicMock:
    authenticator = MagicMock()
    authenticator.get_api_base.return_value = GATEWAY_BASE
    authenticator.get_access_token.return_value = "gateway-access-token"
    return authenticator


def _provider_info(
    config: ChatGPTConfig, api_base: str | None, api_key: str | None
) -> tuple[str | None, str | None, str]:
    return config._get_openai_compatible_provider_info(
        model="gpt-5.4", api_base=api_base, api_key=api_key, custom_llm_provider="chatgpt"
    )


class TestChatGPTChatProviderInfo:
    def test_configured_key_uses_configured_base_without_gateway_login(self):
        authenticator = _gateway_authenticator()
        config = ChatGPTConfig(authenticator=authenticator)

        assert _provider_info(config, "http://127.0.0.1:9999", "configured-key") == (
            "http://127.0.0.1:9999",
            "configured-key",
            "chatgpt",
        )
        authenticator.get_access_token.assert_not_called()

    def test_configured_key_without_base_uses_gateway_base(self):
        config = ChatGPTConfig(authenticator=_gateway_authenticator())

        assert _provider_info(config, None, "configured-key") == (GATEWAY_BASE, "configured-key", "chatgpt")

    def test_gateway_login_is_never_sent_to_a_custom_base(self):
        config = ChatGPTConfig(authenticator=_gateway_authenticator())

        assert _provider_info(config, "http://attacker.example", None) == (
            GATEWAY_BASE,
            "gateway-access-token",
            "chatgpt",
        )

    def test_missing_gateway_login_raises_authentication_error(self):
        authenticator = _gateway_authenticator()
        authenticator.get_access_token.side_effect = GetAccessTokenError(status_code=401, message="not logged in")
        config = ChatGPTConfig(authenticator=authenticator)

        with pytest.raises(AuthenticationError, match="not logged in"):
            _provider_info(config, None, None)


import asyncio
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import respx

import litellm
from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager
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
    assert request.headers["Authorization"] in (
        f"Bearer token-{account.removeprefix('acct-')}",
        f"Bearer {_client_token(account)}",
    )
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


@pytest.mark.parametrize("blocked_default", [False, True])
@pytest.mark.parametrize("with_prompt", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("through_router", [False, True])
@pytest.mark.asyncio
async def test_requests_use_selected_subscription(
    auth_files: tuple[Path, Path],
    endpoint: str,
    through_router: bool,
    with_prompt: bool,
    blocked_default: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    default_dir: Final = auth_files[0].parent / "missing-default"
    if blocked_default:
        default_dir.write_text("ordinary file")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(default_dir / "unused"))
    if with_prompt:
        monkeypatch.setattr(
            litellm,
            "callbacks",
            [
                DotpromptManager(
                    prompt_id="account-check",
                    prompt_data={"content": "Saved account prompt", "metadata": {"model": "chatgpt/gpt-6.1-sol"}},
                )
            ],
        )
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
        params: Final = MappingProxyType(
            {
                "model": f"account-{index}" if router else "chatgpt/gpt-6.1-sol",
                **({"chatgpt_auth_file": str(path)} if router is None else {}),
                **({"prompt_id": "account-check"} if with_prompt else {}),
            }
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
        if with_prompt:
            assert all("Saved account prompt" in call.request.content.decode() for call in transport.calls)


def _client_token(account: str) -> str:
    import base64

    payload: Final = (
        base64.urlsafe_b64encode(json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account}}).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"header.{payload}.signature"


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("configured_login", [False, True])
@pytest.mark.parametrize("with_prompt", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_client_login_keeps_its_account_across_gateway_deployments(
    auth_files: tuple[Path, Path],
    endpoint: str,
    configured_login: bool,
    with_prompt: bool,
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(auth_files[0].parent / "no-gateway-login"))
    if with_prompt:
        monkeypatch.setattr(
            litellm,
            "callbacks",
            [
                DotpromptManager(
                    prompt_id="client-account-check",
                    prompt_data={"content": "Saved client prompt", "metadata": {"model": "chatgpt/gpt-6.1-sol"}},
                )
            ],
        )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "subscriptions",
                "litellm_params": {
                    "model": "chatgpt/gpt-6.1-sol",
                    **({"chatgpt_auth_file": str(path)} if configured_login else {}),
                },
            }
            for path in auth_files
        ],
        num_retries=0,
    )

    def reply(request: httpx.Request) -> httpx.Response:
        account: Final = request.headers["ChatGPT-Account-Id"]
        assert request.headers["Authorization"] == f"Bearer {_client_token(account)}"
        return _subscription_response(
            httpx.Request(
                "POST",
                request.url,
                content=request.content,
                headers={
                    **{key: value for key, value in request.headers.items() if key.lower() != "authorization"},
                    "Authorization": f"Bearer token-{account.removeprefix('acct-')}",
                },
            )
        )

    async def call(account: str) -> str:
        params: Final = MappingProxyType(
            {
                "model": "subscriptions",
                "stream": stream,
                **({"prompt_id": "client-account-check"} if with_prompt else {}),
                "provider_specific_header": {
                    "custom_llm_provider": "chatgpt",
                    "extra_headers": {"Authorization": f"Bearer {_client_token(account)}"},
                },
            }
        )
        response: Final = await (
            router.acompletion(**params, messages=[{"role": "user", "content": "Hello"}])
            if endpoint == "chat"
            else router.aresponses(**params, input=[{"role": "user", "content": "Hello"}])
        )
        if stream:
            chunks: Final = [part async for part in response]
            return (
                "".join(part.choices[0].delta.content or "" for part in chunks)
                if endpoint == "chat"
                else "".join(
                    getattr(part, "delta", "")
                    for part in chunks
                    if getattr(part, "type", "") == "response.output_text.delta"
                )
            )
        return response.choices[0].message.content if endpoint == "chat" else response.output[0].content[0].text

    with respx.mock as transport:
        transport.post("https://chatgpt.com/backend-api/codex/responses").mock(side_effect=reply)
        results: Final = await asyncio.gather(call("acct-client-a"), call("acct-client-b"))
        assert results == ["acct-client-a", "acct-client-b"]
        assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_one_client_limit_does_not_cool_down_other_subscriptions(
    auth_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "pool", "litellm_params": {"model": "chatgpt/gpt-6.1-sol", "chatgpt_auth_file": str(path)}}
            for path in auth_files
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )
    headers: Final = {
        "custom_llm_provider": "chatgpt",
        "extra_headers": {"Authorization": f"Bearer {_client_token('acct-limited')}"},
    }
    with respx.mock as transport:
        route: Final = transport.post("https://chatgpt.com/backend-api/codex/responses")
        route.respond(
            429,
            headers={"retry-after": "60"},
            json={"error": {"message": "Usage limit reached", "type": "rate_limit_error"}},
        )
        for _ in range(2):
            with pytest.raises(litellm.RateLimitError):
                await router.aresponses(model="pool", input="Hello", provider_specific_header=headers)
            await asyncio.sleep(0)
        route.mock(side_effect=_subscription_response)
        response: Final = await router.aresponses(model="pool", input="Hello")
        assert response.output[0].content[0].text in ("acct-a", "acct-b")


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("direct_headers", [False, True])
@pytest.mark.parametrize("configured_key", [False, True])
@pytest.mark.parametrize(
    "target", ["https://example.com/v1", "http://chatgpt.com/backend-api/codex", "https://chatgpt.com.example.com/v1"]
)
@pytest.mark.asyncio
async def test_client_login_cannot_be_redirected_to_another_host(
    endpoint: str, target: str, async_call: bool, direct_headers: bool, configured_key: bool
) -> None:
    headers: Final = {"Authorization": f"Bearer {_client_token('acct-client')}"}
    credential: Final = (
        {"extra_headers": headers}
        if direct_headers
        else {"provider_specific_header": {"custom_llm_provider": "chatgpt", "extra_headers": headers}}
    )
    params: Final = MappingProxyType(
        {
            "model": "chatgpt/gpt-6.1-sol",
            "api_base": target,
            **({"api_key": "configured-test-key"} if configured_key else {}),
            **credential,
        }
    )

    async def request() -> None:
        if endpoint == "chat":
            if async_call:
                await litellm.acompletion(**params, messages=[{"role": "user", "content": "Hello"}])
            else:
                litellm.completion(**params, messages=[{"role": "user", "content": "Hello"}])
        elif async_call:
            await litellm.aresponses(**params, input="Hello")
        else:
            litellm.responses(**params, input="Hello")

    with respx.mock as transport:
        with pytest.raises(litellm.BadRequestError, match="official HTTPS ChatGPT endpoint"):
            await request()
        assert len(transport.calls) == 0


@pytest.mark.asyncio
async def test_cross_provider_fallback_does_not_receive_the_chatgpt_login(
    auth_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "subscription",
                "litellm_params": {"model": "chatgpt/gpt-6.1-sol", "chatgpt_auth_file": str(auth_files[0])},
            },
            {
                "model_name": "api-fallback",
                "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": "sk-example-fallback"},
            },
        ],
        fallbacks=[{"subscription": ["api-fallback"]}],
        num_retries=0,
    )
    with respx.mock as transport:
        transport.post("https://chatgpt.com/backend-api/codex/responses").respond(
            429, json={"error": {"message": "Usage limit reached", "type": "rate_limit_error"}}
        )
        fallback: Final = transport.post("https://api.openai.com/v1/responses").respond(
            200,
            json={
                "id": "resp_fallback",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "output": [
                    {
                        "id": "msg_fallback",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "fallback", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )
        response: Final = await router.aresponses(
            model="subscription",
            input="Hello",
            provider_specific_header={
                "custom_llm_provider": "chatgpt",
                "extra_headers": {
                    "Authorization": f"Bearer {_client_token('acct-client')}",
                    "ChatGPT-Account-Id": "acct-client",
                },
            },
        )
        assert response.output[0].content[0].text == "fallback"
        assert fallback.calls.last.request.headers["Authorization"] == "Bearer sk-example-fallback"
        assert "ChatGPT-Account-Id" not in fallback.calls.last.request.headers


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("with_prompt", [False, True])
@pytest.mark.asyncio
async def test_sdk_extra_headers_use_client_login_without_gateway_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, async_call: bool, with_prompt: bool
) -> None:
    blocked: Final = tmp_path / "blocked"
    blocked.write_text("ordinary file")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(blocked / "unused"))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    headers: Final = {"Authorization": f"Bearer {_client_token('acct-client-a')}"}
    prompt_params: Final = {"prompt_id": "sdk-client"} if with_prompt else {}
    if with_prompt:
        monkeypatch.setattr(
            litellm,
            "callbacks",
            [
                DotpromptManager(
                    prompt_id="sdk-client",
                    prompt_data={"content": "Hello", "metadata": {"model": "chatgpt/gpt-6-luna"}},
                )
            ],
        )
    with respx.mock as transport:
        transport.post("https://chatgpt.com/backend-api/codex/responses").mock(side_effect=_subscription_response)
        if endpoint == "chat":
            params: Final = {
                "model": "chatgpt/gpt-6.1-sol",
                "messages": [{"role": "user", "content": "Hello"}],
                "extra_headers": headers,
                **prompt_params,
            }
            result: Final = await litellm.acompletion(**params) if async_call else litellm.completion(**params)
            assert result.choices[0].message.content == "acct-client-a"
        else:
            response: Final = (
                await litellm.aresponses(
                    model="chatgpt/gpt-6.1-sol", input="Hello", extra_headers=headers, **prompt_params
                )
                if async_call
                else litellm.responses(
                    model="chatgpt/gpt-6.1-sol", input="Hello", extra_headers=headers, **prompt_params
                )
            )
            assert response.output[0].content[0].text == "acct-client-a"
        assert transport.calls.last.request.headers["Authorization"] == headers["Authorization"]
        assert len(transport.calls) == 1


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("credential_source", ["default", "file", "api_key"])
@pytest.mark.parametrize("authorization_header", [None, "Authorization", "authorization"])
@pytest.mark.asyncio
async def test_requests_preserve_gateway_token_and_selected_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    sync: bool,
    credential_source: str,
    authorization_header: str | None,
) -> None:
    monkeypatch.setattr("time.time", lambda: 1700000000)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("CHATGPT_AUTH_FILE", str(tmp_path / "default.json"))
    token: Final = _client_token("acct-token-default")
    auth_file: Final = tmp_path / ("selected.json" if credential_source == "file" else "default.json")
    _write_auth_record(auth_file, token, "acct-saved-workspace")
    configured_token: Final = _client_token("acct-configured")
    expected_token: Final = configured_token if credential_source == "api_key" else token
    expected_account: Final = "acct-configured" if credential_source == "api_key" else "acct-saved-workspace"

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.headers.get_list("authorization") == [f"Bearer {expected_token}"]
        assert request.headers.get_list("chatgpt-account-id") == [expected_account]
        assert request.headers["originator"] == "codex_cli_rs"
        return _subscription_response(
            httpx.Request(
                "POST",
                request.url,
                content=request.content,
                headers={
                    "Authorization": f"Bearer token-{expected_account.removeprefix('acct-')}",
                    "ChatGPT-Account-Id": expected_account,
                },
            )
        )

    params: Final = MappingProxyType(
        {
            "model": "chatgpt/gpt-6.1-sol",
            **({"chatgpt_auth_file": str(auth_file)} if credential_source == "file" else {}),
            **({"api_key": configured_token} if credential_source == "api_key" else {}),
            "extra_headers": {
                "originator": "codex_cli_rs",
                **({authorization_header: "Bearer sk-unrelated-example"} if authorization_header else {}),
            },
        }
    )
    with respx.mock as transport:
        transport.post("https://chatgpt.com/backend-api/codex/responses").mock(side_effect=reply)
        if endpoint == "chat":
            result: Final = (
                await asyncio.to_thread(litellm.completion, **params, messages=[{"role": "user", "content": "Hello"}])
                if sync
                else await litellm.acompletion(**params, messages=[{"role": "user", "content": "Hello"}])
            )
            assert result.choices[0].message.content == expected_account
        else:
            response: Final = (
                await asyncio.to_thread(litellm.responses, **params, input=[{"role": "user", "content": "Hello"}])
                if sync
                else await litellm.aresponses(**params, input=[{"role": "user", "content": "Hello"}])
            )
            assert response.output[0].content[0].text == expected_account
