import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
import respx

from litellm.llms.github_copilot.authenticator import Authenticator, github_api_headers
from litellm.llms.github_copilot.common_utils import GetAccessTokenError, GetDeviceCodeError

COPILOT_USER_URL: Final = "https://api.github.com/copilot_internal/user"
DEVICE_CODE_URL: Final = "https://github.com/login/device/code"
TENANT_API: Final = "https://api.tenant.githubcopilot.com"
TRUNCATED_UTF_16: Final = b"\xff\xfe\x00\xd8"


@pytest.fixture
def logged_in_token_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    token: Final = f"gho_{tmp_path.name}"
    (tmp_path / "access-token").write_text(token)
    return tmp_path, token


def test_init_uses_access_token_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    auth: Final = Authenticator()
    assert auth.token_dir == str(tmp_path)
    assert auth.access_token_file == str(tmp_path / "access-token")


def test_init_uses_default_access_token_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN_DIR", raising=False)

    auth: Final = Authenticator()

    assert auth.token_dir == str(tmp_path / ".config/litellm/github_copilot")
    assert auth.access_token_file == str(tmp_path / ".config/litellm/github_copilot/access-token")


def test_ensure_token_dir_creates_missing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_dir: Final = str(tmp_path / "github_copilot")
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", token_dir)

    auth: Final = Authenticator()

    assert Path(auth.token_dir).is_dir()


def test_get_github_headers_include_access_token() -> None:
    headers: Final = github_api_headers("github-token")

    assert headers == {
        "accept": "application/json",
        "editor-version": "vscode/1.85.1",
        "editor-plugin-version": "copilot/1.155.0",
        "user-agent": "GithubCopilot/1.155.0",
        "accept-encoding": "gzip,deflate,br",
        "content-type": "application/json",
        "authorization": "token github-token",
    }


def test_get_access_token_reads_saved_token(logged_in_token_dir: tuple[Path, str]) -> None:
    _, token = logged_in_token_dir
    assert Authenticator().get_access_token() == token


def test_get_access_token_logs_in_and_saves_token(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    token_dir, _ = logged_in_token_dir
    access_token_file: Final = token_dir / "access-token"
    access_token_file.unlink()
    auth: Final = Authenticator()

    with patch.object(auth, "_login", return_value="github-access-token") as login:
        assert auth.get_access_token() == "github-access-token"

    login.assert_called_once()
    assert access_token_file.read_text() == "github-access-token"


def test_get_api_key_returns_github_token_without_http(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    _, token = logged_in_token_dir
    with patch("litellm.llms.github_copilot.authenticator.get_httpx_client") as get_client:
        assert Authenticator().get_api_key() == token
    get_client.assert_not_called()


def test_get_api_base_uses_user_endpoint_and_memoizes_per_access_token(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    _, token = logged_in_token_dir
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.get(COPILOT_USER_URL).respond(json={"endpoints": {"api": TENANT_API}})
        auth: Final = Authenticator()
        assert auth.get_api_base() == TENANT_API
        assert Authenticator().get_api_base() == TENANT_API

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == f"token {token}"


@pytest.mark.parametrize("status_code", [401, 403, 500])
def test_get_api_base_returns_none_when_user_lookup_fails(
    logged_in_token_dir: tuple[Path, str], status_code: int
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(COPILOT_USER_URL).respond(status_code=status_code)
        assert Authenticator().get_api_base() is None


def test_get_api_base_retries_after_failed_user_lookup(logged_in_token_dir: tuple[Path, str]) -> None:
    auth: Final = Authenticator()
    with respx.mock(assert_all_called=True) as router:
        failed_lookup: Final = router.get(COPILOT_USER_URL).respond(status_code=503)
        assert auth.get_api_base() is None
    assert failed_lookup.call_count == 1

    with respx.mock(assert_all_called=True) as router:
        successful_lookup: Final = router.get(COPILOT_USER_URL).respond(json={"endpoints": {"api": TENANT_API}})
        assert auth.get_api_base() == TENANT_API
    assert successful_lookup.call_count == 1


def test_get_api_base_returns_none_when_user_lookup_response_has_no_endpoint(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(COPILOT_USER_URL).respond(json={"endpoints": {}})
        assert Authenticator().get_api_base() is None


def test_get_api_base_returns_none_when_user_lookup_endpoints_are_not_a_mapping(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(COPILOT_USER_URL).respond(json={"endpoints": []})
        assert Authenticator().get_api_base() is None


def test_get_api_base_rejects_untrusted_user_endpoint(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(COPILOT_USER_URL).respond(json={"endpoints": {"api": "https://attacker.example"}})
        assert Authenticator().get_api_base() is None


def test_get_api_base_does_not_trigger_device_login_without_saved_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    with patch("litellm.llms.github_copilot.authenticator.get_httpx_client") as get_client:
        assert Authenticator().get_api_base() is None
    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_get_access_token_refuses_device_code_login_in_event_loop() -> None:
    auth: Final = Authenticator()
    with (
        patch("builtins.open", side_effect=FileNotFoundError),
        patch.object(auth, "_login") as login,
    ):
        with pytest.raises(GetAccessTokenError) as exc:
            auth.get_access_token()

    assert exc.value.status_code == 401
    assert "event loop" in str(exc.value)
    login.assert_not_called()


def test_get_access_token_refuses_device_code_login_in_worker_thread() -> None:
    auth: Final = Authenticator()
    with (
        patch("builtins.open", side_effect=FileNotFoundError),
        patch.object(auth, "_login") as login,
    ):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(GetAccessTokenError) as exc:
                pool.submit(auth.get_access_token).result()

    assert exc.value.status_code == 401
    assert "worker thread" in str(exc.value)
    login.assert_not_called()


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-8-sig"])
def test_get_access_token_reads_saved_token_with_supported_encoding(
    logged_in_token_dir: tuple[Path, str], encoding: str
) -> None:
    token_dir, token = logged_in_token_dir
    (token_dir / "access-token").write_bytes(f"{token}\n".encode(encoding))

    assert Authenticator().get_access_token() == token


def test_get_access_token_refuses_undecodable_saved_token(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    token_dir, _ = logged_in_token_dir
    access_token_file: Final = token_dir / "access-token"
    access_token_file.write_bytes(TRUNCATED_UTF_16)

    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(GetAccessTokenError) as exc:
        pool.submit(Authenticator().get_access_token).result()

    assert exc.value.status_code == 401
    assert "worker thread" in str(exc.value)
    assert access_token_file.read_bytes() == TRUNCATED_UTF_16


def test_get_access_token_retries_device_flow_three_times(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    token_dir, _ = logged_in_token_dir
    (token_dir / "access-token").unlink()
    auth: Final = Authenticator()

    with (
        patch.object(
            auth,
            "_login",
            side_effect=GetDeviceCodeError(message="device flow failed", status_code=400),
        ) as login,
        pytest.raises(GetAccessTokenError, match="after 3 attempts"),
    ):
        auth.get_access_token()

    assert login.call_count == 3


def test_get_access_token_returns_login_token_when_file_write_fails(
    logged_in_token_dir: tuple[Path, str],
) -> None:
    token_dir, _ = logged_in_token_dir
    (token_dir / "access-token").unlink()
    auth: Final = Authenticator()

    with (
        patch.object(auth, "_login", return_value="github-access-token"),
        patch("builtins.open", side_effect=(FileNotFoundError, PermissionError)),
    ):
        assert auth.get_access_token() == "github-access-token"


@pytest.mark.parametrize(
    ("environment", "url", "body"),
    [
        (
            {"GITHUB_COPILOT_DEVICE_CODE_URL": "https://custom.example/device"},
            "https://custom.example/device",
            {"client_id": "Iv1.b507a08c87ecfe98", "scope": "read:user"},
        ),
        (
            {"GITHUB_COPILOT_CLIENT_ID": "custom-client"},
            DEVICE_CODE_URL,
            {"client_id": "custom-client", "scope": "read:user"},
        ),
    ],
)
def test_get_device_code_uses_runtime_url_and_client_id(
    environment: dict[str, str], url: str, body: dict[str, str]
) -> None:
    client: Final = MagicMock()
    response: Final = MagicMock()
    response.json.return_value = {
        "device_code": "dc",
        "user_code": "UC",
        "verification_uri": "https://github.com/login/device",
    }
    response.raise_for_status.return_value = None
    client.post.return_value = response
    with (
        patch.dict(os.environ, environment),
        patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=client),
    ):
        assert Authenticator()._get_device_code()["device_code"] == "dc"
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == url
    assert client.post.call_args.kwargs["json"] == body


def test_get_device_code_sends_device_flow_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_DEVICE_CODE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_CLIENT_ID", raising=False)
    client: Final = MagicMock()
    response: Final = MagicMock()
    device_code: Final = {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://github.com/login/device",
    }
    response.json.return_value = device_code
    response.raise_for_status.return_value = None
    client.post.return_value = response

    with patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=client):
        assert Authenticator()._get_device_code() == device_code

    client.post.assert_called_once_with(
        DEVICE_CODE_URL,
        headers=github_api_headers(),
        json={"client_id": "Iv1.b507a08c87ecfe98", "scope": "read:user"},
    )


def test_poll_for_access_token_sends_device_code_and_grant_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_CLIENT_ID", raising=False)
    client: Final = MagicMock()
    response: Final = MagicMock()
    response.json.return_value = {"access_token": "github-access-token"}
    response.raise_for_status.return_value = None
    client.post.return_value = response

    with patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=client):
        assert Authenticator()._poll_for_access_token("device-code") == "github-access-token"

    client.post.assert_called_once_with(
        "https://github.com/login/oauth/access_token",
        headers=github_api_headers(),
        json={
            "client_id": "Iv1.b507a08c87ecfe98",
            "device_code": "device-code",
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )


def test_poll_for_access_token_uses_runtime_url_and_client_id() -> None:
    client: Final = MagicMock()
    response: Final = MagicMock()
    response.json.return_value = {"access_token": "github-token"}
    response.raise_for_status.return_value = None
    client.post.return_value = response
    with (
        patch.dict(
            os.environ,
            {
                "GITHUB_COPILOT_ACCESS_TOKEN_URL": "https://custom.example/token",
                "GITHUB_COPILOT_CLIENT_ID": "custom-client",
            },
        ),
        patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=client),
        patch("time.sleep"),
    ):
        assert Authenticator()._poll_for_access_token("dc") == "github-token"
    assert client.post.call_args.args[0] == "https://custom.example/token"
    assert client.post.call_args.kwargs["json"]["client_id"] == "custom-client"


def test_login_requests_device_code_and_polls_for_access_token() -> None:
    auth: Final = Authenticator()
    device_code: Final = {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://github.com/login/device",
    }

    with (
        patch.object(auth, "_get_device_code", return_value=device_code) as get_device_code,
        patch.object(auth, "_poll_for_access_token", return_value="github-access-token") as poll,
        patch("builtins.print") as print_message,
    ):
        assert auth._login() == "github-access-token"

    get_device_code.assert_called_once()
    poll.assert_called_once_with("device-code")
    print_message.assert_called_once_with(
        "Please visit https://github.com/login/device and enter code ABCD-EFGH to authenticate.",
        flush=True,
    )
