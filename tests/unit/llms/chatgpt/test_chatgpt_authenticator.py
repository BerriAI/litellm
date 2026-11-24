import base64
import json
import time
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, mock_open, patch

import pytest
import httpx
import respx

import litellm
from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS, HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS
from litellm.llms.chatgpt.authenticator import (
    DEVICE_CODE_POLL_SLEEP_SECONDS,
    TOKEN_REFRESH_TIMEOUT_SECONDS,
    Authenticator,
    ChatGPTAuthorizationCode,
    ChatGPTTokens,
    aexchange_chatgpt_authorization_code,
    apoll_chatgpt_device_login,
    arefresh_chatgpt_tokens,
    astart_chatgpt_device_login,
    chatgpt_login_values,
    get_cached_authenticator,
    get_chatgpt_auth_file,
)
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_OAUTH_TOKEN_URL,
    ChatGPTAuthError,
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAccessTokenError,
)
from litellm.types.router import GenericLiteLLMParams


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    @pytest.mark.parametrize(
        ("request_timeout", "expected_read"),
        [
            (DEFAULT_REQUEST_TIMEOUT_SECONDS, TOKEN_REFRESH_TIMEOUT_SECONDS),
            (TOKEN_REFRESH_TIMEOUT_SECONDS * 4.0, TOKEN_REFRESH_TIMEOUT_SECONDS),
            (TOKEN_REFRESH_TIMEOUT_SECONDS / 3.0, TOKEN_REFRESH_TIMEOUT_SECONDS / 3.0),
        ],
    )
    def test_refresh_tokens_uses_bounded_timeout(self, authenticator, monkeypatch, request_timeout, expected_read):
        monkeypatch.setattr(litellm, "request_timeout", request_timeout)
        monkeypatch.setattr(litellm, "request_timeout_explicitly_set", False)
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "access_token": "token-new",
            "id_token": "id-123",
        }
        client.post.return_value = response

        with patch(  # test-quality-ok: requested seam for asserting timeout propagation
            "litellm.llms.chatgpt.authenticator.get_httpx_client", return_value=client
        ):
            refreshed = authenticator._refresh_tokens("refresh-123")

        assert refreshed["access_token"] == "token-new"
        timeout = client.post.call_args.kwargs["timeout"]
        assert timeout.read == expected_read
        assert timeout.connect == HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_get_access_token_refuses_device_code_login_in_event_loop(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login_device_code") as mock_login,
            patch.object(authenticator, "_wait_for_access_token") as mock_wait,
        ):
            with pytest.raises(GetAccessTokenError) as exc:
                authenticator.get_access_token()

        assert exc.value.status_code == 401
        assert "event loop" in str(exc.value)
        assert authenticator.auth_file not in str(exc.value)
        mock_login.assert_not_called()
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_access_token_refuses_cooldown_wait_in_event_loop(self, authenticator):
        auth_data = json.dumps({"device_code_requested_at": time.time()})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_login_device_code") as mock_login,
            patch.object(authenticator, "_wait_for_access_token") as mock_wait,
        ):
            with pytest.raises(GetAccessTokenError) as exc:
                authenticator.get_access_token()

        assert exc.value.status_code == 401
        assert "event loop" in str(exc.value)
        assert authenticator.auth_file not in str(exc.value)
        mock_login.assert_not_called()
        mock_wait.assert_not_called()

    def test_get_access_token_refuses_device_code_login_in_worker_thread(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login_device_code") as mock_login,
            patch.object(authenticator, "_wait_for_access_token") as mock_wait,
        ):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(GetAccessTokenError) as exc:
                    pool.submit(authenticator.get_access_token).result()

        assert exc.value.status_code == 401
        assert "worker thread" in str(exc.value)
        assert authenticator.auth_file not in str(exc.value)
        mock_login.assert_not_called()
        mock_wait.assert_not_called()

    def test_get_access_token_device_code_login_without_event_loop(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login_device_code", return_value={"access_token": "tok"}),
        ):
            token = authenticator.get_access_token()

        assert token == "tok"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"

    def test_get_account_id_prefers_stored_value_without_rewriting_file(self, authenticator):
        id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-from-token"}})
        auth_data = json.dumps({"account_id": "acct-stored", "id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            assert authenticator.get_account_id() == "acct-stored"
            mock_write.assert_not_called()

    def test_get_account_id_without_login_returns_none(self, authenticator):
        with (
            patch("builtins.open", mock_open(read_data="{}")),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            assert authenticator.get_account_id() is None
            mock_write.assert_not_called()


def _write_auth_record(path: Path, token: str, account_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "account_id": account_id,
                "expires_at": 4102444800,
            }
        )
    )


class TestChatGPTMultiAccountAuthenticator:
    @pytest.fixture(autouse=True)
    def fixed_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.time", lambda: 1700000000)

    def test_explicit_auth_file_overrides_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "env-dir"))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "env.json")
        custom_file: Final = tmp_path / "account-a" / "auth.json"

        authenticator: Final = Authenticator(auth_file=str(custom_file))

        assert authenticator.auth_file == str(custom_file)
        assert authenticator.token_dir == str(custom_file.parent)
        assert not custom_file.parent.exists()
        authenticator._write_auth_file({"access_token": "token-a"})
        assert json.loads(custom_file.read_text())["access_token"] == "token-a"

    def test_default_auth_file_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "env-dir"))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "env.json")

        authenticator: Final = Authenticator()

        assert authenticator.auth_file == str(tmp_path / "env-dir" / "env.json")

    def test_bare_filename_auth_file_resolves_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        authenticator: Final = Authenticator(auth_file="auth.json")

        assert authenticator.auth_file == str(tmp_path / "auth.json")
        assert authenticator.token_dir == str(tmp_path)

    def test_two_auth_files_are_isolated(self, tmp_path: Path) -> None:
        file_a: Final = tmp_path / "account-a.json"
        file_b: Final = tmp_path / "account-b.json"
        _write_auth_record(file_a, "token-a", "acct-a")
        _write_auth_record(file_b, "token-b", "acct-b")

        authenticator_a: Final = Authenticator(auth_file=str(file_a))
        authenticator_b: Final = Authenticator(auth_file=str(file_b))

        assert authenticator_a.get_access_token() == "token-a"
        assert authenticator_a.get_account_id() == "acct-a"
        assert authenticator_b.get_access_token() == "token-b"
        assert authenticator_b.get_account_id() == "acct-b"

    def test_get_cached_authenticator_reuses_instance_per_path(self, tmp_path: Path) -> None:
        file_a: Final = tmp_path / "account-a.json"
        file_b: Final = tmp_path / "account-b.json"

        authenticator_a: Final = get_cached_authenticator(str(file_a))

        assert get_cached_authenticator(str(file_a)) is authenticator_a
        assert get_cached_authenticator(str(file_b)) is not authenticator_a

    def test_get_chatgpt_auth_file(self) -> None:
        assert get_chatgpt_auth_file(None) is None
        assert get_chatgpt_auth_file({}) is None
        assert get_chatgpt_auth_file({"chatgpt_auth_file": ""}) is None
        assert get_chatgpt_auth_file({"chatgpt_auth_file": "/a/auth.json"}) == "/a/auth.json"
        assert get_chatgpt_auth_file(GenericLiteLLMParams()) is None
        assert get_chatgpt_auth_file(GenericLiteLLMParams(chatgpt_auth_file="/b/auth.json")) == "/b/auth.json"

    def test_relative_and_absolute_auth_file_share_cached_authenticator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        get_cached_authenticator.cache_clear()
        auth_file: Final = tmp_path / "x" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text("{}")

        relative_key: Final = get_chatgpt_auth_file({"chatgpt_auth_file": "x/auth.json"})
        absolute_key: Final = get_chatgpt_auth_file({"chatgpt_auth_file": str(auth_file)})
        assert relative_key is not None and absolute_key is not None

        assert get_cached_authenticator(relative_key) is get_cached_authenticator(absolute_key)


def test_refresh_keeps_saved_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.time", lambda: 1700000000)
    auth_file: Final = tmp_path / "selected.json"
    auth_file.write_text(
        json.dumps(
            {
                "access_token": "expired-example-token",
                "refresh_token": "example-refresh-token",
                "expires_at": 1,
                "account_id": "acct-selected-workspace",
            }
        )
    )
    token: Final = _make_jwt({"exp": 4102444800, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-default"}})
    authenticator: Final = Authenticator(auth_file=str(auth_file))
    with respx.mock as transport:
        transport.post("https://auth.openai.com/oauth/token").mock(
            return_value=httpx.Response(
                200,
                json={"access_token": token, "id_token": token, "refresh_token": "example-rotated-refresh"},
            )
        )
        assert authenticator.get_access_token() == token
    assert authenticator.get_account_id(token) == "acct-selected-workspace"
    assert json.loads(auth_file.read_text())["refresh_token"] == "example-rotated-refresh"


SECRET_IN_RESPONSE: Final = "secret-from-the-oauth-response"


@dataclass(frozen=True)
class _Post:
    url: str
    data: object
    json: object


class _FakeOAuthClient:
    def __init__(self, status_code: int = 200, body: object = None, error: Exception | None = None) -> None:
        self.status_code: Final = status_code
        self.body: Final = body
        self.error: Final = error
        self.posts: Final[list[_Post]] = []

    async def post(
        self,
        url: str,
        data: dict[str, str] | str | bytes | None = None,
        json: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        self.posts.append(_Post(url=url, data=data, json=json))
        if self.error is not None:
            raise self.error
        request: Final = httpx.Request("POST", url)
        response: Final = (
            httpx.Response(self.status_code, content=self.body, request=request)
            if isinstance(self.body, bytes)
            else httpx.Response(self.status_code, json=self.body, request=request)
        )
        response.raise_for_status()
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("user_code_field", ["user_code", "usercode"])
async def test_astart_chatgpt_device_login_reads_the_user_code_under_either_field_name(user_code_field: str) -> None:
    client: Final = _FakeOAuthClient(
        body={"device_auth_id": "device-auth-example", user_code_field: "ABCD-1234", "interval": "9"}
    )

    login: Final = await astart_chatgpt_device_login(client)

    assert (login.device_auth_id, login.user_code, login.interval) == ("device-auth-example", "ABCD-1234", 9)
    assert client.posts == [_Post(url=CHATGPT_DEVICE_CODE_URL, data=None, json={"client_id": CHATGPT_CLIENT_ID})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        pytest.param(DEVICE_CODE_POLL_SLEEP_SECONDS + 7, DEVICE_CODE_POLL_SLEEP_SECONDS + 7, id="above_the_floor"),
        pytest.param(DEVICE_CODE_POLL_SLEEP_SECONDS, DEVICE_CODE_POLL_SLEEP_SECONDS, id="at_the_floor"),
        pytest.param(1, DEVICE_CODE_POLL_SLEEP_SECONDS, id="below_the_floor"),
        pytest.param(0, DEVICE_CODE_POLL_SLEEP_SECONDS, id="zero"),
        pytest.param("not-a-number", DEVICE_CODE_POLL_SLEEP_SECONDS, id="not_a_number"),
        pytest.param(None, DEVICE_CODE_POLL_SLEEP_SECONDS, id="missing"),
    ],
)
async def test_astart_chatgpt_device_login_never_polls_faster_than_the_floor(interval: object, expected: int) -> None:
    client: Final = _FakeOAuthClient(
        body={"device_auth_id": "device-auth-example", "user_code": "ABCD-1234", "interval": interval}
    )

    login: Final = await astart_chatgpt_device_login(client)

    assert login.interval == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"user_code": "ABCD-1234", "note": SECRET_IN_RESPONSE}, id="no_device_auth_id"),
        pytest.param({"device_auth_id": "device-auth-example", "note": SECRET_IN_RESPONSE}, id="no_user_code"),
        pytest.param({"device_auth_id": "", "user_code": "ABCD-1234", "note": SECRET_IN_RESPONSE}, id="empty_id"),
    ],
)
async def test_astart_chatgpt_device_login_rejects_an_incomplete_response(body: dict[str, str]) -> None:
    with pytest.raises(GetDeviceCodeError) as raised:
        await astart_chatgpt_device_login(_FakeOAuthClient(body=body))

    assert raised.value.status_code == 502
    assert SECRET_IN_RESPONSE not in "".join(traceback.format_exception(raised.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [403, 404])
async def test_apoll_chatgpt_device_login_is_pending_while_the_user_has_not_approved(status_code: int) -> None:
    client: Final = _FakeOAuthClient(status_code=status_code, body={"error": "authorization_pending"})

    assert await apoll_chatgpt_device_login(client, "device-auth-example", "ABCD-1234") is None
    assert client.posts == [
        _Post(
            url=CHATGPT_DEVICE_TOKEN_URL,
            data=None,
            json={"device_auth_id": "device-auth-example", "user_code": "ABCD-1234"},
        )
    ]


@pytest.mark.asyncio
async def test_apoll_chatgpt_device_login_returns_the_code_once_the_user_approves() -> None:
    client: Final = _FakeOAuthClient(
        body={"authorization_code": "code-example", "code_challenge": "challenge", "code_verifier": "verifier-example"}
    )

    code: Final = await apoll_chatgpt_device_login(client, "device-auth-example", "ABCD-1234")

    assert code is not None
    assert (code.authorization_code, code.code_verifier) == ("code-example", "verifier-example")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"authorization_code": "code-example"}, id="no_verifier"),
        pytest.param({"code_verifier": "verifier-example"}, id="no_code"),
        pytest.param(b"not json", id="not_json"),
        pytest.param(["code-example"], id="not_an_object"),
    ],
)
async def test_apoll_chatgpt_device_login_treats_an_unusable_success_as_still_pending(body: object) -> None:
    assert await apoll_chatgpt_device_login(_FakeOAuthClient(body=body), "device-auth-example", "ABCD-1234") is None


@pytest.mark.asyncio
async def test_aexchange_chatgpt_authorization_code_redeems_the_code_with_its_verifier() -> None:
    client: Final = _FakeOAuthClient(
        body={"access_token": "access-new", "refresh_token": "refresh-new", "id_token": "id-new"}
    )

    tokens: Final = await aexchange_chatgpt_authorization_code(
        client, ChatGPTAuthorizationCode(authorization_code="code-example", code_verifier="verifier-example")
    )

    assert (tokens.access_token, tokens.refresh_token, tokens.id_token) == ("access-new", "refresh-new", "id-new")
    assert client.posts == [
        _Post(
            url=CHATGPT_OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": "code-example",
                "redirect_uri": f"{CHATGPT_AUTH_BASE}/deviceauth/callback",
                "client_id": CHATGPT_CLIENT_ID,
                "code_verifier": "verifier-example",
            },
            json=None,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["access_token", "refresh_token", "id_token"])
async def test_aexchange_chatgpt_authorization_code_rejects_a_response_missing_a_token(missing: str) -> None:
    complete: Final = {"access_token": "a", "refresh_token": "r", "id_token": "i", "note": SECRET_IN_RESPONSE}
    body: Final = {key: value for key, value in complete.items() if key != missing}
    with pytest.raises(GetAccessTokenError) as raised:
        await aexchange_chatgpt_authorization_code(
            _FakeOAuthClient(body=body), ChatGPTAuthorizationCode(authorization_code="c", code_verifier="v")
        )

    assert raised.value.status_code == 502
    assert SECRET_IN_RESPONSE not in "".join(traceback.format_exception(raised.value))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected_refresh_token"),
    [
        pytest.param({"access_token": "access-new", "id_token": "id-new"}, "refresh-old", id="omitted"),
        pytest.param(
            {"access_token": "access-new", "id_token": "id-new", "refresh_token": ""}, "refresh-old", id="empty"
        ),
        pytest.param(
            {"access_token": "access-new", "id_token": "id-new", "refresh_token": "refresh-rotated"},
            "refresh-rotated",
            id="rotated",
        ),
    ],
)
async def test_arefresh_chatgpt_tokens_keeps_the_old_refresh_token_unless_a_new_one_arrives(
    body: dict[str, str], expected_refresh_token: str
) -> None:
    client: Final = _FakeOAuthClient(body=body)

    tokens: Final = await arefresh_chatgpt_tokens(client, "refresh-old")

    assert tokens.access_token == "access-new"
    assert tokens.refresh_token == expected_refresh_token
    assert tokens.id_token == "id-new"
    assert client.posts == [
        _Post(
            url=CHATGPT_OAUTH_TOKEN_URL,
            data=None,
            json={
                "client_id": CHATGPT_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": "refresh-old",
                "scope": "openid profile email",
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["access_token", "id_token"])
async def test_arefresh_chatgpt_tokens_rejects_a_response_missing_a_token(missing: str) -> None:
    body: Final = {
        key: value
        for key, value in {"access_token": "a", "id_token": "i", "note": SECRET_IN_RESPONSE}.items()
        if key != missing
    }
    with pytest.raises(RefreshAccessTokenError) as raised:
        await arefresh_chatgpt_tokens(_FakeOAuthClient(body=body), "refresh-old")

    assert raised.value.status_code == 502
    assert SECRET_IN_RESPONSE not in "".join(traceback.format_exception(raised.value))


_SENT_SECRET: Final = "secret-the-caller-sent"
_OAUTH_CALLS: Final[tuple[tuple[str, type[ChatGPTAuthError], Callable[[_FakeOAuthClient], Awaitable[object]]], ...]] = (
    ("start", GetDeviceCodeError, astart_chatgpt_device_login),
    ("poll", GetAccessTokenError, lambda client: apoll_chatgpt_device_login(client, _SENT_SECRET, _SENT_SECRET)),
    (
        "exchange",
        GetAccessTokenError,
        lambda client: aexchange_chatgpt_authorization_code(
            client, ChatGPTAuthorizationCode(authorization_code=_SENT_SECRET, code_verifier=_SENT_SECRET)
        ),
    ),
    ("refresh", RefreshAccessTokenError, lambda client: arefresh_chatgpt_tokens(client, _SENT_SECRET)),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client", "expected_status"),
    [
        pytest.param(_FakeOAuthClient(status_code=400, body={"error": SECRET_IN_RESPONSE}), 400, id="http_400"),
        pytest.param(_FakeOAuthClient(status_code=401, body={"error": SECRET_IN_RESPONSE}), 401, id="http_401"),
        pytest.param(_FakeOAuthClient(status_code=500, body=SECRET_IN_RESPONSE.encode()), 500, id="http_500"),
        pytest.param(
            _FakeOAuthClient(error=httpx.ConnectError(f"{SECRET_IN_RESPONSE} {_SENT_SECRET}")), 502, id="transport"
        ),
        pytest.param(
            _FakeOAuthClient(error=litellm.Timeout(f"{SECRET_IN_RESPONSE} {_SENT_SECRET}", "chatgpt", "chatgpt")),
            502,
            id="timeout",
        ),
    ],
)
@pytest.mark.parametrize(("call", "error_type", "invoke"), _OAUTH_CALLS, ids=[name for name, _, _ in _OAUTH_CALLS])
async def test_chatgpt_oauth_failures_never_echo_the_response_body_or_a_token(
    client: _FakeOAuthClient,
    expected_status: int,
    call: str,
    error_type: type[ChatGPTAuthError],
    invoke: Callable[[_FakeOAuthClient], Awaitable[object]],
) -> None:
    with pytest.raises(error_type) as raised:
        await invoke(client)

    rendered: Final = "".join(traceback.format_exception(raised.value))
    assert raised.value.status_code == expected_status
    assert SECRET_IN_RESPONSE not in rendered
    assert _SENT_SECRET not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "error_type", "invoke"),
    [entry for entry in _OAUTH_CALLS if entry[0] != "poll"],
    ids=[name for name, _, _ in _OAUTH_CALLS if name != "poll"],
)
async def test_chatgpt_oauth_success_that_is_not_a_json_object_never_echoes_the_body(
    call: str, error_type: type[ChatGPTAuthError], invoke: Callable[[_FakeOAuthClient], Awaitable[object]]
) -> None:
    with pytest.raises(error_type) as raised:
        await invoke(_FakeOAuthClient(body=[SECRET_IN_RESPONSE]))

    assert raised.value.status_code == 502
    assert SECRET_IN_RESPONSE not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize(
    ("url", "body", "start"),
    [
        pytest.param(
            CHATGPT_OAUTH_TOKEN_URL,
            {"access_token": SECRET_IN_RESPONSE, "refresh_token": SECRET_IN_RESPONSE},
            lambda authenticator: authenticator._refresh_tokens("refresh-old"),
            id="refresh",
        ),
        pytest.param(
            CHATGPT_OAUTH_TOKEN_URL,
            {"access_token": SECRET_IN_RESPONSE, "id_token": SECRET_IN_RESPONSE},
            lambda authenticator: authenticator._exchange_code_for_tokens(
                {"authorization_code": "code-example", "code_verifier": "verifier-example"}
            ),
            id="exchange",
        ),
        pytest.param(
            CHATGPT_DEVICE_CODE_URL,
            {"user_code": "ABCD-1234", "note": SECRET_IN_RESPONSE},
            lambda authenticator: authenticator._request_device_code(),
            id="device_code",
        ),
    ],
)
def test_sync_authenticator_missing_field_errors_never_echo_the_response_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    body: dict[str, str],
    start: Callable[[Authenticator], object],
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    with respx.mock as transport:
        transport.post(url).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(ChatGPTAuthError) as raised:
            start(Authenticator())

    assert SECRET_IN_RESPONSE not in "".join(traceback.format_exception(raised.value))


def test_chatgpt_login_values_derive_the_account_id_and_expiry_from_the_tokens() -> None:
    access_token: Final = _make_jwt({"exp": 4102444800})
    id_token: Final = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-team-a"}})

    values: Final = chatgpt_login_values(
        ChatGPTTokens(access_token=access_token, refresh_token="refresh-example", id_token=id_token)
    )

    assert dict(values) == {
        "chatgpt_access_token": access_token,
        "chatgpt_refresh_token": "refresh-example",
        "chatgpt_id_token": id_token,
        "chatgpt_account_id": "acct-team-a",
        "chatgpt_expires_at": "4102444800",
    }


def test_chatgpt_login_values_omit_what_the_tokens_do_not_carry() -> None:
    values: Final = chatgpt_login_values(
        ChatGPTTokens(access_token=_make_jwt({}), refresh_token="refresh-example", id_token=_make_jwt({}))
    )

    assert set(values) == {"chatgpt_access_token", "chatgpt_refresh_token", "chatgpt_id_token"}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            ChatGPTTokens(access_token=_SENT_SECRET, refresh_token=_SENT_SECRET, id_token=_SENT_SECRET), id="tokens"
        ),
        pytest.param(ChatGPTAuthorizationCode(authorization_code=_SENT_SECRET, code_verifier=_SENT_SECRET), id="code"),
    ],
)
def test_chatgpt_oauth_values_never_print_their_secrets(value: object) -> None:
    assert _SENT_SECRET not in repr(value)
    assert _SENT_SECRET not in str(value)
