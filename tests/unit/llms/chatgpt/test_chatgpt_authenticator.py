import base64
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, mock_open, patch

import httpx
import pytest
import respx

import litellm
from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS, HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS
from litellm.llms.chatgpt.authenticator import (
    TOKEN_REFRESH_TIMEOUT_SECONDS,
    Authenticator,
    get_cached_authenticator,
    get_chatgpt_auth_file,
)
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
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


class TestChatGPTRefreshSerialization:
    """Refresh tokens are single use, so concurrent refreshes of one file must not both spend it."""

    @pytest.fixture
    def expired_auth_file(self, tmp_path: Path) -> Path:
        auth_file: Final = tmp_path / "auth.json"
        auth_file.write_text(
            json.dumps({"access_token": "expired-token", "refresh_token": "refresh-original", "expires_at": 1})
        )
        return auth_file

    @staticmethod
    def _single_use_token_endpoint(transport: respx.MockRouter, spent: list[str]) -> None:
        def refresh(request: httpx.Request) -> httpx.Response:
            refresh_token: Final = json.loads(request.content)["refresh_token"]
            time.sleep(0.2)  # widen the window in which a second refresh could start
            if refresh_token in spent:
                return httpx.Response(401, json={"error": "invalid_grant"})
            spent.append(refresh_token)
            token: Final = _make_jwt({"exp": 4102444800})
            return httpx.Response(
                200,
                json={"access_token": token, "id_token": token, "refresh_token": f"refresh-{len(spent)}"},
            )

        transport.post("https://auth.openai.com/oauth/token").mock(side_effect=refresh)

    def test_concurrent_callers_refresh_once(self, expired_auth_file: Path) -> None:
        spent: Final[list[str]] = []
        with respx.mock as transport:
            self._single_use_token_endpoint(transport, spent)
            # Separate instances, as separate cached authenticators or workers would have.
            callers: Final = [Authenticator(auth_file=str(expired_auth_file)) for _ in range(8)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                tokens: Final = list(pool.map(lambda caller: caller.get_access_token(), callers))
        assert spent == ["refresh-original"]
        assert len(set(tokens)) == 1
        assert json.loads(expired_auth_file.read_text())["refresh_token"] == "refresh-1"

    def test_waiter_uses_token_refreshed_by_another_process(self, expired_auth_file: Path) -> None:
        if sys.platform == "win32":
            pytest.skip("file locks across processes are POSIX only")
        fresh_token: Final = _make_jwt({"exp": 4102444800})
        # A separate open file description conflicts like another process holding the lock.
        holder: Final = os.open(str(expired_auth_file) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(holder, fcntl.LOCK_EX)
        spent: Final[list[str]] = []
        try:
            with respx.mock as transport, ThreadPoolExecutor(max_workers=1) as pool:
                self._single_use_token_endpoint(transport, spent)
                waiting = pool.submit(Authenticator(auth_file=str(expired_auth_file)).get_access_token)
                time.sleep(0.2)
                assert not waiting.done()
                expired_auth_file.write_text(
                    json.dumps(
                        {"access_token": fresh_token, "refresh_token": "refresh-other", "expires_at": 4102444800}
                    )
                )
                fcntl.flock(holder, fcntl.LOCK_UN)
                assert waiting.result(timeout=5) == fresh_token
        finally:
            os.close(holder)
        assert spent == []

    def test_lock_wait_is_bounded(self, expired_auth_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("litellm.llms.chatgpt.authenticator.REFRESH_LOCK_TIMEOUT_SECONDS", 0.2)
        holder: Final = os.open(str(expired_auth_file) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(holder, fcntl.LOCK_EX)
        try:
            with pytest.raises(GetAccessTokenError) as error:
                Authenticator(auth_file=str(expired_auth_file)).get_access_token()
        finally:
            os.close(holder)
        assert error.value.status_code == 503

    def test_valid_token_does_not_take_the_lock(self, tmp_path: Path) -> None:
        auth_file: Final = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"access_token": "valid-token", "expires_at": 4102444800}))
        assert Authenticator(auth_file=str(auth_file)).get_access_token() == "valid-token"
        assert not (tmp_path / "auth.json.lock").exists()


class TestChatGPTAuthFileWrites:
    def test_write_is_private_and_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        auth_file: Final = tmp_path / "auth.json"
        Authenticator(auth_file=str(auth_file))._write_auth_file({"access_token": "token"})
        assert json.loads(auth_file.read_text()) == {"access_token": "token"}
        assert auth_file.stat().st_mode & 0o777 == 0o600
        assert [path.name for path in tmp_path.iterdir()] == ["auth.json"]

    def test_failed_write_keeps_the_previous_file(self, tmp_path: Path) -> None:
        auth_file: Final = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"access_token": "previous"}))
        with patch("litellm.llms.chatgpt.authenticator.json.dump", side_effect=OSError("disk full")):
            Authenticator(auth_file=str(auth_file))._write_auth_file({"access_token": "next"})
        assert json.loads(auth_file.read_text()) == {"access_token": "previous"}
        assert [path.name for path in tmp_path.iterdir()] == ["auth.json"]


def test_account_id_write_back_never_restores_spent_tokens(tmp_path: Path) -> None:
    auth_file: Final = tmp_path / "auth.json"
    stale_token: Final = _make_jwt({"exp": 4102444800, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})
    auth_file.write_text(
        json.dumps({"access_token": stale_token, "refresh_token": "refresh-spent", "expires_at": 4102444800})
    )
    authenticator: Final = Authenticator(auth_file=str(auth_file))
    original_read: Final = authenticator._read_auth_file
    reads: list[int] = []

    def read_then_rotate():
        record = original_read()
        if not reads:
            # A concurrent refresh rotates the tokens right after this caller's first read.
            auth_file.write_text(
                json.dumps({"access_token": "rotated", "refresh_token": "refresh-new", "expires_at": 4102444800})
            )
        reads.append(1)
        return record

    with patch.object(authenticator, "_read_auth_file", side_effect=read_then_rotate):
        assert authenticator.get_account_id() == "acct-1"
    assert json.loads(auth_file.read_text())["refresh_token"] == "refresh-new"
