import base64
import hashlib
import threading
import urllib.parse
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_CLIENT_ID,
    CHATGPT_OAUTH_TOKEN_URL,
    GetAccessTokenError,
)
from litellm.llms.chatgpt.pkce import (
    OAUTH_AUTHORIZE_URL,
    REDIRECT_PATH,
    SCOPE,
    _build_authorize_url,
    _exchange_code_for_tokens,
    _generate_code_challenge,
    _generate_code_verifier,
    _make_handler,
)


class TestPkceHelpers:
    def test_verifier_length_and_charset(self):
        verifier = _generate_code_verifier()
        assert 43 <= len(verifier) <= 128
        # token_urlsafe uses RFC 7636 unreserved chars (A-Z, a-z, 0-9, -, _)
        assert all(c.isalnum() or c in "-_" for c in verifier)

    def test_challenge_is_s256_of_verifier(self):
        verifier = "fixed-verifier-for-testing-abc123-abc123-abc"
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert _generate_code_challenge(verifier) == expected

    def test_authorize_url_contains_required_params(self):
        url = _build_authorize_url(
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            code_challenge="test-challenge",
            state="test-state",
        )
        assert url.startswith(OAUTH_AUTHORIZE_URL + "?")
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert parsed["response_type"] == ["code"]
        assert parsed["client_id"] == [CHATGPT_CLIENT_ID]
        assert parsed["redirect_uri"] == ["http://127.0.0.1:1455/auth/callback"]
        assert parsed["scope"] == [SCOPE]
        assert parsed["code_challenge"] == ["test-challenge"]
        assert parsed["code_challenge_method"] == ["S256"]
        assert parsed["state"] == ["test-state"]
        assert parsed["id_token_add_organizations"] == ["true"]
        assert parsed["codex_cli_simplified_flow"] == ["true"]
        assert "originator" in parsed


class _FakeWFile:
    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data


class _FakeHandler:
    """Minimal stand-in that lets us invoke the generated do_GET without sockets."""

    def __init__(self, handler_cls, path):
        self.handler_cls = handler_cls
        self.path = path
        self.wfile = _FakeWFile()
        self.response_status = None
        self.headers_sent = {}

    def send_response(self, code):
        self.response_status = code

    def send_header(self, k, v):
        self.headers_sent[k] = v

    def end_headers(self):
        pass


def _invoke_handler(handler_cls, path):
    fake = _FakeHandler(handler_cls, path)
    # Bind both methods to the fake; they only read self.path / self.wfile / etc.
    fake._respond_html = handler_cls._respond_html.__get__(fake, type(fake))
    handler_cls.do_GET(fake)
    return fake


class TestPkceCallbackHandler:
    def test_success_captures_code_and_sets_event(self):
        result = {}
        event = threading.Event()
        handler_cls = _make_handler(result, event, expected_state="abc")
        fake = _invoke_handler(handler_cls, f"{REDIRECT_PATH}?code=xyz&state=abc")
        assert fake.response_status == 200
        assert result == {"code": "xyz"}
        assert event.is_set()

    def test_rejects_state_mismatch(self):
        result = {}
        event = threading.Event()
        handler_cls = _make_handler(result, event, expected_state="abc")
        fake = _invoke_handler(handler_cls, f"{REDIRECT_PATH}?code=xyz&state=wrong")
        assert fake.response_status == 400
        assert result.get("error") == "state mismatch"
        assert "code" not in result
        assert event.is_set()

    def test_propagates_provider_error(self):
        result = {}
        event = threading.Event()
        handler_cls = _make_handler(result, event, expected_state="abc")
        fake = _invoke_handler(
            handler_cls,
            f"{REDIRECT_PATH}?error=access_denied&error_description=User+cancelled&state=abc",
        )
        assert fake.response_status == 400
        assert result["error"] == "User cancelled"
        assert event.is_set()

    def test_ignores_unrelated_paths(self):
        result = {}
        event = threading.Event()
        handler_cls = _make_handler(result, event, expected_state="abc")
        fake = _invoke_handler(handler_cls, "/favicon.ico")
        assert fake.response_status == 404
        assert result == {}
        assert not event.is_set()


class TestPkceTokenExchange:
    def test_posts_form_data_and_returns_tokens(self):
        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
        }
        fake_response.raise_for_status.return_value = None
        fake_client.post.return_value = fake_response

        with patch(
            "litellm.llms.chatgpt.pkce._get_httpx_client",
            return_value=fake_client,
        ):
            tokens = _exchange_code_for_tokens(
                code="the-code",
                code_verifier="the-verifier",
                redirect_uri="http://127.0.0.1:1455/auth/callback",
            )

        assert tokens == {"access_token": "a", "refresh_token": "r", "id_token": "i"}
        args, kwargs = fake_client.post.call_args
        assert args[0] == CHATGPT_OAUTH_TOKEN_URL
        assert kwargs["data"] == {
            "grant_type": "authorization_code",
            "code": "the-code",
            "redirect_uri": "http://127.0.0.1:1455/auth/callback",
            "client_id": CHATGPT_CLIENT_ID,
            "code_verifier": "the-verifier",
        }

    def test_raises_on_missing_fields(self):
        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.json.return_value = {"access_token": "a"}  # missing fields
        fake_response.raise_for_status.return_value = None
        fake_client.post.return_value = fake_response

        with patch(
            "litellm.llms.chatgpt.pkce._get_httpx_client",
            return_value=fake_client,
        ):
            with pytest.raises(GetAccessTokenError):
                _exchange_code_for_tokens(code="c", code_verifier="v", redirect_uri="r")

    def test_raises_on_http_error(self):
        fake_client = MagicMock()
        request = httpx.Request("POST", CHATGPT_OAUTH_TOKEN_URL)
        response = httpx.Response(400, request=request)
        fake_client.post.side_effect = httpx.HTTPStatusError(
            "bad", request=request, response=response
        )

        with patch(
            "litellm.llms.chatgpt.pkce._get_httpx_client",
            return_value=fake_client,
        ):
            with pytest.raises(GetAccessTokenError) as excinfo:
                _exchange_code_for_tokens(code="c", code_verifier="v", redirect_uri="r")
            assert excinfo.value.status_code == 400


class TestLoginPkcePortInUse:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_raises_when_port_cannot_bind(self, authenticator):
        with patch(
            "litellm.llms.chatgpt.pkce.http.server.HTTPServer",
            side_effect=OSError("address in use"),
        ):
            with pytest.raises(GetAccessTokenError) as excinfo:
                authenticator.login_pkce(
                    open_browser=False, port=1455, timeout_seconds=1
                )
            assert "Failed to bind loopback server" in excinfo.value.message
