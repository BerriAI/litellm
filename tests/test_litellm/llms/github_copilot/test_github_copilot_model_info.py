import json
from datetime import datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import GetAPIKeyError
from litellm.llms.github_copilot.model_info import GithubCopilotModelInfo
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

API_BASE = "https://api.githubcopilot.com"
API_KEY = "gh.test-key-123456789"

MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "gpt-4o", "capabilities": {"type": "chat"}, "model_picker_enabled": True},
        {"id": "claude-opus-5.5", "capabilities": {"type": "chat"}, "model_picker_enabled": True},
        {"id": "text-embedding-3-small", "capabilities": {"type": "embeddings"}, "model_picker_enabled": False},
    ],
}

EXPECTED_MODELS = [
    "github_copilot/gpt-4o",
    "github_copilot/claude-opus-5.5",
    "github_copilot/text-embedding-3-small",
]


class StubAuthenticator(Authenticator):
    """Token files point at a tmp path, and device login raises instead of blocking on a user code."""

    def __init__(self, api_key_file: str, access_token_file: str) -> None:
        self.api_key_file = api_key_file
        self.access_token_file = access_token_file

    def _login(self) -> str:
        raise AssertionError("device-code login must never run during model listing")


def _write_api_key(path, expires_in: timedelta) -> str:
    path.write_text(json.dumps({"token": API_KEY, "expires_at": (datetime.now() + expires_in).timestamp()}))
    return str(path)


@pytest.fixture
def valid_credential(tmp_path):
    return StubAuthenticator(
        api_key_file=_write_api_key(tmp_path / "api-key.json", timedelta(hours=1)),
        access_token_file=str(tmp_path / "absent-access-token"),
    )


@pytest.fixture
def no_credential(tmp_path):
    return StubAuthenticator(
        api_key_file=str(tmp_path / "absent-api-key.json"),
        access_token_file=str(tmp_path / "absent-access-token"),
    )


@pytest.fixture
def expired_credential(tmp_path):
    return StubAuthenticator(
        api_key_file=_write_api_key(tmp_path / "api-key.json", timedelta(hours=-1)),
        access_token_file=str(tmp_path / "absent-access-token"),
    )


def _copilot(handler) -> HTTPHandler:
    """A real HTTPHandler whose only fake part is the transport, so the request is built for real."""
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _serving(payload, status_code: int = 200, base: str = API_BASE, requires_key: str | None = API_KEY):
    """Serves the catalog at `base` only, and only to a caller presenting `requires_key`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) != f"{base}/models":
            return httpx.Response(404, json={"error": f"no catalog at {request.url}"})
        if requires_key is not None and request.headers.get("authorization") != f"Bearer {requires_key}":
            return httpx.Response(403, json={"error": "wrong account"})
        if request.headers.get("copilot-integration-id") != "vscode-chat":
            return httpx.Response(403, json={"error": "catalog is scoped to the integration id"})
        return httpx.Response(status_code, json=payload)

    return handler


def test_get_models_returns_every_prefixed_id(valid_credential):
    model_info = GithubCopilotModelInfo(valid_credential, _copilot(_serving(MODELS_PAYLOAD)))

    assert model_info.get_models(api_base=API_BASE) == EXPECTED_MODELS


def test_get_models_sends_the_copilot_integration_headers(valid_credential):
    # Copilot scopes the catalog to the integration id, so listing has to send the same headers the
    # completion path sends. This transport answers with whatever credentials it was handed.
    def echo_headers(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": request.headers.get("authorization", "no-authorization")},
                    {"id": request.headers.get("copilot-integration-id", "no-integration-id")},
                ]
            },
        )

    model_info = GithubCopilotModelInfo(valid_credential, _copilot(echo_headers))

    assert model_info.get_models(api_base=API_BASE) == [
        f"github_copilot/Bearer {API_KEY}",
        "github_copilot/vscode-chat",
    ]


def test_get_models_bounds_the_request_so_a_slow_copilot_cannot_stall_the_caller(valid_credential):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions.get("timeout") or {})
        return httpx.Response(200, json=MODELS_PAYLOAD)

    model_info = GithubCopilotModelInfo(valid_credential, _copilot(handler))
    model_info.get_models(api_base=API_BASE)

    assert seen["read"] == pytest.approx(10.0)
    assert seen["connect"] == pytest.approx(10.0)


def test_get_models_honours_an_explicit_api_base(valid_credential):
    # The transport serves a catalog at this base only and 404s anything else, so a listing that
    # ignored the argument would raise instead of returning.
    model_info = GithubCopilotModelInfo(
        valid_credential, _copilot(_serving(MODELS_PAYLOAD, base="https://copilot.example.com"))
    )

    assert model_info.get_models(api_base="https://copilot.example.com") == EXPECTED_MODELS


def test_get_models_ignores_an_explicit_api_key_so_listing_matches_completions(valid_credential):
    # Completions always authenticate with the cached Copilot credential, so listing must too.
    # This transport only serves the caller presenting that credential.
    model_info = GithubCopilotModelInfo(valid_credential, _copilot(_serving(MODELS_PAYLOAD)))

    assert model_info.get_models(api_key="a-different-accounts-key", api_base=API_BASE) == EXPECTED_MODELS


def test_get_models_without_any_credential_never_starts_device_login(no_credential):
    model_info = GithubCopilotModelInfo(no_credential, _copilot(_serving(MODELS_PAYLOAD)))

    with pytest.raises(ValueError, match="not authenticated"):
        model_info.get_models(api_base=API_BASE)


def test_get_models_with_an_expired_key_and_no_access_token_never_starts_device_login(expired_credential):
    model_info = GithubCopilotModelInfo(expired_credential, _copilot(_serving(MODELS_PAYLOAD)))

    with pytest.raises(ValueError, match="not authenticated"):
        model_info.get_models(api_base=API_BASE)


def test_get_models_raises_on_http_error(valid_credential):
    model_info = GithubCopilotModelInfo(valid_credential, _copilot(_serving({"error": "forbidden"}, status_code=403)))

    with pytest.raises(Exception, match="Failed to fetch models from GitHub Copilot"):
        model_info.get_models(api_base=API_BASE)


def test_get_models_raises_on_unexpected_payload(valid_credential):
    model_info = GithubCopilotModelInfo(valid_credential, _copilot(_serving({"models": []})))

    with pytest.raises(ValidationError, match="data"):
        model_info.get_models(api_base=API_BASE)


def test_get_base_model_strips_prefix():
    assert GithubCopilotModelInfo.get_base_model("github_copilot/gpt-4o") == "gpt-4o"
    assert GithubCopilotModelInfo.get_base_model("gpt-4o") == "gpt-4o"


def test_provider_config_manager_returns_model_info():
    model_info = ProviderConfigManager.get_provider_model_info(model=None, provider=LlmProviders.GITHUB_COPILOT)
    assert isinstance(model_info, GithubCopilotModelInfo)


class RefreshingAuthenticator(StubAuthenticator):
    """Has only an access token on disk, and can exchange it for an api key without logging in."""

    def _refresh_api_key(self) -> dict:
        return {"token": API_KEY, "expires_at": (datetime.now() + timedelta(hours=1)).timestamp()}


class UnauthorizedAuthenticator(StubAuthenticator):
    def get_api_key(self) -> str:
        raise GetAPIKeyError(status_code=401, message="the cached credential was rejected")


def test_get_models_with_only_an_access_token_exchanges_it_without_logging_in(tmp_path):
    access_token_file = tmp_path / "access-token"
    access_token_file.write_text("gho.test-access-token\n")
    authenticator = RefreshingAuthenticator(
        api_key_file=str(tmp_path / "absent-api-key.json"),
        access_token_file=str(access_token_file),
    )

    model_info = GithubCopilotModelInfo(authenticator, _copilot(_serving(MODELS_PAYLOAD)))

    assert model_info.get_models(api_base=API_BASE) == EXPECTED_MODELS


def test_get_models_reports_a_rejected_credential_instead_of_listing(valid_credential, tmp_path):
    authenticator = UnauthorizedAuthenticator(
        api_key_file=valid_credential.api_key_file,
        access_token_file=str(tmp_path / "absent-access-token"),
    )

    model_info = GithubCopilotModelInfo(authenticator, _copilot(_serving(MODELS_PAYLOAD)))

    with pytest.raises(ValueError, match="not authenticated"):
        model_info.get_models(api_base=API_BASE)


def test_validate_environment_adds_the_cached_credential(valid_credential):
    headers = GithubCopilotModelInfo(valid_credential).validate_environment(
        headers={}, model="github_copilot/gpt-4o", messages=[], optional_params={}, litellm_params={}
    )

    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["copilot-integration-id"] == "vscode-chat"


def test_validate_environment_leaves_caller_headers_untouched(valid_credential):
    headers = GithubCopilotModelInfo(valid_credential).validate_environment(
        headers={"Authorization": "Bearer caller-supplied"},
        model="github_copilot/gpt-4o",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer caller-supplied"


def test_validate_environment_without_a_credential_returns_the_caller_headers(no_credential):
    headers = GithubCopilotModelInfo(no_credential).validate_environment(
        headers={"x-caller": "kept"}, model="github_copilot/gpt-4o", messages=[], optional_params={}, litellm_params={}
    )

    assert headers == {"x-caller": "kept"}


def test_get_api_key_returns_an_explicit_key_unchanged():
    assert GithubCopilotModelInfo.get_api_key("explicit-key") == "explicit-key"
