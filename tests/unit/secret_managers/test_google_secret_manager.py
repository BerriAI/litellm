import base64
from dataclasses import dataclass
from typing import Final

import litellm
import pytest
import respx

from litellm.secret_managers.google_secret_manager import GoogleSecretManager


@dataclass(frozen=True, slots=True)
class _CachedVertexCredentials:
    token: str
    quota_project_id: str | None
    expired: bool = False

    def refresh(self, request: object) -> None:
        return None


def _google_secret_manager(
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
) -> GoogleSecretManager:
    monkeypatch.setenv("GOOGLE_SECRET_MANAGER_PROJECT_ID", project_id)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    secret_manager: Final = GoogleSecretManager()
    credentials: Final = _CachedVertexCredentials(
        token="test-gsm-token",
        quota_project_id=project_id,
    )
    vertex_chat_completion: Final = litellm.vertex_chat_completion
    monkeypatch.setitem(
        vertex_chat_completion._credentials_project_mapping,
        (None, project_id),
        (credentials, project_id),
    )
    return secret_manager


def test_google_secret_manager_decodes_secret_and_requests_latest_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id: Final = "test-secret-project"
    secret_manager: Final = _google_secret_manager(monkeypatch, project_id)
    secret_url: Final = (
        f"https://secretmanager.googleapis.com/v1/projects/{project_id}/secrets/OPENAI_API_KEY/versions/latest:access"
    )
    encoded_secret: Final = base64.b64encode(b"anything").decode("ascii")
    upstream: Final[respx.MockRouter]

    with respx.mock(assert_all_called=True) as upstream:
        secret_route: Final = upstream.get(secret_url).respond(
            200,
            json={"payload": {"data": encoded_secret}},
        )

        result: Final = secret_manager.get_secret_from_google_secret_manager("OPENAI_API_KEY")

        assert result == "anything"
        assert secret_route.called
        assert len(upstream.calls) == 1
        assert str(upstream.calls.last.request.url) == secret_url
        assert upstream.calls.last.request.headers["Authorization"] == "Bearer test-gsm-token"


def test_google_secret_manager_returns_cached_values_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id: Final = "test-secret-project"
    secret_manager: Final = _google_secret_manager(monkeypatch, project_id)
    secret_manager.cache.set_cache("cached-none", None)
    secret_manager.cache.set_cache("cached-string", "lite-llm")
    upstream: Final[respx.MockRouter]

    with respx.mock() as upstream:
        missing_value: Final = secret_manager.get_secret_from_google_secret_manager("cached-none")
        cached_value: Final = secret_manager.get_secret_from_google_secret_manager("cached-string")

        assert missing_value is None
        assert cached_value == "lite-llm"
        assert upstream.calls.call_count == 0
