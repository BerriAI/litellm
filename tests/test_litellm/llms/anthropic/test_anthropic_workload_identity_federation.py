"""Tests for Anthropic Workload Identity Federation support."""

import os
import sys
from typing import Optional
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.llms.anthropic.workload_identity_federation import (
    ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_GRANT_TYPE,
    AnthropicWorkloadIdentityFederationCredentials,
    AnthropicWorkloadIdentityFederationError,
    AnthropicWorkloadIdentityFederationTokenProvider,
    get_workload_identity_federation_credentials_from_env,
    exchange_anthropic_workload_identity_federation_token,
    reset_workload_identity_federation_provider_cache,
)


FAKE_REQUIRED_CREDS_KWARGS = dict(
    federation_rule_id="fdrl_test_rule",
    organization_id="org_test",
    service_account_id="svac_test",
)


def _make_creds(
    *,
    identity_token_file: Optional[str] = None,
    identity_token: Optional[str] = None,
    workspace_id: Optional[str] = "wrkspc_test",
) -> AnthropicWorkloadIdentityFederationCredentials:
    return AnthropicWorkloadIdentityFederationCredentials(
        identity_token_file=identity_token_file,
        identity_token=identity_token,
        workspace_id=workspace_id,
        **FAKE_REQUIRED_CREDS_KWARGS,
    )


def _build_response(*, access_token: str = "anth_at_test", expires_in: Optional[int] = 3600) -> httpx.Response:
    payload: dict = {"access_token": access_token}
    if expires_in is not None:
        payload["expires_in"] = expires_in
    return httpx.Response(status_code=200, json=payload, request=httpx.Request("POST", "https://x"))


class _StubClient:
    def __init__(
        self,
        response: Optional[httpx.Response] = None,
        exc: Optional[Exception] = None,
    ):
        self.calls: list = []
        self._response = response or _build_response()
        self._exc = exc

    def post(self, url: str, data: dict, headers: dict) -> httpx.Response:
        self.calls.append({"url": url, "data": data, "headers": headers})
        if self._exc is not None:
            raise self._exc
        return self._response


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    reset_workload_identity_federation_provider_cache()
    yield
    reset_workload_identity_federation_provider_cache()


def _clear_anthropic_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
        "ANTHROPIC_WORKSPACE_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(var, raising=False)


class TestAnthropicWorkloadIdentityFederationCredentials:
    def test_missing_required_field_raises(self):
        with pytest.raises(AnthropicWorkloadIdentityFederationError) as exc:
            AnthropicWorkloadIdentityFederationCredentials(
                federation_rule_id="fdrl_x",
                organization_id="org_x",
                service_account_id="",
                identity_token_file="/tmp/jwt",
            )
        assert "service_account_id" in str(exc.value)

    def test_missing_both_identity_sources_raises(self):
        with pytest.raises(AnthropicWorkloadIdentityFederationError) as exc:
            AnthropicWorkloadIdentityFederationCredentials(
                federation_rule_id="fdrl_x",
                organization_id="org_x",
                service_account_id="svac_x",
            )
        assert "identity_token" in str(exc.value)

    def test_workspace_id_is_optional(self):
        creds = AnthropicWorkloadIdentityFederationCredentials(
            federation_rule_id="fdrl_x",
            organization_id="org_x",
            service_account_id="svac_x",
            identity_token="raw.jwt",
        )
        assert creds.workspace_id is None


class TestGetWorkloadIdentityFederationCredentialsFromEnv:
    def test_returns_none_when_required_var_missing(self, monkeypatch):
        _clear_anthropic_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_x")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_x")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "raw.jwt")
        assert get_workload_identity_federation_credentials_from_env() is None

    def test_returns_none_when_no_identity_source(self, monkeypatch):
        _clear_anthropic_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_x")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_x")
        monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_x")
        assert get_workload_identity_federation_credentials_from_env() is None

    def test_returns_credentials_with_identity_token_env(self, monkeypatch):
        _clear_anthropic_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_x")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_x")
        monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_x")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline.jwt")
        creds = get_workload_identity_federation_credentials_from_env()
        assert creds is not None
        assert creds.identity_token == "inline.jwt"
        assert creds.identity_token_file is None
        assert creds.workspace_id is None

    def test_returns_credentials_with_identity_token_file_env(self, monkeypatch, tmp_path):
        _clear_anthropic_env(monkeypatch)
        token_file = tmp_path / "jwt"
        token_file.write_text("file.jwt")
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_x")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_x")
        monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_x")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_x")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token_file))
        creds = get_workload_identity_federation_credentials_from_env()
        assert creds is not None
        assert creds.identity_token_file == str(token_file)
        assert creds.workspace_id == "wrkspc_x"


class TestAnthropicWorkloadIdentityFederationTokenProviderExchange:
    def test_exchange_posts_jwt_bearer_form(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion.v1")
        stub = _StubClient()
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            api_base="https://api.anthropic.com",
            http_client=stub,  # type: ignore[arg-type]
        )

        token = provider.get_token()
        assert token == "anth_at_test"
        assert len(stub.calls) == 1
        call = stub.calls[0]
        assert call["url"] == "https://api.anthropic.com/v1/oauth/token"
        assert call["data"]["grant_type"] == ANTHROPIC_WORKLOAD_IDENTITY_FEDERATION_GRANT_TYPE
        assert call["data"]["assertion"] == "assertion.v1"
        assert call["data"]["federation_rule_id"] == "fdrl_test_rule"
        assert call["data"]["organization_id"] == "org_test"
        assert call["data"]["service_account_id"] == "svac_test"
        assert call["data"]["workspace_id"] == "wrkspc_test"
        assert call["headers"]["content-type"] == "application/x-www-form-urlencoded"

    def test_workspace_id_omitted_when_unset(self):
        stub = _StubClient()
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token="raw.jwt", workspace_id=None),
            http_client=stub,  # type: ignore[arg-type]
        )
        provider.get_token()
        assert "workspace_id" not in stub.calls[0]["data"]

    def test_identity_token_used_directly(self):
        stub = _StubClient()
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token="inline.assertion"),
            http_client=stub,  # type: ignore[arg-type]
        )
        provider.get_token()
        assert stub.calls[0]["data"]["assertion"] == "inline.assertion"

    def test_assertion_re_read_on_each_exchange(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion.v1")
        stub = _StubClient(response=_build_response(expires_in=3600))
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=stub,  # type: ignore[arg-type]
        )

        provider.get_token()
        token_file.write_text("assertion.v2")
        provider._access_token = None
        provider._expires_at = 0.0
        provider.get_token()

        assert [c["data"]["assertion"] for c in stub.calls] == [
            "assertion.v1",
            "assertion.v2",
        ]

    def test_assertion_kwarg_overrides_credentials(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("file.assertion")
        stub = _StubClient(response=_build_response(expires_in=3600))
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=stub,  # type: ignore[arg-type]
        )
        provider.get_token(assertion="caller.assertion")
        assert stub.calls[0]["data"]["assertion"] == "caller.assertion"

    def test_assertion_kwarg_forces_refresh(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("file.assertion")
        stub = _StubClient(response=_build_response(expires_in=3600))
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=stub,  # type: ignore[arg-type]
        )
        provider.get_token()
        provider.get_token(assertion="explicit.assertion")
        assert len(stub.calls) == 2
        assert stub.calls[1]["data"]["assertion"] == "explicit.assertion"

    def test_missing_access_token_in_response_raises(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion")
        bad_response = httpx.Response(
            status_code=200,
            json={"expires_in": 3600},
            request=httpx.Request("POST", "https://x"),
        )
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=_StubClient(response=bad_response),  # type: ignore[arg-type]
        )
        with pytest.raises(AnthropicWorkloadIdentityFederationError) as exc:
            provider.get_token()
        assert "access_token" in str(exc.value)

    def test_http_error_response_raises(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion")
        err_response = httpx.Response(
            status_code=401,
            text="unauthorized",
            request=httpx.Request("POST", "https://x"),
        )
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=_StubClient(response=err_response),  # type: ignore[arg-type]
        )
        with pytest.raises(AnthropicWorkloadIdentityFederationError) as exc:
            provider.get_token()
        assert "HTTP 401" in str(exc.value)

    def test_missing_jwt_file_raises(self, tmp_path):
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(tmp_path / "does-not-exist")),
            http_client=_StubClient(),  # type: ignore[arg-type]
        )
        with pytest.raises(AnthropicWorkloadIdentityFederationError) as exc:
            provider.get_token()
        assert "ANTHROPIC_IDENTITY_TOKEN_FILE" in str(exc.value)


class TestAnthropicWorkloadIdentityFederationTokenProviderRefresh:
    def test_serves_cached_token_when_fresh(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion")
        stub = _StubClient(response=_build_response(expires_in=3600))
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=stub,  # type: ignore[arg-type]
        )

        first = provider.get_token()
        second = provider.get_token()
        assert first == second
        assert len(stub.calls) == 1

    def test_advisory_refresh_failure_returns_cached_token(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion")
        stub = _StubClient(response=_build_response(expires_in=3600))
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=stub,  # type: ignore[arg-type]
        )

        provider.get_token()
        original_token = provider._access_token
        import time as _time

        provider._expires_at = _time.time() + 90
        stub._exc = AnthropicWorkloadIdentityFederationError("network blip")

        served = provider.get_token()
        assert served == original_token

    def test_mandatory_refresh_failure_raises(self, tmp_path):
        token_file = tmp_path / "jwt"
        token_file.write_text("assertion")
        stub = _StubClient(response=_build_response(expires_in=3600))
        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=_make_creds(identity_token_file=str(token_file)),
            http_client=stub,  # type: ignore[arg-type]
        )
        provider.get_token()
        import time as _time

        provider._expires_at = _time.time() + 10
        stub._exc = AnthropicWorkloadIdentityFederationError("hard fail")

        with pytest.raises(AnthropicWorkloadIdentityFederationError):
            provider.get_token()


class TestExchangeAnthropicWorkloadIdentityFederationToken:
    def test_returns_none_when_env_vars_missing(self, monkeypatch):
        _clear_anthropic_env(monkeypatch)
        assert exchange_anthropic_workload_identity_federation_token() is None

    def test_explicit_credentials_and_assertion_passthrough(self, monkeypatch):
        _clear_anthropic_env(monkeypatch)
        creds = AnthropicWorkloadIdentityFederationCredentials(
            federation_rule_id="fdrl_x",
            organization_id="org_x",
            service_account_id="svac_x",
            identity_token="placeholder",
        )
        stub = _StubClient()
        from litellm.llms.anthropic import workload_identity_federation as wif_module

        provider = AnthropicWorkloadIdentityFederationTokenProvider(
            credentials=creds,
            http_client=stub,  # type: ignore[arg-type]
        )
        with patch.object(
            wif_module,
            "get_or_create_workload_identity_federation_provider",
            return_value=provider,
        ):
            token = exchange_anthropic_workload_identity_federation_token(
                credentials=creds, assertion="caller.assertion"
            )
        assert token == "anth_at_test"
        assert stub.calls[0]["data"]["assertion"] == "caller.assertion"


class TestValidateEnvironmentWiring:
    def test_validate_environment_uses_workload_identity_federation_when_no_api_key(self, monkeypatch, tmp_path):
        _clear_anthropic_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_x")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_x")
        monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_x")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline.jwt")

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with patch(
            "litellm.llms.anthropic.common_utils.exchange_anthropic_workload_identity_federation_token",
            return_value="anth_at_exchanged",
        ):
            config = AnthropicModelInfo()
            headers = config.validate_environment(
                headers={},
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

        assert headers["authorization"] == "Bearer anth_at_exchanged"
        assert "x-api-key" not in headers

    def test_validate_environment_raises_when_no_creds(self, monkeypatch):
        _clear_anthropic_env(monkeypatch)

        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with pytest.raises(litellm.AuthenticationError) as exc:
            config.validate_environment(
                headers={},
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
        assert "ANTHROPIC_API_KEY" in str(exc.value)
