import concurrent.futures
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import litellm
from litellm.llms.anthropic.wif import (
    AnthropicWifParams,
    _raise_anthropic_wif_error,
    build_anthropic_wif_spec,
    get_anthropic_wif_token,
    resolve_anthropic_wif_params,
)
from litellm.llms.base_llm.auth.identity_source import (
    InternalIssuerSource,
    KeycloakSource,
    identity_source_ref,
)
from litellm.llms.base_llm.auth.jwt_signing import build_jwks, rfc7638_thumbprint
from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine
from litellm.llms.base_llm.auth.types import (
    AssertionSourceError,
    ExchangeError,
    InsecureTokenUrl,
    MalformedTokenResponse,
    TokenEndpointError,
    TokenTransportError,
)

WIF_ENV_VARS: Final = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_SOURCE",
    "ANTHROPIC_SCOPE",
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_BASE_URL",
    "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS",
)

GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"


@pytest.fixture(autouse=True)
def _clean_wif_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in WIF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordedRequest:
    def __init__(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> None:
        self.url = url
        self.content = content
        self.headers = dict(headers)
        self.timeout = timeout

    def json_body(self) -> dict:
        return json.loads(self.content)


class ScriptedPoster:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[RecordedRequest] = []
        self._responses = list(responses)

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.requests.append(RecordedRequest(url, content, headers, timeout))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class ManualExecutor(concurrent.futures.Executor):
    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []

    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        self.pending.append(lambda: fn(*args, **kwargs))
        return future


def token_response(token: str = "sk-ant-oat01-minted", expires_in: int | None = 3600) -> httpx.Response:
    body: Final[dict[str, str | int]] = {
        "access_token": token,
        "token_type": "Bearer",
        **({} if expires_in is None else {"expires_in": expires_in}),
    }
    return httpx.Response(200, json=body)


def make_engine(poster: ScriptedPoster, clock: FakeClock | None = None) -> JwtBearerTokenExchangeEngine:
    return JwtBearerTokenExchangeEngine(
        poster=poster,
        clock=clock if clock is not None else FakeClock(),
        refresh_executor=ManualExecutor(),
    )


def write_token_file(directory: Path, content: str, name: str = "identity-token") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    token_file = directory / name
    token_file.write_text(content, encoding="utf-8")
    return token_file


class TestWireProtocolExact:
    def test_minimal_body_and_headers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_SCOPE", "user:inference")
        token_file = write_token_file(tmp_path, "jwt-assertion-value\n")
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)

        token = get_anthropic_wif_token(
            {
                "anthropic_federation_rule_id": "fdrl_abc123",
                "anthropic_organization_id": "org-uuid-1",
                "anthropic_identity_token_file": str(token_file),
            },
            "https://api.anthropic.com",
            "claude-sonnet-4-5",
            engine,
        )

        assert token == "sk-ant-oat01-minted"
        assert len(poster.requests) == 1
        request = poster.requests[0]
        assert request.url == "https://api.anthropic.com/v1/oauth/token"
        assert "anthropic-beta" not in request.headers
        assert request.headers["content-type"] == "application/json"
        assert request.json_body() == {
            "grant_type": GRANT_TYPE,
            "federation_rule_id": "fdrl_abc123",
            "organization_id": "org-uuid-1",
            "assertion": "jwt-assertion-value",
        }

    def test_optional_fields_present_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = write_token_file(tmp_path, "jwt-assertion-value")
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)

        get_anthropic_wif_token(
            {
                "anthropic_federation_rule_id": "fdrl_abc123",
                "anthropic_organization_id": "org-uuid-1",
                "anthropic_service_account_id": "svcacct_1",
                "anthropic_workspace_id": "wrkspc_1",
                "anthropic_identity_token_file": str(token_file),
            },
            "https://api.anthropic.com",
            "claude-sonnet-4-5",
            engine,
        )

        request = poster.requests[0]
        assert "anthropic-beta" not in request.headers
        assert request.headers["content-type"] == "application/json"
        assert request.json_body() == {
            "grant_type": GRANT_TYPE,
            "federation_rule_id": "fdrl_abc123",
            "organization_id": "org-uuid-1",
            "service_account_id": "svcacct_1",
            "workspace_id": "wrkspc_1",
            "assertion": "jwt-assertion-value",
        }

    def test_spec_cache_key_identity(self):
        params = AnthropicWifParams(
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            assertion_ref="oidc/env/ANTHROPIC_IDENTITY_TOKEN",
        )
        spec = build_anthropic_wif_spec(params, "https://api.anthropic.com")
        assert spec.cache_key_identity == ("fdrl_1", "org-1", "", "")
        assert spec.body_encoding == "json"
        assert spec.assertion_field == "assertion"

    def test_full_params_spec_has_no_request_headers(self):
        """The token exchange sends no anthropic-beta header at all (verified against the
        live endpoint); this must hold even for a fully populated params set, so a future
        edit cannot reintroduce the header gated on service_account_id or workspace_id."""
        params = AnthropicWifParams(
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            service_account_id="svcacct_1",
            workspace_id="wrkspc_1",
            assertion_ref="oidc/env/ANTHROPIC_IDENTITY_TOKEN",
        )
        spec = build_anthropic_wif_spec(params, "https://api.anthropic.com")
        assert dict(spec.request_headers) == {}


class TestExchangeHostTrust:
    """A federated exchange sends the workload's identity token to api_base and presents the minted
    org-scoped token to it, so api_base is a trust decision. Anyone able to write api_base, on the
    deployment or on a credential it references, could otherwise redirect both, which is why this is
    enforced where the exchange is built rather than at each write path."""

    def _mint(self, api_base: str | None, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        poster = ScriptedPoster([token_response()])
        get_anthropic_wif_token(
            {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
            api_base,
            "claude-sonnet-4-5",
            make_engine(poster),
        )
        return poster.requests[0].url

    def test_anthropic_is_trusted_without_configuration(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS", raising=False)
        assert self._mint("https://api.anthropic.com", monkeypatch) == "https://api.anthropic.com/v1/oauth/token"

    def test_an_unlisted_host_never_receives_the_identity_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS", raising=False)
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        poster = ScriptedPoster([token_response()])

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            get_anthropic_wif_token(
                {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
                "https://attacker.example",
                "claude-sonnet-4-5",
                make_engine(poster),
            )

        assert poster.requests == [], "the exchange must be refused before anything is sent"
        assert "attacker.example" in str(exc_info.value)
        assert "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS" in str(exc_info.value), (
            "an operator running a private gateway has to be told how to allow it"
        )

    def test_a_lookalike_host_does_not_pass_on_a_substring(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS", raising=False)
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        poster = ScriptedPoster([token_response()])

        with pytest.raises(litellm.AuthenticationError):
            get_anthropic_wif_token(
                {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
                "https://api.anthropic.com.evil.test",
                "claude-sonnet-4-5",
                make_engine(poster),
            )

        assert poster.requests == []

    def test_an_operator_can_allow_a_private_gateway(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS", "gateway.internal")
        assert self._mint("https://gateway.internal", monkeypatch) == "https://gateway.internal/v1/oauth/token"


class TestBaseUrlDerivation:
    def _mint(self, api_base: str | None, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        # These cases are about how a base is normalised into a token URL, not about which hosts an
        # operator trusts, so the private hosts they use are allowlisted explicitly. The trust
        # boundary itself is covered by TestExchangeHostTrust.
        monkeypatch.setenv(
            "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS",
            "gw.example.com,env.example.com,base.example.com,model.example.com",
        )
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)
        get_anthropic_wif_token(
            {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
            api_base,
            "claude-sonnet-4-5",
            engine,
        )
        return poster.requests[0].url

    def test_explicit_api_base_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_BASE", "https://env.example.com")
        assert self._mint("https://gw.example.com/", monkeypatch) == "https://gw.example.com/v1/oauth/token"

    def test_env_api_base(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_BASE", "https://env.example.com")
        assert self._mint(None, monkeypatch) == "https://env.example.com/v1/oauth/token"

    def test_env_base_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://base.example.com")
        assert self._mint(None, monkeypatch) == "https://base.example.com/v1/oauth/token"

    def test_default_base(self, monkeypatch: pytest.MonkeyPatch):
        assert self._mint(None, monkeypatch) == "https://api.anthropic.com/v1/oauth/token"

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://gw.example.com/v1/messages",
            "https://gw.example.com/v1/messages/",
            "https://gw.example.com/v1/messages//v1/messages",
        ],
    )
    def test_chat_appended_bases_normalize_to_clean_token_url(self, api_base: str, monkeypatch: pytest.MonkeyPatch):
        """main.py appends /v1/messages before dispatch (twice for trailing-slash
        bases); the exchange must still target the deployment base."""
        assert self._mint(api_base, monkeypatch) == "https://gw.example.com/v1/oauth/token"

    def test_trailing_slash_env_base_normalizes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://base.example.com/")
        assert self._mint(None, monkeypatch) == "https://base.example.com/v1/oauth/token"


class TestSecretManagerEnvResolution:
    """WIF env vars resolve through get_secret_str so configured secret managers
    work, exactly like every sibling Anthropic credential."""

    def test_values_resolve_through_get_secret_str(self, monkeypatch: pytest.MonkeyPatch):
        secrets: Final = {
            "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_sm",
            "ANTHROPIC_ORGANIZATION_ID": "org-sm",
            "ANTHROPIC_IDENTITY_TOKEN": "sm-inline-jwt",
        }
        monkeypatch.setattr(
            "litellm.secret_managers.main.get_secret_str",
            lambda secret_name, default_value=None: secrets.get(secret_name, default_value),
        )

        params = resolve_anthropic_wif_params(None)

        assert params == AnthropicWifParams(
            federation_rule_id="fdrl_sm",
            organization_id="org-sm",
            assertion_ref="oidc/env/ANTHROPIC_IDENTITY_TOKEN",
        )

    def test_non_str_secret_value_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch):
        secrets: Final = {
            "ANTHROPIC_FEDERATION_RULE_ID": {"unexpected": "shape"},
            "ANTHROPIC_ORGANIZATION_ID": "org-sm",
            "ANTHROPIC_IDENTITY_TOKEN": "sm-inline-jwt",
        }
        monkeypatch.setattr(
            "litellm.secret_managers.main.get_secret_str",
            lambda secret_name, default_value=None: secrets.get(secret_name, default_value),
        )

        assert resolve_anthropic_wif_params(None) is None


class TestResolutionMatrix:
    def test_params_beat_env_per_field(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_env")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-env")
        monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svc-env")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_env")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", "/var/run/secrets/env-token")

        params = resolve_anthropic_wif_params(
            {
                "anthropic_federation_rule_id": "fdrl_param",
                "anthropic_organization_id": "org-param",
                "anthropic_service_account_id": "svc-param",
                "anthropic_workspace_id": "wrkspc_param",
                "anthropic_identity_token_file": "/var/run/secrets/param-token",
            }
        )

        assert params == AnthropicWifParams(
            federation_rule_id="fdrl_param",
            organization_id="org-param",
            service_account_id="svc-param",
            workspace_id="wrkspc_param",
            assertion_ref="oidc/file//var/run/secrets/param-token",
        )

    def test_env_only_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_env")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-env")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "raw-env-jwt")

        params = resolve_anthropic_wif_params(None)

        assert params is not None
        assert params.assertion_ref == "oidc/env/ANTHROPIC_IDENTITY_TOKEN"
        assert params.service_account_id is None
        assert params.workspace_id is None

    def test_file_param_beats_inline_param(self):
        params = resolve_anthropic_wif_params(
            {
                "anthropic_federation_rule_id": "fdrl_1",
                "anthropic_organization_id": "org-1",
                "anthropic_identity_token_file": "/var/run/secrets/tok",
                "anthropic_identity_token": "oidc/env/OTHER",
            }
        )
        assert params is not None
        assert params.assertion_ref == "oidc/file//var/run/secrets/tok"

    def test_inline_param_beats_env_file(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", "/var/run/secrets/env-tok")
        params = resolve_anthropic_wif_params(
            {
                "anthropic_federation_rule_id": "fdrl_1",
                "anthropic_organization_id": "org-1",
                "anthropic_identity_token": "oidc/env/OTHER",
            }
        )
        assert params is not None
        assert params.assertion_ref == "oidc/env/OTHER"

    def test_env_file_beats_env_inline(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", "/var/run/secrets/env-tok")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "raw-env-jwt")
        params = resolve_anthropic_wif_params(
            {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"}
        )
        assert params is not None
        assert params.assertion_ref == "oidc/file//var/run/secrets/env-tok"

    def test_empty_workspace_env_coerced_to_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
        params = resolve_anthropic_wif_params(
            {
                "anthropic_federation_rule_id": "fdrl_1",
                "anthropic_organization_id": "org-1",
                "anthropic_identity_token": "oidc/env/TOK",
            }
        )
        assert params is not None
        assert params.workspace_id is None
        spec = build_anthropic_wif_spec(params, "https://api.anthropic.com")
        assert "workspace_id" not in spec.static_body

    @pytest.mark.parametrize(
        "litellm_params",
        [
            {},
            {"anthropic_federation_rule_id": "fdrl_1"},
            {"anthropic_organization_id": "org-1"},
            {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
            {"anthropic_organization_id": "org-1", "anthropic_identity_token": "oidc/env/TOK"},
            {"anthropic_federation_rule_id": "fdrl_1", "anthropic_identity_token": "oidc/env/TOK"},
        ],
    )
    def test_gate_unmet_returns_none(self, litellm_params: dict):
        assert resolve_anthropic_wif_params(litellm_params) is None

    def test_gate_unmet_facade_returns_none_without_engine_call(self):
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)
        assert get_anthropic_wif_token({}, None, "claude-sonnet-4-5", engine) is None
        assert poster.requests == []


class TestServiceAccountIdIsOptional:
    """Anthropic's reference docs mark service_account_id required, but a live exchange
    against a federation rule targeting a single service account mints successfully
    without it; resolution must not gate activation on it, and the wire body must omit
    the key entirely rather than send it as null."""

    def test_activates_and_omits_service_account_id_when_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = write_token_file(tmp_path, "jwt-assertion-value")
        litellm_params: Final = {
            "anthropic_federation_rule_id": "fdrl_1",
            "anthropic_organization_id": "org-1",
            "anthropic_identity_token_file": str(token_file),
        }

        params = resolve_anthropic_wif_params(litellm_params)
        assert params is not None
        assert params.service_account_id is None

        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)
        token = get_anthropic_wif_token(litellm_params, "https://api.anthropic.com", "claude-sonnet-4-5", engine)

        assert token == "sk-ant-oat01-minted"
        assert "service_account_id" not in poster.requests[0].json_body()


class TestInlineRefRestrictions:
    RAW_JWT: Final = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ3b3JrbG9hZCJ9.c2lnbmF0dXJl"

    @pytest.mark.parametrize("bad_ref", [RAW_JWT, "oidc/env_path/ANTHROPIC_TOKEN_PATH"])
    def test_rejected_inline_refs(self, bad_ref: str):
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            get_anthropic_wif_token(
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_identity_token": bad_ref,
                },
                None,
                "claude-sonnet-4-5",
                engine,
            )

        assert "oidc/env/" in exc_info.value.message
        assert "oidc/file/" in exc_info.value.message
        assert self.RAW_JWT not in exc_info.value.message
        assert poster.requests == []


class TestFileAllowlistAndSymlink:
    SECRET_CONTENT: Final = "super-secret-jwt-content"

    def _call(self, token_file: Path, poster: ScriptedPoster) -> str | None:
        engine = make_engine(poster)
        return get_anthropic_wif_token(
            {
                "anthropic_federation_rule_id": "fdrl_1",
                "anthropic_organization_id": "org-1",
                "anthropic_identity_token_file": str(token_file),
            },
            "https://api.anthropic.com",
            "claude-sonnet-4-5",
            engine,
        )

    def test_file_outside_allowlist_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path / "allowed"))
        token_file = write_token_file(tmp_path / "outside", self.SECRET_CONTENT)
        poster = ScriptedPoster([token_response()])

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            self._call(token_file, poster)

        assert str(token_file) in exc_info.value.message
        assert self.SECRET_CONTENT not in exc_info.value.message
        assert poster.requests == []

    def test_disallowed_path_message_names_allowlist_and_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """The disallowed_path error must explain the allowlist and name the env var an
        operator would set, not surface as a bare '(disallowed_path)' code dump."""
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path / "allowed"))
        token_file = write_token_file(tmp_path / "outside", self.SECRET_CONTENT)
        poster = ScriptedPoster([token_response()])

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            self._call(token_file, poster)

        message = exc_info.value.message
        assert "(disallowed_path)" not in message
        assert "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS" in message
        assert "allowed credential director" in message

    def test_symlink_escape_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(allowed))
        outside_file = write_token_file(tmp_path / "outside", self.SECRET_CONTENT)
        link = allowed / "identity-token"
        link.symlink_to(outside_file)
        poster = ScriptedPoster([token_response()])

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            self._call(link, poster)

        assert self.SECRET_CONTENT not in exc_info.value.message
        assert poster.requests == []

    def test_file_inside_allowlist_succeeds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = write_token_file(tmp_path, self.SECRET_CONTENT)
        poster = ScriptedPoster([token_response()])

        assert self._call(token_file, poster) == "sk-ant-oat01-minted"
        assert poster.requests[0].json_body()["assertion"] == self.SECRET_CONTENT


class TestErrorMappingExhaustive:
    @pytest.mark.parametrize(
        "error",
        [
            AssertionSourceError(kind="missing", source_ref="oidc/env/TOK"),
            AssertionSourceError(kind="disallowed_path", source_ref="oidc/file//etc/passwd"),
            InsecureTokenUrl(host="token.example"),
            TokenEndpointError(status_code=500, redacted_body="error: server_error"),
            TokenTransportError(detail="ConnectError: refused"),
            MalformedTokenResponse(detail="token response failed RFC 6749 5.1 schema validation"),
        ],
    )
    def test_every_variant_maps_to_authentication_error(self, error: ExchangeError):
        with pytest.raises(litellm.AuthenticationError) as exc_info:
            _raise_anthropic_wif_error(error, model="claude-sonnet-4-5", workspace_id_set=False)

        assert exc_info.value.llm_provider == "anthropic"
        assert exc_info.value.model == "claude-sonnet-4-5"

    def test_assertion_source_error_detail_is_rendered_when_present(self):
        with pytest.raises(litellm.AuthenticationError) as exc_info:
            _raise_anthropic_wif_error(
                AssertionSourceError(kind="unreadable", source_ref="oidc/keycloak/abc123", detail="invalid_client"),
                model="claude-sonnet-4-5",
                workspace_id_set=True,
            )

        assert "invalid_client" in exc_info.value.message

    def test_assertion_source_error_without_detail_is_unchanged(self):
        """Regression floor: the token_file/env path never populates detail, so its message must stay
        byte-identical to before the field existed."""
        with pytest.raises(litellm.AuthenticationError) as exc_info:
            _raise_anthropic_wif_error(
                AssertionSourceError(kind="unreadable", source_ref="oidc/env/ANTHROPIC_IDENTITY_TOKEN"),
                model="claude-sonnet-4-5",
                workspace_id_set=True,
            )

        assert exc_info.value.message == (
            "litellm.AuthenticationError: Anthropic workload identity federation failed. Could not obtain "
            "the OIDC identity token (unreadable) from oidc/env/ANTHROPIC_IDENTITY_TOKEN."
        )

    def test_endpoint_error_raised_through_facade(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        poster = ScriptedPoster([httpx.Response(500, json={"error": "server_error"})])
        engine = make_engine(poster)

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            get_anthropic_wif_token(
                {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
                None,
                "claude-sonnet-4-5",
                engine,
            )

        assert exc_info.value.llm_provider == "anthropic"
        assert "HTTP 500" in exc_info.value.message
        assert "server_error" in exc_info.value.message

    @pytest.mark.parametrize(
        "litellm_params,status_code,body",
        [
            (
                {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"},
                401,
                {"error": "invalid_grant."},
            ),
            (
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_workspace_id": "wrkspc_1",
                },
                500,
                {"error": "server_error."},
            ),
        ],
    )
    def test_token_endpoint_error_message_has_no_doubled_period(
        self, litellm_params: dict, status_code: int, body: dict, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        poster = ScriptedPoster([httpx.Response(status_code, json=body)])
        engine = make_engine(poster)

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            get_anthropic_wif_token(litellm_params, None, "claude-sonnet-4-5", engine)

        assert ".." not in exc_info.value.message


class TestWorkspaceHint:
    def _raise_401(self, litellm_params: dict, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "inline-jwt")
        poster = ScriptedPoster([httpx.Response(401, json={"error": "invalid_grant"})])
        engine = make_engine(poster)
        with pytest.raises(litellm.AuthenticationError) as exc_info:
            get_anthropic_wif_token(litellm_params, None, "claude-sonnet-4-5", engine)
        assert len(poster.requests) == 2
        return exc_info.value.message

    def test_hint_when_workspace_unset(self, monkeypatch: pytest.MonkeyPatch):
        message = self._raise_401(
            {"anthropic_federation_rule_id": "fdrl_1", "anthropic_organization_id": "org-1"}, monkeypatch
        )
        assert "ANTHROPIC_WORKSPACE_ID" in message

    def test_no_hint_when_workspace_set(self, monkeypatch: pytest.MonkeyPatch):
        message = self._raise_401(
            {
                "anthropic_federation_rule_id": "fdrl_1",
                "anthropic_organization_id": "org-1",
                "anthropic_workspace_id": "wrkspc_1",
            },
            monkeypatch,
        )
        assert "ANTHROPIC_WORKSPACE_ID" not in message


class TestFileRereadOnRefresh:
    def test_mandatory_refresh_carries_rotated_assertion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = write_token_file(tmp_path, "first-assertion")
        clock = FakeClock(start=1_000.0)
        poster = ScriptedPoster(
            [token_response("sk-ant-oat01-first", 3600), token_response("sk-ant-oat01-second", 3600)]
        )
        engine = make_engine(poster, clock=clock)
        litellm_params = {
            "anthropic_federation_rule_id": "fdrl_1",
            "anthropic_organization_id": "org-1",
            "anthropic_identity_token_file": str(token_file),
        }

        first = get_anthropic_wif_token(litellm_params, "https://api.anthropic.com", "claude-sonnet-4-5", engine)
        token_file.write_text("second-assertion", encoding="utf-8")
        clock.advance(3600 - 10)
        second = get_anthropic_wif_token(litellm_params, "https://api.anthropic.com", "claude-sonnet-4-5", engine)

        assert first == "sk-ant-oat01-first"
        assert second == "sk-ant-oat01-second"
        assert len(poster.requests) == 2
        assert poster.requests[1].json_body()["assertion"] == "second-assertion"


_ISSUER_PRIVATE_VALUE: Final = 55566677788899900011122233344455566677788899900011122233344455
ISSUER_SIGNING_KEY_REF: Final = "oidc/env/ISSUER_SIGNING_KEY_PEM"
KEYCLOAK_TOKEN_URL: Final = "https://keycloak.internal.example/realms/litellm/protocol/openid-connect/token"


def _issuer_signing_key() -> ec.EllipticCurvePrivateKey:
    return ec.derive_private_key(_ISSUER_PRIVATE_VALUE, ec.SECP256R1())


def _issuer_signing_key_pem() -> str:
    return (
        _issuer_signing_key()
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode()
    )


def _get_secret_str_returning(pem: str, ref: str) -> Callable[..., str | None]:
    def fake_get_secret_str(secret_name: str, default_value: str | None = None) -> str | None:
        return pem if secret_name == ref else default_value

    return fake_get_secret_str


class TestIdentitySourceDiscriminatorAbsentIsByteIdenticalToLegacy:
    """anthropic_identity_source unset must resolve exactly like today: no new dispatch code
    runs, and no assertion_source closure is attached, so the engine falls back to its own
    reader precisely as it always has."""

    def test_file_config_carries_no_assertion_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = write_token_file(tmp_path, "jwt-assertion-value")

        params = resolve_anthropic_wif_params(
            {
                "anthropic_federation_rule_id": "fdrl_1",
                "anthropic_organization_id": "org-1",
                "anthropic_identity_token_file": str(token_file),
            }
        )

        assert params == AnthropicWifParams(
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            assertion_ref=f"oidc/file/{token_file}",
        )
        assert params.assertion_source is None

    def test_env_config_carries_no_assertion_source(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_env")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-env")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "raw-env-jwt")

        params = resolve_anthropic_wif_params(None)

        assert params is not None
        assert params.assertion_ref == "oidc/env/ANTHROPIC_IDENTITY_TOKEN"
        assert params.assertion_source is None


class TestInternalIssuerIdentitySourceDispatch:
    """A config.yaml-shaped litellm_params block for the internal_issuer identity source."""

    LITELLM_PARAMS: Final = {
        "anthropic_federation_rule_id": "fdrl_1",
        "anthropic_organization_id": "org-1",
        "anthropic_identity_source": "internal_issuer",
        "anthropic_issuer_url": "https://issuer.internal.example",
        "anthropic_issuer_subject": "workload-a",
        "anthropic_issuer_ttl_seconds": 300,
        "anthropic_issuer_signing_key_ref": ISSUER_SIGNING_KEY_REF,
    }

    def test_assertion_ref_matches_the_identity_source_hash(self):
        params = resolve_anthropic_wif_params(self.LITELLM_PARAMS)

        assert params is not None
        expected_config = InternalIssuerSource(
            issuer_url="https://issuer.internal.example",
            subject="workload-a",
            ttl_seconds=300,
            signing_key_ref=ISSUER_SIGNING_KEY_REF,
        )
        assert params.assertion_ref == identity_source_ref(expected_config)
        assert params.assertion_ref.startswith("oidc/internal_issuer/")

    def test_ref_is_stable_and_rolls_on_field_change(self):
        first = resolve_anthropic_wif_params(self.LITELLM_PARAMS)
        second = resolve_anthropic_wif_params(dict(self.LITELLM_PARAMS))
        changed = resolve_anthropic_wif_params({**self.LITELLM_PARAMS, "anthropic_issuer_subject": "workload-b"})

        assert first is not None and second is not None and changed is not None
        assert first.assertion_ref == second.assertion_ref
        assert first.assertion_ref != changed.assertion_ref

    def test_assertion_source_mints_a_verifiable_jwt(self, monkeypatch: pytest.MonkeyPatch):
        pem = _issuer_signing_key_pem()
        monkeypatch.setattr(
            "litellm.secret_managers.main.get_secret_str",
            _get_secret_str_returning(pem, ISSUER_SIGNING_KEY_REF),
        )

        params = resolve_anthropic_wif_params(self.LITELLM_PARAMS)
        assert params is not None
        assert params.assertion_source is not None

        assertion = params.assertion_source()

        assert assertion is not None
        public_key = _issuer_signing_key().public_key()
        expected_kid = build_jwks(public_key)["keys"][0]["kid"]
        assert jwt.get_unverified_header(assertion)["kid"] == expected_kid
        assert expected_kid == rfc7638_thumbprint(public_key)
        claims = jwt.decode(assertion, public_key, algorithms=["ES256"], options={"verify_aud": False})
        assert claims["sub"] == "workload-a"
        assert claims["iss"] == "https://issuer.internal.example"

    def test_full_exchange_sends_the_minted_assertion(self, monkeypatch: pytest.MonkeyPatch):
        pem = _issuer_signing_key_pem()
        monkeypatch.setattr(
            "litellm.secret_managers.main.get_secret_str",
            _get_secret_str_returning(pem, ISSUER_SIGNING_KEY_REF),
        )
        poster = ScriptedPoster([token_response()])
        engine = make_engine(poster)

        token = get_anthropic_wif_token(self.LITELLM_PARAMS, "https://api.anthropic.com", "claude-sonnet-4-5", engine)

        assert token == "sk-ant-oat01-minted"
        sent_assertion = poster.requests[0].json_body()["assertion"]
        jwt.decode(
            sent_assertion, _issuer_signing_key().public_key(), algorithms=["ES256"], options={"verify_aud": False}
        )


class TestKeycloakIdentitySourceDispatch:
    """A config.yaml-shaped litellm_params block for the keycloak identity source. The minted
    closure's own network behavior is covered by test_client_credentials.py's DI-poster tests;
    this only proves wif.py threads the fields into the right config and hash."""

    LITELLM_PARAMS: Final = {
        "anthropic_federation_rule_id": "fdrl_1",
        "anthropic_organization_id": "org-1",
        "anthropic_identity_source": "keycloak",
        "anthropic_keycloak_token_url": KEYCLOAK_TOKEN_URL,
        "anthropic_keycloak_client_id": "litellm",
        "anthropic_keycloak_client_secret_ref": "oidc/env/KEYCLOAK_CLIENT_SECRET",
    }

    def test_assertion_ref_matches_the_identity_source_hash(self):
        params = resolve_anthropic_wif_params(self.LITELLM_PARAMS)

        assert params is not None
        expected_config = KeycloakSource(
            token_url=KEYCLOAK_TOKEN_URL,
            client_id="litellm",
            client_secret_ref="oidc/env/KEYCLOAK_CLIENT_SECRET",
        )
        assert params.assertion_ref == identity_source_ref(expected_config)
        assert params.assertion_ref.startswith("oidc/keycloak/")

    def test_assertion_source_is_a_fresh_closure(self):
        params = resolve_anthropic_wif_params(self.LITELLM_PARAMS)

        assert params is not None
        assert params.assertion_source is not None
        assert callable(params.assertion_source)

    def test_auth_method_change_rolls_the_ref(self):
        default_method = resolve_anthropic_wif_params(self.LITELLM_PARAMS)
        post_method = resolve_anthropic_wif_params(
            {**self.LITELLM_PARAMS, "anthropic_keycloak_auth_method": "client_secret_post"}
        )

        assert default_method is not None and post_method is not None
        assert default_method.assertion_ref != post_method.assertion_ref

    def test_client_secret_ref_pointer_name_change_rolls_the_ref_without_resolving_it(self):
        """The hash covers the pointer NAME, never a resolved secret (decision 7) -- true even
        though nothing in this test ever calls get_secret_str."""
        first = resolve_anthropic_wif_params(self.LITELLM_PARAMS)
        second = resolve_anthropic_wif_params(
            {**self.LITELLM_PARAMS, "anthropic_keycloak_client_secret_ref": "oidc/env/OTHER_SECRET_NAME"}
        )

        assert first is not None and second is not None
        assert first.assertion_ref != second.assertion_ref


class TestIdentitySourceValidationFailsClosed:
    """Unknown discriminator, a missing required variant field, and a field belonging to the
    other variant are all hard config errors at resolution time -- never a silent fallback to
    token_file (decision 5)."""

    def test_unknown_discriminator_raises(self):
        with pytest.raises(litellm.AuthenticationError, match="anthropic_identity_source"):
            resolve_anthropic_wif_params(
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_identity_source": "bogus",
                }
            )

    def test_internal_issuer_missing_required_fields_raises(self):
        with pytest.raises(litellm.AuthenticationError):
            resolve_anthropic_wif_params(
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_identity_source": "internal_issuer",
                    "anthropic_issuer_url": "https://issuer.internal.example",
                }
            )

    def test_keycloak_missing_required_fields_raises(self):
        with pytest.raises(litellm.AuthenticationError):
            resolve_anthropic_wif_params(
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_identity_source": "keycloak",
                    "anthropic_keycloak_client_id": "litellm",
                }
            )

    def test_mixed_variant_fields_raise(self):
        with pytest.raises(litellm.AuthenticationError, match="belongs to a different identity source"):
            resolve_anthropic_wif_params(
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_identity_source": "internal_issuer",
                    "anthropic_issuer_url": "https://issuer.internal.example",
                    "anthropic_issuer_subject": "workload-a",
                    "anthropic_issuer_signing_key_ref": ISSUER_SIGNING_KEY_REF,
                    "anthropic_keycloak_client_id": "leaked-from-other-variant",
                }
            )

    def test_secret_pasted_into_wrong_field_never_appears_in_the_error(self):
        secret_value = "super-secret-client-value-xyz"
        with pytest.raises(litellm.AuthenticationError) as exc_info:
            resolve_anthropic_wif_params(
                {
                    "anthropic_federation_rule_id": "fdrl_1",
                    "anthropic_organization_id": "org-1",
                    "anthropic_identity_source": "internal_issuer",
                    "anthropic_issuer_url": "https://issuer.internal.example",
                    "anthropic_issuer_subject": "workload-a",
                    "anthropic_issuer_signing_key_ref": ISSUER_SIGNING_KEY_REF,
                    "anthropic_issuer_ttl_seconds": secret_value,
                }
            )

        assert secret_value not in exc_info.value.message


class TestConfigYamlShapedIdentitySources:
    """One litellm_params dict per identity source, shaped exactly like the
    model_list[].litellm_params block a proxy config.yaml carries -- proving an operator can
    configure each of Phase 1's supported sources."""

    def test_legacy_token_file_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = write_token_file(tmp_path, "jwt-assertion-value")
        litellm_params = {
            "model": "anthropic/claude-sonnet-4-5",
            "anthropic_federation_rule_id": "fdrl_prod",
            "anthropic_organization_id": "org_prod",
            "anthropic_identity_token_file": str(token_file),
        }

        params = resolve_anthropic_wif_params(litellm_params)

        assert params is not None
        assert params.assertion_ref == f"oidc/file/{token_file}"
        assert params.assertion_source is None

    def test_internal_issuer_source(self):
        litellm_params = {
            "model": "anthropic/claude-sonnet-4-5",
            "anthropic_federation_rule_id": "fdrl_prod",
            "anthropic_organization_id": "org_prod",
            "anthropic_identity_source": "internal_issuer",
            "anthropic_issuer_url": "https://litellm.internal.example",
            "anthropic_issuer_subject": "litellm-proxy",
            "anthropic_issuer_ttl_seconds": 300,
            "anthropic_issuer_signing_key_ref": "os.environ/ISSUER_SIGNING_KEY_PEM",
        }

        params = resolve_anthropic_wif_params(litellm_params)

        assert params is not None
        assert params.assertion_ref.startswith("oidc/internal_issuer/")
        assert params.assertion_source is not None

    def test_keycloak_source(self):
        litellm_params = {
            "model": "anthropic/claude-sonnet-4-5",
            "anthropic_federation_rule_id": "fdrl_prod",
            "anthropic_organization_id": "org_prod",
            "anthropic_identity_source": "keycloak",
            "anthropic_keycloak_token_url": KEYCLOAK_TOKEN_URL,
            "anthropic_keycloak_client_id": "litellm",
            "anthropic_keycloak_auth_method": "client_secret_post",
            "anthropic_keycloak_client_secret_ref": "os.environ/KEYCLOAK_CLIENT_SECRET",
            "anthropic_keycloak_scope": "anthropic-wif",
        }

        params = resolve_anthropic_wif_params(litellm_params)

        assert params is not None
        assert params.assertion_ref.startswith("oidc/keycloak/")
        assert params.assertion_source is not None
