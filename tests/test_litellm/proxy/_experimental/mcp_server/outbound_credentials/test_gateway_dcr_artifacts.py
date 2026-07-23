"""Tests for the sealed artifacts of the aggregate gateway DCR front door.

The openers are the gateway's trust boundary for attacker-controlled values, so the bar here is
totality: every hostile candidate must map to a typed failure rather than raise, and no artifact of
one kind may ever open as another.
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.gateway_dcr_artifacts import (
    AUTHORIZATION_CODE_TTL_SECONDS,
    CONNECT_FLOW_TTL_SECONDS,
    MAX_ARTIFACT_BYTES,
    ArtifactBadSignature,
    ArtifactExpired,
    ArtifactMalformed,
    ArtifactTooLarge,
    AuthorizationCode,
    ConnectFlow,
    NotThisArtifact,
    OpenedAuthorizationCode,
    OpenedConnectFlow,
    RegisteredClient,
    gateway_dcr_keys_from_master_key,
    is_gateway_authorization_code,
    is_gateway_dcr_client_id,
    open_authorization_code,
    open_client,
    open_connect_flow,
    seal_authorization_code,
    seal_client,
    seal_connect_flow,
)

NOW = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
KEYS = gateway_dcr_keys_from_master_key("sk-master-for-artifact-tests")
OTHER_KEYS = gateway_dcr_keys_from_master_key("sk-a-completely-different-master-key")

CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"


def _client(**overrides) -> RegisteredClient:
    return RegisteredClient(**{"redirect_uris": (CLAUDE_CALLBACK,), "client_name": "Claude", **overrides})


def _flow(**overrides) -> ConnectFlow:
    return ConnectFlow(
        **{
            "client_id": "llm_dcrc_x",
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            "user_id": "user-1",
            "state": "opaque-client-state",
            **overrides,
        }
    )


def _code(**overrides) -> AuthorizationCode:
    return AuthorizationCode(
        **{
            "client_id": "llm_dcrc_x",
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            "user_id": "user-1",
            **overrides,
        }
    )


class TestRoundTrip:
    def test_a_registered_client_reopens_with_its_redirect_uris_intact(self):
        """A public DCR client's redirect URIs ARE the registration; losing them would leave the
        authorize step with nothing to validate the presented redirect_uri against."""
        uris = (CLAUDE_CALLBACK, "http://127.0.0.1:8976/callback")

        client_id = seal_client(_client(redirect_uris=uris), KEYS, NOW)

        assert is_gateway_dcr_client_id(client_id)
        assert open_client(client_id, KEYS) == RegisteredClient(redirect_uris=uris, client_name="Claude")

    def test_a_connect_flow_reopens_with_a_single_use_handle(self):
        opened = open_connect_flow(seal_connect_flow(_flow(), KEYS, NOW), KEYS, NOW)

        assert isinstance(opened, OpenedConnectFlow)
        assert opened.flow == _flow()
        assert opened.jti

    def test_an_authorization_code_reopens_with_a_single_use_handle(self):
        sealed = seal_authorization_code(_code(), KEYS, NOW)
        opened = open_authorization_code(sealed, KEYS, NOW)

        assert is_gateway_authorization_code(sealed)
        assert isinstance(opened, OpenedAuthorizationCode)
        assert opened.code == _code()
        assert opened.jti

    def test_two_seals_of_the_same_value_get_distinct_single_use_handles(self):
        """The jti is what a replay guard claims, so two codes issued for the same request must
        not collide into one another's guard slot."""
        first = open_authorization_code(seal_authorization_code(_code(), KEYS, NOW), KEYS, NOW)
        second = open_authorization_code(seal_authorization_code(_code(), KEYS, NOW), KEYS, NOW)

        assert isinstance(first, OpenedAuthorizationCode) and isinstance(second, OpenedAuthorizationCode)
        assert first.jti != second.jti


class TestExpiry:
    def test_an_authorization_code_is_dead_the_instant_its_ttl_elapses(self):
        sealed = seal_authorization_code(_code(), KEYS, NOW)
        just_expired = NOW + timedelta(seconds=AUTHORIZATION_CODE_TTL_SECONDS)

        assert isinstance(open_authorization_code(sealed, KEYS, just_expired - timedelta(seconds=1)), OpenedAuthorizationCode)
        assert isinstance(open_authorization_code(sealed, KEYS, just_expired), ArtifactExpired)

    def test_a_connect_flow_expires_on_its_own_longer_window(self):
        sealed = seal_connect_flow(_flow(), KEYS, NOW)

        assert isinstance(open_connect_flow(sealed, KEYS, NOW + timedelta(seconds=CONNECT_FLOW_TTL_SECONDS - 1)), OpenedConnectFlow)
        assert isinstance(open_connect_flow(sealed, KEYS, NOW + timedelta(seconds=CONNECT_FLOW_TTL_SECONDS)), ArtifactExpired)

    def test_a_registration_does_not_expire(self):
        """An OAuth client registration is long-lived; a clock is not what bounds it."""
        client_id = seal_client(_client(), KEYS, NOW)

        assert open_client(client_id, KEYS) == _client()


class TestKindAndKeySeparation:
    @pytest.mark.parametrize(
        "opener",
        [
            lambda value: open_connect_flow(value, KEYS, NOW),
            lambda value: open_authorization_code(value, KEYS, NOW),
        ],
    )
    def test_a_client_id_never_opens_as_another_artifact(self, opener):
        assert isinstance(opener(seal_client(_client(), KEYS, NOW)), NotThisArtifact)

    def test_an_authorization_code_re_prefixed_as_a_connect_flow_is_rejected(self):
        """The wire prefix is not signed, so the ``kind`` claim is what actually separates the
        artifacts; swapping the prefix must not promote a code into a flow."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.gateway_dcr_artifacts import (
            AUTHORIZATION_CODE_PREFIX,
            CONNECT_FLOW_PREFIX,
        )

        sealed = seal_authorization_code(_code(), KEYS, NOW)
        re_prefixed = CONNECT_FLOW_PREFIX + sealed.removeprefix(AUTHORIZATION_CODE_PREFIX)

        assert isinstance(open_connect_flow(re_prefixed, KEYS, NOW), ArtifactMalformed)

    def test_an_artifact_signed_under_another_master_key_is_rejected(self):
        assert isinstance(open_client(seal_client(_client(), OTHER_KEYS, NOW), KEYS), ArtifactBadSignature)

    def test_a_session_token_is_not_mistaken_for_a_gateway_artifact(self):
        assert isinstance(open_client("llm_session_abc.def.ghi", KEYS), NotThisArtifact)
        assert not is_gateway_dcr_client_id("llm_session_abc")


class TestHostileInput:
    @pytest.mark.parametrize(
        "candidate",
        [
            "",
            "llm_dcrc_",
            "llm_dcrc_not-a-jwt",
            "llm_dcrc_a.b.c",
            "llm_dcrc_" + "\udcff",
            "llm_dcrc_" + "A" * (MAX_ARTIFACT_BYTES + 1),
        ],
        ids=["empty", "prefix-only", "not-a-jwt", "three-junk-segments", "lone-surrogate", "oversize"],
    )
    def test_every_hostile_client_id_returns_a_typed_failure(self, candidate):
        """Total over hostile input: the openers must never raise, because they run on an
        unauthenticated request path where an exception is a 500 instead of an OAuth error."""
        result = open_client(candidate, KEYS)

        assert isinstance(result, (NotThisArtifact, ArtifactBadSignature, ArtifactMalformed, ArtifactExpired))

    def test_a_forged_unsigned_artifact_is_rejected(self):
        """``alg: none`` is the classic JWT downgrade; the decoder pins HS256."""
        forged = "llm_dcrc_" + jwt.encode({"iss": "litellm-mcp-gateway-dcr", "kind": "client"}, "", algorithm="none")

        assert isinstance(open_client(forged, KEYS), ArtifactMalformed)

    def test_an_artifact_from_a_foreign_issuer_is_rejected(self):
        foreign = "llm_dcrc_" + jwt.encode(
            {"iss": "litellm-mcp-gateway", "iat": 0, "jti": "j", "kind": "client", "redirect_uris": [CLAUDE_CALLBACK]},
            KEYS.signing_key.get_secret_value(),
            algorithm="HS256",
        )

        assert isinstance(open_client(foreign, KEYS), ArtifactMalformed)

    def test_an_unknown_extra_claim_is_rejected(self):
        """``extra="forbid"`` keeps a signed-but-unexpected claim from riding along into a future
        reader that starts trusting it."""
        smuggled = "llm_dcrc_" + jwt.encode(
            {
                "iss": "litellm-mcp-gateway-dcr",
                "iat": 0,
                "jti": "j",
                "kind": "client",
                "redirect_uris": [CLAUDE_CALLBACK],
                "user_role": "proxy_admin",
            },
            KEYS.signing_key.get_secret_value(),
            algorithm="HS256",
        )

        assert isinstance(open_client(smuggled, KEYS), ArtifactMalformed)

    def test_an_unknown_extra_claim_on_an_authorization_code_is_rejected(self):
        """The expiring artifacts need their own coverage: the client claims are a separate model,
        so a test that only smuggles into a registration leaves these two unguarded."""
        smuggled = "llm_gcode_" + jwt.encode(
            {
                "iss": "litellm-mcp-gateway-dcr",
                "iat": 0,
                "exp": int((NOW + timedelta(hours=1)).timestamp()),
                "jti": "j",
                "kind": "authorization_code",
                "client_id": "llm_dcrc_x",
                "redirect_uri": CLAUDE_CALLBACK,
                "code_challenge": "cc",
                "user_id": "user-1",
                "scope": "everything",
            },
            KEYS.signing_key.get_secret_value(),
            algorithm="HS256",
        )

        assert isinstance(open_authorization_code(smuggled, KEYS, NOW), ArtifactMalformed)

    def test_a_flow_shaped_token_claiming_the_wrong_kind_is_rejected(self):
        """``kind`` separation is enforced by each model's Literal rather than a comparison, so it
        needs a case that differs ONLY in that claim."""
        smuggled = "llm_gflow_" + jwt.encode(
            {
                "iss": "litellm-mcp-gateway-dcr",
                "iat": 0,
                "exp": int((NOW + timedelta(hours=1)).timestamp()),
                "jti": "j",
                "kind": "authorization_code",
                "client_id": "llm_dcrc_x",
                "redirect_uri": CLAUDE_CALLBACK,
                "code_challenge": "cc",
                "user_id": "user-1",
                "state": "s",
            },
            KEYS.signing_key.get_secret_value(),
            algorithm="HS256",
        )

        assert isinstance(open_connect_flow(smuggled, KEYS, NOW), ArtifactMalformed)

    def test_an_empty_redirect_uri_list_is_rejected(self):
        empty = "llm_dcrc_" + jwt.encode(
            {"iss": "litellm-mcp-gateway-dcr", "iat": 0, "jti": "j", "kind": "client", "redirect_uris": []},
            KEYS.signing_key.get_secret_value(),
            algorithm="HS256",
        )

        assert isinstance(open_client(empty, KEYS), ArtifactMalformed)


class TestSizeCap:
    def test_an_oversize_registration_is_a_typed_mint_failure_not_an_exception(self):
        huge = seal_client(_client(client_name="n" * (MAX_ARTIFACT_BYTES * 2)), KEYS, NOW)

        assert isinstance(huge, ArtifactTooLarge)
        assert huge.max_bytes == MAX_ARTIFACT_BYTES


class TestKeyDerivation:
    def test_different_master_keys_derive_different_signing_keys(self):
        assert KEYS.signing_key.get_secret_value() != OTHER_KEYS.signing_key.get_secret_value()

    def test_the_signing_key_is_not_the_master_key(self):
        assert "sk-master-for-artifact-tests" not in KEYS.signing_key.get_secret_value()

    def test_the_domain_label_separates_gateway_artifacts_from_session_tokens(self):
        """Key separation is the backstop if issuer/prefix/kind separation is ever weakened."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
            session_keys_from_master_key,
        )

        master = "sk-shared-master-key-for-both-families"

        assert (
            gateway_dcr_keys_from_master_key(master).signing_key.get_secret_value()
            != session_keys_from_master_key(master).signing_key.get_secret_value()
        )
