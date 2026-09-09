"""Live e2e: access tokens issued by a real Keycloak realm (idp.py) against a
proxy running with `enable_jwt_auth: true` and the `litellm_jwtauth` block from
CONTRIBUTING.md (sub -> user_id, email -> user_email, groups -> team ids,
user_id_upsert).

Every identity is provisioned in Keycloak for the test that uses it: a group
named after the litellm team, and a user in that group whose password exists
only for the length of the test. Tokens then come from Keycloak's direct-access
grant, so no test ever holds a signing key and the claims the proxy reads are
the ones an IdP really emits (`sub` is Keycloak's user uuid, `groups` comes off
a protocol mapper, `aud` is Keycloak's own audience).

The rejection cases stay honest about where the rejection has to come from: the
bad-signature case corrupts a genuine signature, and the expiry case takes its
token from the realm's one-second client and waits for it to lapse rather than
forging a stale `exp`. Those identities get their own freshly created team, so a
rejection can only be blamed on the token, while the unknown-team case names a
group no litellm team was ever created for. An acceptance is proven twice, at
the boundary (200 from a real provider) and in the spend log the proxy
attributes to the claims. The last case keeps a plain `sk-` virtual key working
on the same proxy, guarding against the flag turning JWT on for everyone.
"""

from __future__ import annotations

import time
from typing import Final

import pytest

from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import UnauthorizedError, UnknownApiError, unwrap
from idp import SHORT_LIVED_CLIENT_ID, SHORT_LIVED_TOKEN_SECONDS, Identity
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, TeamNewBody
from other_client import OtherClient

pytestmark = pytest.mark.e2e


def _provision(client: OtherClient, resources: ResourceManager, *, marker: str) -> Identity:
    """A Keycloak group and a user in it, torn down with the test. The group name
    is what the token's `groups` claim carries, which is what the proxy resolves
    as a litellm team id."""
    identity: Final = client.idp.provision(marker=marker, group=f"e2e-jwt-team-{marker}")
    resources.defer(lambda: client.idp.delete_user(identity.user_id))
    resources.defer(lambda: client.idp.delete_group(identity.group_id))
    resources.defer(lambda: client.proxy.delete_user(identity.user_id))
    return identity


@pytest.fixture
def identity(client: OtherClient, resources: ResourceManager) -> Identity:
    """An IdP identity whose group is also a real litellm team, so anything the
    proxy rejects is about the token and never about an unresolvable team."""
    marker: Final = unique_marker()
    provisioned: Final = _provision(client, resources, marker=marker)
    team_id: Final = client.proxy.create_team(TeamNewBody(team_alias=f"e2e-jwt-{marker}", team_id=provisioned.group))
    resources.defer(lambda: client.proxy.delete_team(team_id))
    return provisioned


def _ping() -> ChatBody:
    return ChatBody(
        model=CHEAP_OPENAI_MODEL,
        messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
        max_tokens=16,
    )


def _corrupt_signature(token: str) -> str:
    header, payload, signature = token.split(".")
    flipped: Final = "A" if signature[10] != "A" else "B"
    return f"{header}.{payload}.{signature[:10]}{flipped}{signature[11:]}"


class TestJwtAuth:
    @pytest.mark.covers("other.auth.jwt.valid_token_allows", "other.auth.jwt.spend_attributed_to_claims")
    def test_valid_token_for_an_existing_team_is_accepted_and_attributed(
        self, client: OtherClient, identity: Identity
    ) -> None:
        token: Final = client.idp.access_token(identity)

        response: Final = unwrap(client.proxy.chat(token, _ping()))
        assert response.id is not None and response.choices, (
            f"chat under a valid JWT returned no completion: {response}"
        )

        rows: Final = client.proxy.poll_logs_for_request_id(response.id)
        assert rows, f"no spend log row for request {response.id} within the poll deadline"
        row: Final = rows[0]
        assert row.team_id == identity.group, (
            f"spend row must carry the team from the JWT groups claim {identity.group!r}, got {row.team_id!r}"
        )
        assert row.user == identity.user_id, (
            f"spend row must carry the user from the JWT sub claim {identity.user_id!r}, got {row.user!r}"
        )

    @pytest.mark.covers("other.auth.jwt.invalid_signature_denied")
    def test_tampered_signature_is_rejected(self, client: OtherClient, identity: Identity) -> None:
        tampered: Final = _corrupt_signature(client.idp.access_token(identity))

        result: Final = client.proxy.chat(tampered, _ping())
        assert isinstance(result, UnauthorizedError), (
            f"a JWT whose signature does not verify must be rejected with 401, got {result}"
        )
        assert "signature verification failed" in result.body.lower(), (
            f"the 401 must come from signature verification, not another auth failure, got {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.jwt.expired_denied")
    def test_expired_token_is_rejected(self, client: OtherClient, identity: Identity) -> None:
        expiring: Final = client.idp.access_token(identity, client_id=SHORT_LIVED_CLIENT_ID)
        time.sleep(SHORT_LIVED_TOKEN_SECONDS + 1)

        result: Final = client.proxy.chat(expiring, _ping())
        assert isinstance(result, UnauthorizedError), (
            f"an expired JWT must be rejected with 401 even though its signature verifies, got {result}"
        )
        assert "expired" in result.body.lower(), f"the 401 must say the token expired, got {result.body[:300]}"

    @pytest.mark.covers("other.auth.jwt.unknown_team_denied")
    def test_token_naming_a_team_that_does_not_exist_is_rejected(
        self, client: OtherClient, resources: ResourceManager
    ) -> None:
        stranger: Final = _provision(client, resources, marker=unique_marker())
        token: Final = client.idp.access_token(stranger)

        result: Final = client.proxy.chat(token, _ping())
        assert isinstance(result, UnknownApiError) and result.status_code == 403, (
            f"a valid JWT whose groups name no existing team must be rejected with 403, got {result}"
        )
        assert stranger.group in result.body, (
            f"the 403 must name the team it could not resolve ({stranger.group}), got {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.jwt.virtual_key_unaffected")
    def test_plain_virtual_key_still_works_with_jwt_auth_enabled(self, client: OtherClient, scoped_key: str) -> None:
        response: Final = unwrap(client.proxy.chat(scoped_key, _ping()))
        assert response.choices, f"an sk- key must keep working on a proxy with enable_jwt_auth, got {response}"
