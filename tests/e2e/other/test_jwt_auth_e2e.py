"""Live e2e: RS256 JWTs minted by the test-only issuer (jwt_issuer.py) against a
proxy running with `enable_jwt_auth: true` and the `litellm_jwtauth` block from
CONTRIBUTING.md (sub -> user_id, email -> user_email, groups -> team ids,
user_id_upsert).

Every case mints through the issuer, so the tests never hold a signing key: the
bad-signature case corrupts a genuine signature, the expired case asks the
issuer for a token whose `exp` is already in the past. Those identities get
their own freshly created team so a rejection can only be blamed on the token,
while the unknown-team case names a team that was never created. An acceptance
is proven twice, at the boundary (200 from a real provider) and in the spend log
the proxy attributes to the claims. The last case keeps a plain `sk-` virtual
key working on the same proxy, guarding against the flag turning JWT on for
everyone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import UnauthorizedError, UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, JwtClaimsBody, TeamNewBody
from other_client import OtherClient

pytestmark = pytest.mark.e2e


@dataclass(frozen=True, slots=True)
class JwtIdentity:
    user_id: str
    team_id: str

    def claims(self, *, exp: int | None = None) -> JwtClaimsBody:
        return JwtClaimsBody(sub=self.user_id, email=f"{self.user_id}@example.com", groups=(self.team_id,), exp=exp)


@pytest.fixture
def identity(client: OtherClient, resources: ResourceManager) -> JwtIdentity:
    marker: Final = unique_marker()
    team_id: Final = client.proxy.create_team(
        TeamNewBody(team_alias=f"e2e-jwt-{marker}", team_id=f"e2e-jwt-team-{marker}")
    )
    resources.defer(lambda: client.proxy.delete_team(team_id))
    user_id: Final = f"e2e-jwt-user-{marker}"
    resources.defer(lambda: client.proxy.delete_user(user_id))
    return JwtIdentity(user_id=user_id, team_id=team_id)


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
        self, client: OtherClient, identity: JwtIdentity
    ) -> None:
        token: Final = client.mint_jwt(identity.claims())

        response: Final = unwrap(client.proxy.chat(token, _ping()))
        assert response.id is not None and response.choices, (
            f"chat under a valid JWT returned no completion: {response}"
        )

        rows: Final = client.proxy.poll_logs_for_request_id(response.id)
        assert rows, f"no spend log row for request {response.id} within the poll deadline"
        row: Final = rows[0]
        assert row.team_id == identity.team_id, (
            f"spend row must carry the team from the JWT groups claim {identity.team_id!r}, got {row.team_id!r}"
        )
        assert row.user == identity.user_id, (
            f"spend row must carry the user from the JWT sub claim {identity.user_id!r}, got {row.user!r}"
        )

    @pytest.mark.covers("other.auth.jwt.invalid_signature_denied")
    def test_tampered_signature_is_rejected(self, client: OtherClient, identity: JwtIdentity) -> None:
        tampered: Final = _corrupt_signature(client.mint_jwt(identity.claims()))

        result: Final = client.proxy.chat(tampered, _ping())
        assert isinstance(result, UnauthorizedError), (
            f"a JWT whose signature does not verify must be rejected with 401, got {result}"
        )
        assert "signature verification failed" in result.body.lower(), (
            f"the 401 must come from signature verification, not another auth failure, got {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.jwt.expired_denied")
    def test_expired_token_is_rejected(self, client: OtherClient, identity: JwtIdentity) -> None:
        expired: Final = client.mint_jwt(identity.claims(exp=1))

        result: Final = client.proxy.chat(expired, _ping())
        assert isinstance(result, UnauthorizedError), (
            f"an expired JWT must be rejected with 401 even though its signature verifies, got {result}"
        )
        assert "expired" in result.body.lower(), f"the 401 must say the token expired, got {result.body[:300]}"

    @pytest.mark.covers("other.auth.jwt.unknown_team_denied")
    def test_token_naming_a_team_that_does_not_exist_is_rejected(self, client: OtherClient) -> None:
        marker: Final = unique_marker()
        never_created: Final = JwtIdentity(user_id=f"e2e-jwt-user-{marker}", team_id=f"e2e-jwt-missing-team-{marker}")
        token: Final = client.mint_jwt(never_created.claims())

        result: Final = client.proxy.chat(token, _ping())
        assert isinstance(result, UnknownApiError) and result.status_code == 403, (
            f"a valid JWT whose groups name no existing team must be rejected with 403, got {result}"
        )
        assert never_created.team_id in result.body, (
            f"the 403 must name the team it could not resolve ({never_created.team_id}), got {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.jwt.virtual_key_unaffected")
    def test_plain_virtual_key_still_works_with_jwt_auth_enabled(self, client: OtherClient, scoped_key: str) -> None:
        response: Final = unwrap(client.proxy.chat(scoped_key, _ping()))
        assert response.choices, f"an sk- key must keep working on a proxy with enable_jwt_auth, got {response}"
