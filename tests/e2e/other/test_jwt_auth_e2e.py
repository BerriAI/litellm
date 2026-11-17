"""Real Keycloak tokens exercise verification, attribution and virtual-key coexistence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Final, Literal

import pytest
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from e2e_config import CHEAP_OPENAI_MODEL, settle_propagation, unique_marker
from e2e_http import NoBody, UnauthorizedError, UnknownApiError, get_external, unwrap
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from idp import SHORT_LIVED_CLIENT_ID, WRONG_AUDIENCE_CLIENT_ID, Identity
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, TeamBlockBody, TeamNewBody, UserNewBody
from other_client import OtherClient
from pydantic import BaseModel, JsonValue, TypeAdapter

pytestmark = pytest.mark.e2e


class IssuedClaims(BaseModel):
    """Read the IdP's signed payload only to check the test precondition."""

    exp: int
    sub: str
    iss: str
    aud: str | list[str]


def _claims(token: str) -> IssuedClaims:
    payload: Final = token.split(".")[1]
    return IssuedClaims.model_validate_json(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def _provision(client: OtherClient, resources: ResourceManager, *, marker: str) -> Identity:
    """A Keycloak group and a user in it, torn down with the test. The group name
    is what the token's `groups` claim carries, which is what the proxy resolves
    as a litellm team id."""
    identity: Final = client.idp.provision(marker=marker, group=f"e2e-jwt-team-{marker}", defer=resources.defer)
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


@dataclass(frozen=True, slots=True)
class BoundTeam:
    identity: Identity
    team_id: str
    team_alias: str


def _team(client: OtherClient, resources: ResourceManager, *, marker: str, team_id: str) -> str:
    """A litellm team whose alias differs from its id, so a header naming one
    cannot accidentally match the other."""
    team_alias: Final = f"e2e-jwt-alias-{marker}"
    created: Final = client.proxy.create_team(TeamNewBody(team_alias=team_alias, team_id=team_id))
    resources.defer(lambda: client.proxy.delete_team(created))
    return team_alias


@pytest.fixture
def bound_team(client: OtherClient, resources: ResourceManager) -> BoundTeam:
    """An identity whose single group is a real team, plus that team's alias."""
    marker: Final = unique_marker()
    provisioned: Final = _provision(client, resources, marker=marker)
    team_alias: Final = _team(client, resources, marker=marker, team_id=provisioned.group)
    return BoundTeam(identity=provisioned, team_id=provisioned.group, team_alias=team_alias)


@dataclass(frozen=True, slots=True)
class AliasedTeam:
    identity: Identity
    alias: str
    target: str


@pytest.fixture
def aliased_team(client: OtherClient, resources: ResourceManager) -> AliasedTeam:
    """An identity whose team carries a model_aliases entry, the name a managed
    client such as Claude Code sends and the team rewrites to a real model group."""
    marker: Final = unique_marker()
    provisioned: Final = _provision(client, resources, marker=marker)
    alias: Final = f"e2e-jwt-model-alias-{marker}"
    team_id: Final = client.proxy.create_team(
        TeamNewBody(
            team_alias=f"e2e-jwt-aliased-{marker}",
            team_id=provisioned.group,
            models=[CHEAP_OPENAI_MODEL],
            model_aliases={alias: CHEAP_OPENAI_MODEL},
        )
    )
    resources.defer(lambda: client.proxy.delete_team(team_id))
    return AliasedTeam(identity=provisioned, alias=alias, target=CHEAP_OPENAI_MODEL)


def _ping(model: str = CHEAP_OPENAI_MODEL) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
        max_tokens=16,
    )


def _corrupt_signature(token: str) -> str:
    header, payload, signature = token.split(".")
    flipped: Final = "A" if signature[10] != "A" else "B"
    return f"{header}.{payload}.{signature[:10]}{flipped}{signature[11:]}"


class TestJwtAuth:
    @pytest.mark.covers("other.auth.jwt.valid_token_allows", "other.auth.jwt.spend_attributed_to_claims")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_valid_token_for_an_existing_team_is_accepted_and_attributed(
        self, client: OtherClient, identity: Identity
    ) -> None:
        token: Final = client.idp.access_token(identity)

        assert _claims(token).sub == identity.user_id, "IdP must emit the provisioned user as sub"
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

    @pytest.mark.covers("other.auth.jwt.proxy_admin_personal_budget_enforced")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_proxy_admin_token_is_blocked_once_its_personal_budget_is_spent(
        self, client: OtherClient, resources: ResourceManager
    ) -> None:
        identity: Final = client.idp.provision_user(
            marker=unique_marker(), groups=(), group_ids=(), defer=resources.defer
        )
        resources.defer(lambda: client.proxy.delete_user(identity.user_id))
        unwrap(
            client.user_new(
                UserNewBody(
                    user_id=identity.user_id,
                    user_email=f"{identity.username}@example.com",
                    user_role="proxy_admin",
                    auto_create_key=False,
                    max_budget=0.0,
                )
            )
        )

        blocked: Final = client.proxy.chat(client.idp.access_token(identity), _ping())
        assert isinstance(blocked, UnknownApiError), f"proxy_admin JWT must be stopped by its max_budget, got {blocked}"
        assert (blocked.status_code, "ExceededBudget" in blocked.body) == (422, True), blocked
        assert f"User={identity.user_id}" in blocked.body, f"budget denial must name the JWT's user: {blocked.body}"

    @pytest.mark.covers("other.auth.jwt.invalid_signature_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_tampered_signature_is_rejected(self, client: OtherClient, identity: Identity) -> None:
        tampered: Final = _corrupt_signature(client.idp.access_token(identity))

        result: Final = client.proxy.chat(tampered, _ping())
        assert isinstance(result, UnauthorizedError), (
            f"a JWT whose signature does not verify must be rejected with 401, got {result}"
        )
        assert "signature" in result.body.lower(), (
            f"the 401 must come from signature verification, not another auth failure, got {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.jwt.expired_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_expired_token_is_rejected(self, client: OtherClient, identity: Identity) -> None:
        expiring: Final = client.idp.access_token(identity, client_id=SHORT_LIVED_CLIENT_ID)
        delay: Final = _claims(expiring).exp - time.time() + 1
        assert delay <= 5, f"short-lived client expiry or IdP clock drifted: wait would be {delay}s"
        time.sleep(max(0, delay))

        result: Final = client.proxy.chat(expiring, _ping())
        assert isinstance(result, UnauthorizedError), (
            f"an expired JWT must be rejected with 401 even though its signature verifies, got {result}"
        )
        assert "expired" in result.body.lower(), f"the 401 must say the token expired, got {result.body[:300]}"

    @pytest.mark.covers("other.auth.jwt.wrong_issuer_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_signed_token_from_the_wrong_issuer_is_rejected(self, client: OtherClient, identity: Identity) -> None:
        token: Final = client.idp.access_token(identity, issuer_host="unexpected-issuer.invalid")
        claims: Final = _claims(token)
        assert claims.iss != client.idp.issuer and "litellm-e2e" in claims.aud

        result: Final = client.proxy.chat(token, _ping())
        assert isinstance(result, UnauthorizedError), f"wrong issuer must be rejected: {result}"
        assert "issuer" in result.body.lower(), f"expected issuer validation to reject the token: {result}"

    @pytest.mark.covers("other.auth.jwt.wrong_audience_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_signed_token_for_another_application_is_rejected(self, client: OtherClient, identity: Identity) -> None:
        token: Final = client.idp.access_token(identity, client_id=WRONG_AUDIENCE_CLIENT_ID)
        claims: Final = _claims(token)
        assert claims.iss == client.idp.issuer and "litellm-e2e" not in (
            [claims.aud] if isinstance(claims.aud, str) else claims.aud
        )

        result: Final = client.proxy.chat(token, _ping())
        assert isinstance(result, UnauthorizedError), f"wrong audience must be rejected: {result}"
        assert "audience" in result.body.lower(), f"expected audience validation to reject the token: {result}"

    @pytest.mark.covers("other.auth.jwt.unknown_team_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_plain_virtual_key_still_works_with_jwt_auth_enabled(self, client: OtherClient, scoped_key: str) -> None:
        response: Final = unwrap(client.proxy.chat(scoped_key, _ping()))
        assert response.choices, f"an sk- key must keep working on a proxy with enable_jwt_auth, got {response}"


def _team_of_request(client: OtherClient, token: str, team: str) -> str | None:
    response: Final = unwrap(client.chat_as_team(token, team, _ping()))
    assert response.id is not None and response.choices, (
        f"chat with x-litellm-team-id={team!r} returned no completion: {response}"
    )
    rows: Final = client.proxy.poll_logs_for_request_id(response.id)
    assert rows, f"no spend log row for request {response.id} within the poll deadline"
    return rows[0].team_id


def _denial(client: OtherClient, token: str, team: str) -> str:
    result: Final = client.chat_as_team(token, team, _ping())
    assert isinstance(result, UnknownApiError) and result.status_code == 403, (
        f"x-litellm-team-id={team!r} names no team the caller is in, so it must be rejected with 403, got {result}"
    )
    assert team in result.body, f"the 403 must name the header value it rejected ({team!r}), got {result.body[:300]}"
    return result.body


class TestJwtTeamHeader:
    @pytest.mark.covers("other.auth.jwt.team_header_alias_binds_team")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_team_header_with_the_team_alias_binds_the_same_team_as_the_team_id(
        self, client: OtherClient, bound_team: BoundTeam
    ) -> None:
        token: Final = client.idp.access_token(bound_team.identity)
        assert bound_team.team_alias != bound_team.team_id

        by_id: Final = _team_of_request(client, token, bound_team.team_id)
        assert by_id == bound_team.team_id, (
            f"precondition: x-litellm-team-id with the team id must bind {bound_team.team_id!r}, got {by_id!r}"
        )

        by_alias: Final = _team_of_request(client, token, bound_team.team_alias)
        assert by_alias == bound_team.team_id, (
            f"x-litellm-team-id={bound_team.team_alias!r} must bind the same team as its id "
            f"{bound_team.team_id!r}, got {by_alias!r}"
        )

    @pytest.mark.covers("other.auth.jwt.team_model_alias_listed_and_routes")
    @pytest.mark.parametrize("anthropic", [False, True], ids=["openai_shape", "anthropic_shape"])
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_team_model_alias_is_listed_by_v1_models_under_the_same_token_that_routes_it(
        self, client: OtherClient, aliased_team: AliasedTeam, anthropic: bool
    ) -> None:
        token: Final = client.idp.access_token(aliased_team.identity)

        routed: Final = unwrap(client.proxy.chat(token, _ping(model=aliased_team.alias)))
        assert routed.choices, f"precondition: /chat/completions must route the team alias, got {routed}"

        listed: Final = tuple(entry.id for entry in unwrap(client.list_models_as(token, anthropic=anthropic)).data)
        assert aliased_team.alias in listed, (
            f"/v1/models must list team alias {aliased_team.alias!r} that the same token routes on "
            f"/chat/completions, got {listed}"
        )
        assert aliased_team.target in listed, f"the alias target {aliased_team.target!r} must stay listed, got {listed}"

    @pytest.mark.covers("other.auth.jwt.team_header_non_member_alias_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_team_header_with_the_alias_of_a_team_the_caller_is_not_in_is_rejected_like_an_unknown_value(
        self, client: OtherClient, resources: ResourceManager, bound_team: BoundTeam
    ) -> None:
        token: Final = client.idp.access_token(bound_team.identity)
        other_marker: Final = unique_marker()
        other_alias: Final = _team(client, resources, marker=other_marker, team_id=f"e2e-jwt-other-{other_marker}")
        unknown: Final = f"e2e-jwt-unknown-{unique_marker()}"

        for_other_alias: Final = _denial(client, token, other_alias)
        for_unknown: Final = _denial(client, token, unknown)
        assert for_other_alias.replace(other_alias, "<value>") == for_unknown.replace(unknown, "<value>"), (
            "a non-member alias and an unknown value must get the same denial body, so the response does not "
            f"reveal whether the team exists; got {for_other_alias[:300]!r} vs {for_unknown[:300]!r}"
        )


class _RsaJwk(BaseModel):
    kid: str
    kty: str
    n: str = ""
    e: str = ""


class _Jwks(BaseModel):
    keys: list[_RsaJwk]


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

Forgery = Literal["groups_swapped_under_genuine_signature", "alg_none", "hs256_keyed_with_published_rsa_key"]


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64url_json(value: dict[str, JsonValue]) -> str:
    return _b64url(json.dumps(value, separators=(",", ":")).encode())


def _segment(segment: str) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))


def _b64url_int(value: str) -> int:
    return int.from_bytes(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)), "big")


def _published_rsa_pem(client: OtherClient, kid: str) -> bytes:
    jwks: Final = unwrap(get_external(client.idp.jwks_url, response_type=_Jwks))
    key: Final = next(k for k in jwks.keys if k.kid == kid and k.kty == "RSA")
    public_key: Final = RSAPublicNumbers(_b64url_int(key.e), _b64url_int(key.n)).public_key()
    return public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)


def _forge(client: OtherClient, genuine: str, *, victim_team: str, forgery: Forgery) -> str:
    """Rewrite a genuine token's groups claim to a team its subject is not in."""
    header_segment, payload_segment, signature = genuine.split(".")
    header: Final = _segment(header_segment)
    payload: Final = _b64url_json({**_segment(payload_segment), "groups": [victim_team]})
    match forgery:
        case "groups_swapped_under_genuine_signature":
            return f"{header_segment}.{payload}.{signature}"
        case "alg_none":
            return f"{_b64url_json({**header, 'alg': 'none'})}.{payload}."
        case "hs256_keyed_with_published_rsa_key":
            signing_input: Final = f"{_b64url_json({**header, 'alg': 'HS256'})}.{payload}"
            secret: Final = _published_rsa_pem(client, str(header["kid"]))
            mac: Final = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
            return f"{signing_input}.{_b64url(mac)}"


def _real_team(client: OtherClient, resources: ResourceManager, *, team_id: str, models: list[str]) -> str:
    created: Final = client.proxy.create_team(
        TeamNewBody(team_alias=f"alias-{team_id}", team_id=team_id, models=models)
    )
    resources.defer(lambda: client.proxy.delete_team(created))
    return created


@dataclass(frozen=True, slots=True)
class TwoTeams:
    identity: Identity
    locked_team: str
    open_team: str


@pytest.fixture
def two_teams(client: OtherClient, resources: ResourceManager) -> TwoTeams:
    """An identity in two teams. The one that sorts first cannot reach the test
    model, so picking the first group instead of the first team with access would
    bind the wrong team."""
    marker: Final = unique_marker()
    locked: Final = f"e2e-jwt-team-{marker}-0-locked"
    opened: Final = f"e2e-jwt-team-{marker}-1-open"
    identity: Final = client.idp.provision_groups(marker=marker, groups=(locked, opened), defer=resources.defer)
    resources.defer(lambda: client.proxy.delete_user(identity.user_id))
    _ = _real_team(client, resources, team_id=locked, models=[f"e2e-jwt-unreachable-model-{marker}"])
    _ = _real_team(client, resources, team_id=opened, models=[CHEAP_OPENAI_MODEL])
    return TwoTeams(identity=identity, locked_team=locked, open_team=opened)


def _team_of(client: OtherClient, token: str) -> str | None:
    response: Final = unwrap(client.proxy.chat(token, _ping()))
    assert response.id is not None and response.choices, f"chat returned no completion: {response}"
    rows: Final = client.proxy.poll_logs_for_request_id(response.id)
    assert rows, f"no spend log row for request {response.id} within the poll deadline"
    return rows[0].team_id


def _block_team(client: OtherClient, team_id: str) -> None:
    unwrap(
        client.proxy.transport.post(
            "/team/block",
            headers=client.proxy.transport.master,
            json=TeamBlockBody(team_id=team_id),
            response_type=NoBody,
        )
    )


class TestJwtTeamAdversarial:
    @pytest.mark.covers("other.auth.jwt.forged_team_claim_denied")
    @pytest.mark.parametrize(
        "forgery", ["groups_swapped_under_genuine_signature", "alg_none", "hs256_keyed_with_published_rsa_key"]
    )
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_token_forged_to_claim_another_team_is_rejected(
        self, client: OtherClient, resources: ResourceManager, identity: Identity, forgery: Forgery
    ) -> None:
        victim: Final = _real_team(client, resources, team_id=f"e2e-jwt-victim-{unique_marker()}", models=[])
        genuine: Final = client.idp.access_token(identity)
        assert unwrap(client.proxy.chat(genuine, _ping())).choices, "precondition: the genuine token must be served"

        result: Final = client.proxy.chat(_forge(client, genuine, victim_team=victim, forgery=forgery), _ping())
        assert isinstance(result, UnauthorizedError), (
            f"a token forged ({forgery}) to claim team {victim!r} must be rejected with 401 before any team is "
            f"resolved, got {result}"
        )

    @pytest.mark.covers("other.auth.jwt.team_header_non_member_id_denied")
    @meta(Subject(domain=Domain.PROXY_AUTH, models=(CHEAP_OPENAI_MODEL,), mode=Mode.NONSTREAM))
    def test_team_header_with_the_id_of_a_team_the_caller_is_not_in_is_rejected_like_an_unknown_value(
        self, client: OtherClient, resources: ResourceManager, identity: Identity
    ) -> None:
        token: Final = client.idp.access_token(identity)
        other_team: Final = _real_team(client, resources, team_id=f"e2e-jwt-other-{unique_marker()}", models=[])
        unknown: Final = f"e2e-jwt-unknown-{unique_marker()}"

        for_other_team: Final = _denial(client, token, other_team)
        for_unknown: Final = _denial(client, token, unknown)
        assert for_other_team.replace(other_team, "<value>") == for_unknown.replace(unknown, "<value>"), (
            "a real team the caller is not in and an unknown value must get the same denial body, got "
            f"{for_other_team[:300]!r} vs {for_unknown[:300]!r}"
        )

    @pytest.mark.covers("other.auth.jwt.team_header_without_groups_denied")
    @meta(Subject(domain=Domain.PROXY_AUTH, models=(CHEAP_OPENAI_MODEL,), mode=Mode.NONSTREAM))
    def test_token_without_groups_cannot_select_a_team_by_header(
        self, client: OtherClient, resources: ResourceManager
    ) -> None:
        groupless: Final = client.idp.provision_user(
            marker=unique_marker(), groups=(), group_ids=(), defer=resources.defer
        )
        resources.defer(lambda: client.proxy.delete_user(groupless.user_id))
        team: Final = _real_team(client, resources, team_id=f"e2e-jwt-unclaimed-{unique_marker()}", models=[])

        _ = _denial(client, client.idp.access_token(groupless), team)

    @pytest.mark.covers("other.auth.jwt.multi_team_binds_team_with_model_access")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_token_in_two_teams_binds_the_team_that_can_reach_the_model(
        self, client: OtherClient, two_teams: TwoTeams
    ) -> None:
        bound: Final = _team_of(client, client.idp.access_token(two_teams.identity))
        assert bound == two_teams.open_team, (
            f"the request must be attributed to {two_teams.open_team!r}, the only team that can reach "
            f"{CHEAP_OPENAI_MODEL}, not the first group {two_teams.locked_team!r}; got {bound!r}"
        )

    @pytest.mark.covers("other.auth.jwt.team_header_cannot_widen_model_access")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_team_header_naming_a_team_without_model_access_is_denied_even_if_another_group_has_it(
        self, client: OtherClient, two_teams: TwoTeams
    ) -> None:
        token: Final = client.idp.access_token(two_teams.identity)
        assert unwrap(client.chat_as_team(token, two_teams.open_team, _ping())).choices, (
            f"precondition: {two_teams.open_team!r} must reach {CHEAP_OPENAI_MODEL}"
        )

        result: Final = client.chat_as_team(token, two_teams.locked_team, _ping())
        assert isinstance(result, UnknownApiError) and result.status_code == 403, (
            f"x-litellm-team-id={two_teams.locked_team!r} binds a team without {CHEAP_OPENAI_MODEL}, so the "
            f"request must be denied with 403 rather than served through the other group, got {result}"
        )
        assert CHEAP_OPENAI_MODEL in result.body, f"the denial must be about model access, got {result.body[:300]}"

    @pytest.mark.covers("other.auth.jwt.blocked_team_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_blocking_a_team_stops_tokens_that_were_already_served(
        self, client: OtherClient, identity: Identity
    ) -> None:
        token: Final = client.idp.access_token(identity)
        assert unwrap(client.proxy.chat(token, _ping())).choices, "precondition: the team must be served first"

        _block_team(client, identity.group)
        settle_propagation(time.monotonic())

        result: Final = client.proxy.chat(token, _ping())
        assert isinstance(result, UnauthorizedError), f"a blocked team's token must be denied with 401, got {result}"
        assert "blocked" in result.body.lower() and identity.group in result.body, (
            f"the denial must name the blocked team {identity.group!r}, got {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.jwt.deleted_team_denied")
    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_deleting_a_team_stops_tokens_that_were_already_served(
        self, client: OtherClient, resources: ResourceManager
    ) -> None:
        member: Final = _provision(client, resources, marker=unique_marker())
        team: Final = client.proxy.create_team(TeamNewBody(team_alias=f"alias-{member.group}", team_id=member.group))
        token: Final = client.idp.access_token(member)
        assert unwrap(client.proxy.chat(token, _ping())).choices, "precondition: the team must be served first"

        client.proxy.delete_team(team)
        settle_propagation(time.monotonic())

        result: Final = client.proxy.chat(token, _ping())
        assert isinstance(result, UnknownApiError) and result.status_code == 403, (
            f"a token whose only team was deleted must be rejected like an unknown team, got {result}"
        )
        assert member.group in result.body, (
            f"the 403 must name the deleted team {member.group!r}, got {result.body[:300]}"
        )
