from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.proxy.auth.host.steps.jwt_auto_register import create_key_mapping
from litellm.proxy.utils import hash_token
from tests.unit.proxy.rust_auth_stores import JwtMapping, RustAuthStores

CLAIM: Final = ("email", "dev@example.com")


@dataclass
class _KeyTable:
    deleted: list[object] = field(default_factory=list)  # mutable-ok: records each delete

    async def delete(self, where: dict[str, object]) -> None:
        self.deleted.append(where)


def _prisma(keys: _KeyTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_verificationtoken=keys))


async def _map(keys: _KeyTable, existing_hash: str | None, issuer: str | None = None) -> tuple[str | None, bool]:
    return await create_key_mapping(
        _prisma(keys),  # pyright: ignore[reportArgumentType]  # serves only the key table this module touches
        *CLAIM,
        issuer=issuer,
        behavior="auto_register",
        existing_hash=existing_hash,
        team_id=None,
        user_id="u1",
        org_id=None,
        agent_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("issuer", "stored_issuer"), [(None, ""), ("https://idp", "https://idp")])
async def test_a_claim_is_mapped_to_the_key_it_was_given(
    rust_auth_stores: RustAuthStores, issuer: str | None, stored_issuer: str
) -> None:
    assert await _map(_KeyTable(), "hash-a", issuer) == ("hash-a", False)
    assert rust_auth_stores.jwt_key_mapping.mappings == [
        JwtMapping(*CLAIM, token="hash-a", jwt_issuer=stored_issuer, written_by="auto_register")
    ]


@pytest.mark.asyncio
async def test_a_request_that_loses_the_race_returns_the_winners_key_and_keeps_its_own(
    rust_auth_stores: RustAuthStores,
) -> None:
    keys: Final = _KeyTable()
    _ = await _map(keys, "hash-winner")

    assert await _map(keys, "hash-loser") == ("hash-winner", True)
    assert keys.deleted == []
    assert [mapping.token for mapping in rust_auth_stores.jwt_key_mapping.mappings] == ["hash-winner"]


@pytest.mark.asyncio
async def test_a_request_that_loses_the_race_deletes_the_key_it_minted(
    rust_auth_stores: RustAuthStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def mint(**_: object) -> dict[str, str]:
        return {"token": "sk-minted"}

    monkeypatch.setattr("litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn", mint)
    keys: Final = _KeyTable()
    _ = await _map(keys, "hash-winner")

    assert await _map(keys, None) == ("hash-winner", True)
    assert keys.deleted == [{"token": hash_token("sk-minted")}]


@pytest.mark.asyncio
async def test_a_claim_held_by_an_inactive_mapping_maps_to_no_key(rust_auth_stores: RustAuthStores) -> None:
    rust_auth_stores.jwt_key_mapping.mappings.append(JwtMapping(*CLAIM, token="hash-retired", is_active=False))

    assert await _map(_KeyTable(), "hash-a") == (None, True)


@pytest.mark.asyncio
async def test_a_database_error_propagates(rust_auth_stores: RustAuthStores) -> None:
    outage: Final = ConnectionError("database unreachable")
    rust_auth_stores.jwt_key_mapping.error = outage

    with pytest.raises(ConnectionError) as raised:
        _ = await _map(_KeyTable(), "hash-a")
    assert raised.value is outage
