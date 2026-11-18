from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.proxy.auth.host.steps.jwt_auto_register import create_key_mapping
from tests.unit.proxy.rust_auth_stores import FakeJwtKeyMappingStore, JwtMapping, RustAuthStores

CLAIM: Final = ("email", "dev@example.com")


@dataclass
class _MappingTable:
    """The `LiteLLM_JWTKeyMapping` table, unique on issuer, claim name and claim value like the schema.

    Its rows are what the Rust JWT key mapping store reads back, so each created row is mirrored there."""

    store: FakeJwtKeyMappingStore = field(default_factory=FakeJwtKeyMappingStore)
    rows: list[dict[str, object]] = field(default_factory=list)  # mutable-ok: the table's stored rows
    failure: Exception | None = None

    async def create(self, data: dict[str, object]) -> None:
        if self.failure is not None:
            raise self.failure
        scope: Final = (data["jwt_issuer"], data["jwt_claim_name"], data["jwt_claim_value"])
        if any((row["jwt_issuer"], row["jwt_claim_name"], row["jwt_claim_value"]) == scope for row in self.rows):
            raise Exception("Unique constraint failed on the fields: (`jwt_issuer`,`jwt_claim_name`,`jwt_claim_value`)")
        self.rows.append({**data, "is_active": True})
        self.store.mappings.append(
            JwtMapping(
                jwt_claim_name=str(data["jwt_claim_name"]),
                jwt_claim_value=str(data["jwt_claim_value"]),
                token=str(data["token"]),
                jwt_issuer=str(data["jwt_issuer"]),
            )
        )


@dataclass
class _KeyTable:
    deleted: list[object] = field(default_factory=list)  # mutable-ok: records each delete

    async def delete(self, where: dict[str, object]) -> None:
        self.deleted.append(where)


def _prisma(mappings: _MappingTable, keys: _KeyTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_jwtkeymapping=mappings, litellm_verificationtoken=keys))


async def _map(prisma: SimpleNamespace, existing_hash: str) -> tuple[str | None, bool]:
    return await create_key_mapping(
        prisma,  # pyright: ignore[reportArgumentType]  # serves only the two tables this module touches
        *CLAIM,
        issuer=None,
        behavior="auto_register",
        existing_hash=existing_hash,
        team_id=None,
        user_id="u1",
        org_id=None,
        agent_id=None,
    )


@pytest.mark.asyncio
async def test_a_claim_is_mapped_to_the_key_it_was_given() -> None:
    mappings: Final = _MappingTable()

    assert await _map(_prisma(mappings, _KeyTable()), "hash-a") == ("hash-a", False)
    assert [(row["jwt_issuer"], row["token"], row["created_by"]) for row in mappings.rows] == [
        ("", "hash-a", "auto_register")
    ]


@pytest.mark.asyncio
async def test_a_request_that_loses_the_race_returns_the_winners_key_and_keeps_its_own(
    rust_auth_stores: RustAuthStores,
) -> None:
    mappings: Final = _MappingTable(store=rust_auth_stores.jwt_key_mapping)
    keys: Final = _KeyTable()
    _ = await _map(_prisma(mappings, keys), "hash-winner")

    assert await _map(_prisma(mappings, keys), "hash-loser") == ("hash-winner", True)
    assert keys.deleted == []
    assert [row["token"] for row in mappings.rows] == ["hash-winner"]
    assert rust_auth_stores.jwt_key_mapping.lookups == [(*CLAIM, None)]


@pytest.mark.asyncio
async def test_a_database_error_other_than_the_unique_conflict_propagates() -> None:
    outage: Final = ConnectionError("database unreachable")

    with pytest.raises(ConnectionError) as raised:
        _ = await _map(_prisma(_MappingTable(failure=outage), _KeyTable()), "hash-a")
    assert raised.value is outage
