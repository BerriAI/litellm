"""LiteLLM_JWTKeyMapping test doubles for the bulk key deletion paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tests.unit.proxy.rust_auth_stores import FakeJwtKeyMappingStore, JwtMapping


@dataclass(frozen=True, slots=True)
class JWTMappingRow:
    token: str
    jwt_claim_name: str
    jwt_claim_value: str
    jwt_issuer: str | None = None


class CascadingJWTMappingTable:
    """Mapping rows that LiteLLM_JWTKeyMapping_token_fkey drops when their key row is deleted.

    The rows live in the fake Rust JWT key mapping store, which serves the cache-key reads; ``find_many``
    serves the Prisma reads that still run inside a delete transaction.
    """

    def __init__(self, store: FakeJwtKeyMappingStore, rows: Sequence[JWTMappingRow]) -> None:
        self._store = store
        store.mappings.extend(
            JwtMapping(
                token=row.token,
                jwt_claim_name=row.jwt_claim_name,
                jwt_claim_value=row.jwt_claim_value,
                jwt_issuer=row.jwt_issuer or "",
            )
            for row in rows
        )

    @property
    def rows(self) -> tuple[JWTMappingRow, ...]:
        return tuple(
            JWTMappingRow(m.token, m.jwt_claim_name, m.jwt_claim_value, m.jwt_issuer or None)
            for m in self._store.mappings
        )

    async def find_many(self, where: Mapping[str, Mapping[str, Sequence[str]]]) -> list[JWTMappingRow]:
        return [row for row in self.rows if row.token in where["token"]["in"]]

    def cascade(self, deleted_tokens: Sequence[str]) -> None:
        self._store.mappings[:] = [m for m in self._store.mappings if m.token not in deleted_tokens]
