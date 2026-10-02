"""LiteLLM_JWTKeyMapping test doubles for the bulk key deletion paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JWTMappingRow:
    token: str
    jwt_claim_name: str
    jwt_claim_value: str
    jwt_issuer: str | None = None


class CascadingJWTMappingTable:
    """Mapping rows that LiteLLM_JWTKeyMapping_token_fkey drops when their key row is deleted."""

    def __init__(self, rows: Sequence[JWTMappingRow]) -> None:
        self.rows: tuple[JWTMappingRow, ...] = tuple(rows)

    async def find_many(self, where: Mapping[str, Mapping[str, Sequence[str]]]) -> list[JWTMappingRow]:
        return [row for row in self.rows if row.token in where["token"]["in"]]

    def cascade(self, deleted_tokens: Sequence[str]) -> None:
        self.rows = tuple(row for row in self.rows if row.token not in deleted_tokens)
