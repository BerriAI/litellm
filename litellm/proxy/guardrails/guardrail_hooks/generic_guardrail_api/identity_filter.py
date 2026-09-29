from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import Self

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.config_parsing import config_values
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIMetadata,
)

IdentitySkipOption: TypeAlias = Literal["skip_if_key_alias_in", "skip_if_team_id_in"]


def _is_listed(value: object, listed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in listed


@dataclass(frozen=True, slots=True)
class IdentitySkipFilter:
    key_aliases: frozenset[str] = frozenset()
    team_ids: frozenset[str] = frozenset()

    @classmethod
    def from_config(
        cls,
        skip_if_key_alias_in: Sequence[str] | None,
        skip_if_team_id_in: Sequence[str] | None,
    ) -> Self:
        return cls(
            key_aliases=frozenset(config_values(skip_if_key_alias_in, option_name="skip_if_key_alias_in")),
            team_ids=frozenset(config_values(skip_if_team_id_in, option_name="skip_if_team_id_in")),
        )

    def matched_option(self, metadata: GenericGuardrailAPIMetadata) -> IdentitySkipOption | None:
        if _is_listed(metadata.get("user_api_key_alias"), self.key_aliases):
            return "skip_if_key_alias_in"
        if _is_listed(metadata.get("user_api_key_team_id"), self.team_ids):
            return "skip_if_team_id_in"
        return None
