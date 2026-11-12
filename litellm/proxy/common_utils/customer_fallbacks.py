from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.proxy.customer_fallbacks import CustomerFallbacks


@dataclass(frozen=True, slots=True)
class ResolvedRouterSettings:
    values: Mapping[str, object]
    authoritative_fields: frozenset[str] = frozenset()

    def client_settings(self, request_data: Mapping[str, object]) -> Mapping[str, object]:
        if not self.authoritative_fields:
            return request_data
        override: Final = as_str_mapping(request_data.get("router_settings_override"))
        return MappingProxyType(
            {
                **{key: value for key, value in request_data.items() if key not in self.authoritative_fields},
                **(
                    {
                        "router_settings_override": {
                            key: value for key, value in override.items() if key not in self.authoritative_fields
                        }
                    }
                    if override is not None
                    else {}
                ),
            }
        )


def resolve_router_settings(
    inherited: Mapping[str, object] | None,
    customer_fallbacks: CustomerFallbacks | None,
) -> ResolvedRouterSettings:
    inherited_settings: Final = inherited or MappingProxyType({})
    if customer_fallbacks is None:
        return ResolvedRouterSettings(inherited_settings)
    return ResolvedRouterSettings(
        MappingProxyType({**inherited_settings, "fallbacks": customer_fallbacks}),
        frozenset({"fallbacks"}),
    )
