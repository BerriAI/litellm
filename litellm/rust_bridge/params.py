"""Select SDK extension fields without copying or converting their values."""

from collections.abc import Mapping, Sequence
from typing import Final

from litellm.types.utils import all_litellm_params


def provider_param_names(kwargs: Mapping[str, object], consumed: Sequence[str]) -> tuple[str, ...]:
    sdk_fields: Final = frozenset(all_litellm_params) | {
        "model",
        "document",
        "timeout",
        "extra_headers",
        "custom_llm_provider",
        "input_sources",
    }
    consumed_fields: Final = frozenset(consumed)
    return tuple(name for name in kwargs if name in consumed_fields or name not in sdk_fields)
