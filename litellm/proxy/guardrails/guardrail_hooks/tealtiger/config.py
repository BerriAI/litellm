from itertools import chain
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.types.proxy.guardrails.guardrail_hooks.tealtiger import (
    DEFAULT_POLICIES,
    PiiPolicy,
    Policy,
    PolicyMode,
)

from .patterns import unknown_pattern_names


class TealTigerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policies: tuple[Policy, ...] = DEFAULT_POLICIES
    policy_mode: PolicyMode = PolicyMode.ENFORCE


def parse_config(policies: object, policy_mode: object) -> TealTigerConfig:
    """Validate the raw `policies` / `policy_mode` litellm_params into typed config, failing at startup."""
    raw: Final = {
        key: value for key, value in (("policies", policies), ("policy_mode", policy_mode)) if value is not None
    }
    config: Final = TealTigerConfig.model_validate(raw)
    unknown: Final = tuple(
        chain.from_iterable(unknown_pattern_names(p.patterns) for p in config.policies if isinstance(p, PiiPolicy))
    )
    if unknown:
        raise ValueError(f"TealTiger: unknown pii pattern or category name(s): {', '.join(unknown)}")
    return config
