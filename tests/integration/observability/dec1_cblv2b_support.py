"""Shared helpers for the dec1 cblv2b contract tests (second wave on CustomLoggerV2 and the Rust callback ports).

Every rig is the round 1 `cblv2_rig` (scripted provider, scripted S3 peer, probe sink; `dec1_cblv2_support`) with
`litellm_settings.callbacks` naming instances from `dec1_cblv2b_probe`. Nothing here edits the round 1 modules;
they are imported as they are.
"""

from __future__ import annotations

import uuid
from typing import Final

from integration._support.client import Scenario, held
from integration.observability.dec1_cblv2_support import Rig, key_hash
from integration.observability.dec1_cblv2b_probe import PROBE_MODULE
from pydantic import JsonValue

LICENSE_SKIP: Final = "dynamic callback disabling is honoured only with LITELLM_LICENSE set"
DISABLE_HEADER: Final = "x-litellm-disable-callbacks"


def instances(*names: str) -> list[JsonValue]:
    """`litellm_settings.callbacks` entries for probe instances, in the given (config) order."""
    return [f"{PROBE_MODULE}.{name}" for name in names]


def new_key(scenario: Scenario, **fields: JsonValue) -> tuple[str, str, str]:
    """A fresh virtual key with a unique alias: (key, alias, sha256 hash)."""
    alias: Final = f"cblv2b-{uuid.uuid4().hex[:12]}"
    key: Final = scenario.key(key_alias=alias, **fields)
    return key, alias, key_hash(key)


def records(
    rig: Rig, text: str, logger: str, hook: str, count: int, *, holding: float = 3
) -> tuple[dict[str, JsonValue], ...]:
    """The probe records of one logger and hook for the request carrying `text`, once exactly `count` arrived and no
    more arrive for `holding` seconds (so a count of 0 proves absence)."""
    return held(
        lambda: tuple(record for record in rig.mentioning(text, logger) if record["hook"] == hook),
        lambda found: len(found) == count,
        holding=holding,
        seconds=30,
    )


def loggers_called(rig: Rig, text: str, hook: str) -> tuple[str, ...]:
    """Names of the loggers that ran `hook` for the request carrying `text`, in the order the proxy called them."""
    return tuple(logger for logger, called in rig.calls(text) if called == hook)
