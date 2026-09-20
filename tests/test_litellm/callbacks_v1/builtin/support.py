"""What every port's tests share: golden envelopes, a fake transport, and the parity diff.

The parity diff is the migration test. A legacy logger and its port handle the same call,
and every difference between what they send the vendor must be `Allowed` with a reason;
an unexplained one fails. An `Allowed` entry that no longer matches anything fails too,
and so does one citing a `Gap` its port no longer lists, so the allow-list and the gap
ledger shrink together as the contract grows instead of rotting apart.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Final

from litellm.callbacks_v1 import Callback, EnvelopeV1, EventName
from litellm.callbacks_v1.builtin import anthropic_cache_control, generic_api, openmeter
from litellm.callbacks_v1.builtin.manifest import Gap
from litellm.callbacks_v1.builtin.port import Delivery
from litellm.callbacks_v1.builtin.runtime import Patcher, Sink, Transport

ROOT: Final = Path(__file__).parents[4]
GOLDEN_PATH: Final = ROOT / "litellm-rust/crates/callbacks-v1/golden/v1"
FIXED_NOW: Final = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def golden(kind: EventName) -> EnvelopeV1:
    """The envelope Rust pins for `kind`; a port tested on it cannot drift from the contract.
    `tests/test_litellm/test_callbacks_v1.py` is what validates these files against the types."""
    return json.loads((GOLDEN_PATH / f"{kind}.json").read_text())


def golden_call(terminal: EventName, **metadata: str) -> tuple[EnvelopeV1, ...]:
    """One call's envelopes in order, ending in `terminal`, with `metadata` on call.started."""
    started: Final = golden("call.started")
    with_metadata: Final[EnvelopeV1] = {
        **started,
        "event": {**started["event"], "metadata": {**started["event"]["metadata"], **metadata}},  # pyright: ignore[reportTypedDictNotRequiredAccess, reportGeneralTypeIssues]  # the golden call.started event always has metadata
    }
    return (with_metadata, golden("request.sending"), golden("response.received"), golden(terminal))


@dataclass
class FakeTransport:
    deliveries: list[Delivery] = field(default_factory=list)

    def send(self, delivery: Delivery) -> None:
        self.deliveries.append(delivery)

    def bodies(self) -> list[object]:
        return [json.loads(delivery.body) for delivery in self.deliveries]


def plain(value: object) -> object:
    """A contract value as plain dicts and lists, so it compares and serialises like JSON."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [plain(item) for item in value]
    return value


# One configured port per manifest entry; `test_conventions` requires one for every entry.
# A port is pure and unnamed, so an example is a value: the host, not the port, is what
# needs a transport and what decides the registration name.
OPENMETER: Final = openmeter.OpenMeter(openmeter.Config(api_key="test-key", endpoint="http://openmeter.test"))
GENERIC_API: Final = generic_api.GenericApi(generic_api.Config(endpoint="http://generic.test/logs", batch_size=1))
ANTHROPIC_CACHE_CONTROL: Final = anthropic_cache_control.AnthropicCacheControl(
    anthropic_cache_control.Config(points=(anthropic_cache_control.InjectionPoint(role="system"),))
)
EXAMPLES: Final[Mapping[str, object]] = {
    "openmeter": OPENMETER,
    "generic_api": GENERIC_API,
    "anthropic_cache_control": ANTHROPIC_CACHE_CONTROL,
}

# The subscriber a host builds for each example port, under the manifest's name. Naming and
# wrapping are the host's job, which is why these live here and not in the ports; each call
# is also where the type checker holds a port to its protocol. The clock is fixed so a port
# that stamps a time is comparable run to run, and one that does not is handed it anyway:
# that is the point of the two kinds having one shape.
SUBSCRIBERS: Final[Mapping[str, Callable[[Transport], Callback]]] = {
    "openmeter": lambda transport: Sink("openmeter", OPENMETER, transport, lambda: FIXED_NOW),
    "generic_api": lambda transport: Sink("generic_api", GENERIC_API, transport, lambda: FIXED_NOW),
    "anthropic_cache_control": lambda _transport: Patcher("anthropic_cache_control", ANTHROPIC_CACHE_CONTROL),
}


@dataclass(frozen=True, slots=True)
class Allowed:
    """A difference from the legacy twin that is intended. `path` is an fnmatch pattern over
    dotted paths (`data.cost`, `0.metadata.*`; a list index is a segment); `reason` says why.

    `gap` is the `Gap.key` this difference exists because of. A difference with no `gap` is
    one the port intends to keep; a difference with one goes away when the contract grows."""

    path: str
    reason: str
    gap: str | None = None


def _leaves(value: object, prefix: str = "") -> Mapping[str, object]:
    if isinstance(value, Mapping) and value:
        return {
            path: leaf
            for key, item in value.items()
            for path, leaf in _leaves(item, f"{prefix}.{key}" if prefix else str(key)).items()
        }
    if isinstance(value, list) and value:
        return {
            path: leaf
            for index, item in enumerate(value)
            for path, leaf in _leaves(item, f"{prefix}.{index}" if prefix else str(index)).items()
        }
    return {prefix: value}


def unexplained(legacy: object, port: object, allowed: Sequence[Allowed], gaps: Sequence[Gap]) -> tuple[str, ...]:
    """Every difference between two vendor bodies that `allowed` does not cover, plus every
    `allowed` entry that covered nothing or cites a gap not in `gaps`. Empty means parity."""
    assert all(entry.reason.strip() for entry in allowed), "every Allowed difference needs a reason"
    open_keys: Final = {gap.key for gap in gaps}
    closed: Final = [entry for entry in allowed if entry.gap is not None and entry.gap not in open_keys]
    before: Final = _leaves(plain(legacy))
    after: Final = _leaves(plain(port))
    differences: Final = {
        **{path: f"missing in port: {path} (legacy sent {before[path]!r})" for path in before.keys() - after.keys()},
        **{path: f"only in port: {path} = {after[path]!r}" for path in after.keys() - before.keys()},
        **{
            path: f"differs: {path}: legacy {before[path]!r} != port {after[path]!r}"
            for path in before.keys() & after.keys()
            if before[path] != after[path]
        },
    }
    covered: Final = {path: [entry for entry in allowed if fnmatchcase(path, entry.path)] for path in differences}
    stale: Final = [entry for entry in allowed if not any(entry in entries for entries in covered.values())]
    return (
        *(message for path, message in sorted(differences.items()) if not covered[path]),
        *(f"stale Allowed({entry.path!r}): it no longer matches any difference" for entry in stale),
        *(f"Allowed({entry.path!r}) cites gap {entry.gap!r}, which the port does not list" for entry in closed),
    )


def assert_parity(legacy: bytes, port: bytes, allowed: Sequence[Allowed], gaps: Sequence[Gap]) -> None:
    """Compares two raw vendor request bodies as JSON and fails with one line per unexplained
    difference, so the fix is to port it or to allow it. Two empty bodies are not parity.
    `gaps` is the port's own `GAPS`, which every `Allowed.gap` has to be found in."""
    assert legacy and port, "nothing to compare: a side sent no body"
    problems: Final = unexplained(json.loads(legacy), json.loads(port), allowed, gaps)
    assert not problems, f"{len(problems)} unexplained difference(s) from the legacy twin:\n  " + "\n  ".join(problems)
