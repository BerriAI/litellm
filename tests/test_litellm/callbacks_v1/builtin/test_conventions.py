"""The rules every port obeys, checked for every manifest entry. A new port gets these for free."""

import ast
import importlib
import subprocess
import sys
from collections.abc import Generator
from dataclasses import is_dataclass
from pathlib import Path
from typing import Final

import pytest

from litellm import callbacks_v1
from litellm.callbacks_v1.builtin import port as port_vocabulary
from litellm.callbacks_v1.builtin.manifest import PORTS, Entry, Gap, ledger
from litellm.callbacks_v1.builtin.port import InterceptorPort, SinkPort
from litellm.callbacks_v1.builtin.runtime import Sink
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_logger import CustomLogger
from tests.test_litellm.callbacks_v1.builtin.support import EXAMPLES, SUBSCRIBERS, FakeTransport, golden_call

# The legacy hooks a v1 sink can stand in for. A legacy twin overriding any other
# CustomLogger hook does something the v1 contract has no place for.
SINK_HOOKS: Final = frozenset(
    {
        "log_pre_api_call",
        "log_post_api_call",
        "log_success_event",
        "log_failure_event",
        "async_log_success_event",
        "async_log_failure_event",
        "async_log_pre_api_call",
    }
)
PUBLIC_HOOKS: Final = frozenset(
    name
    for name, value in vars(CustomLogger).items()
    if callable(value) and (name.startswith(("log_", "async_")) or name.endswith("_hook"))
)

# What a port may import. A port is values and pure methods, the shape a sandboxed custom
# callback is confined to; anything that can reach a socket, a thread, the environment or the
# rest of litellm belongs to `runtime.py`, which a port may not import at all. A new pure
# stdlib module is a reviewed line here.
PURE_MODULES: Final = frozenset(
    {
        "base64",
        "collections.abc",
        "dataclasses",
        "datetime",
        "hashlib",
        "json",
        "math",
        "re",
        "types",
        "typing",
        "typing_extensions",
        "urllib.parse",
        "litellm.callbacks_v1",
        "litellm.callbacks_v1.builtin.port",
    }
)
# What a port never carries. `runtime.py` names it and wraps it into a subscriber, and
# `manifest.py` holds everything the migration needs to say about it.
NOT_A_PORTS: Final = frozenset(
    {"NAME", "GAPS", "events", "on_event", "async_on_event", "before_send", "async_before_send"}
)

pytestmark = pytest.mark.parametrize("entry", PORTS, ids=[entry.name for entry in PORTS])


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None]:
    yield
    for subscriber in callbacks_v1.snapshot():
        callbacks_v1.unregister(subscriber)


def _legacy(entry: Entry) -> type:
    module, _, attribute = entry.legacy.partition(":")
    return getattr(importlib.import_module(module), attribute)


def test_importing_a_port_registers_nothing(entry: Entry) -> None:
    script: Final = (
        f"import {entry.module.__name__}; from litellm import callbacks_v1; "
        "import sys; sys.exit(len(callbacks_v1.snapshot()))"
    )
    assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 0


def _impure_imports(source: str) -> tuple[str, ...]:
    nodes: Final = tuple(ast.walk(ast.parse(source)))
    plain: Final = tuple(alias.name for node in nodes if isinstance(node, ast.Import) for alias in node.names)
    froms: Final = tuple(
        str(node.module)
        for node in nodes
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in PURE_MODULES)
    )
    return (*(name for name in plain if name not in PURE_MODULES), *froms)


def test_a_port_imports_nothing_that_can_act(entry: Entry) -> None:
    for module in (entry.module, port_vocabulary):
        source = Path(str(module.__file__)).read_text()  # rebind-ok: one source per checked module
        assert _impure_imports(source) == (), f"{module.__name__} imports outside the pure allow-list"


def test_a_port_is_a_frozen_class_over_a_frozen_config(entry: Entry) -> None:
    example: Final = EXAMPLES[entry.name]
    assert all(isinstance(gap, Gap) for gap in entry.gaps)
    assert is_dataclass(example) and type(example).__dataclass_params__.frozen
    assert is_dataclass(entry.module.Config) and entry.module.Config.__dataclass_params__.frozen
    assert "Legacy twin:" in (entry.module.__doc__ or "")


def test_a_port_declares_the_protocol_of_its_kind_and_nothing_of_the_contract(entry: Entry) -> None:
    """A port names its protocol as a base, so the type checker holds it to the shape at the
    class. This is the same claim at runtime, for a port whose module a type checker never saw."""
    declared: Final = type(EXAMPLES[entry.name]).__mro__
    expected: Final = SinkPort if entry.kind == "sink" else InterceptorPort
    assert expected in declared, f"{entry.name} is a {entry.kind}; it should declare {expected.__name__} as its base"
    assert (SinkPort in declared) == (entry.kind == "sink")

    example: Final = EXAMPLES[entry.name]
    carried: Final = {member for member in NOT_A_PORTS if hasattr(example, member)}
    assert not carried, f"{entry.name} carries {sorted(carried)}; that is runtime.py's or manifest.py's"


def test_a_port_registers_under_its_manifest_name_with_the_handlers_of_its_kind(entry: Entry) -> None:
    callbacks_v1.register(SUBSCRIBERS[entry.name](FakeTransport()))

    (subscriber,) = callbacks_v1.snapshot()
    assert subscriber.name == entry.name
    observes: Final = subscriber.on_event is not None or subscriber.async_on_event is not None
    intercepts: Final = subscriber.before_send is not None or subscriber.async_before_send is not None
    assert (observes, intercepts) == ((True, False) if entry.kind == "sink" else (False, True))
    # A sink's handler only enqueues, so the sync one serves async calls too.
    assert subscriber.async_on_event is None


def test_a_sink_handler_does_no_io_and_keeps_no_finished_call(entry: Entry) -> None:
    if entry.kind != "sink":
        pytest.skip("interceptors have no outbox")
    transport: Final = FakeTransport()
    callback: Final = SUBSCRIBERS[entry.name](transport)
    assert isinstance(callback, Sink), "a sink port supplies pure methods; `runtime.Sink` is its only subscriber"

    for envelope in golden_call("call.succeeded", user_api_key_user_id="user-1"):
        callback.on_event(envelope)

    assert callback.outbox.flush()
    assert len(transport.deliveries) == 1
    assert callback.open_calls() == 0


def test_the_legacy_twin_resolves_and_a_sink_twin_is_only_a_sink(entry: Entry) -> None:
    legacy: Final = _legacy(entry)
    if entry.kind != "sink":
        return
    overridden: Final = {
        name
        for cls in legacy.__mro__
        if cls not in (CustomLogger, CustomBatchLogger, object)
        for name in vars(cls)
        if name in PUBLIC_HOOKS
    }
    assert overridden <= SINK_HOOKS, f"{entry.legacy} also overrides {sorted(overridden - SINK_HOOKS)}"


def test_a_port_names_each_gap_once_by_what_the_contract_lacks(entry: Entry) -> None:
    keys: Final = tuple(gap.key for gap in entry.gaps)
    assert len(keys) == len(frozenset(keys)), f"{entry.name} lists a gap key twice: {sorted(keys)}"
    assert all(key and key == key.lower() and key.replace("_", "").isalnum() for key in keys)


def test_ready_means_no_open_gap(entry: Entry) -> None:
    assert entry.status != "ready" or entry.gaps == ()
    assert entry.name in ledger()
