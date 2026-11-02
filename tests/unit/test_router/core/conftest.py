from collections.abc import Callable, Generator
from pathlib import Path
from typing import Final

import pytest

from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import CORE_BUILDERS, CoreDeclined, ScriptedHost


@pytest.fixture
def host() -> ScriptedHost:
    return ScriptedHost()


@pytest.fixture(params=sorted(CORE_BUILDERS))
def make_core(request: pytest.FixtureRequest, host: ScriptedHost) -> Callable[..., RouterCore]:
    """``make_core(**router_arguments)``: the core under test, on the ``host`` fixture. Every test that
    uses it runs once per core in ``CORE_BUILDERS`` (``-k python`` or ``-k rust`` picks one)."""
    name: Final[str] = str(request.param)  # pyright: ignore[reportAny]  # the fixture parameter
    builder: Final = CORE_BUILDERS[name]

    def build(**arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
        try:
            return builder(host, **arguments)
        except CoreDeclined as declined:
            pytest.skip(f"{name} core declines this configuration: {declined}")

    return build


_RUST_PARITY_PENDING: Final = frozenset(
    line.strip()
    for line in (Path(__file__).parent / "rust_parity_pending.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Characterization tests the Rust core does not match yet are listed in
    ``rust_parity_pending.txt`` and expected to fail (strictly: one that starts passing must leave the list)."""
    here: Final = Path(__file__).parent
    for item in items:
        path: Final = Path(str(item.path))
        if not path.is_relative_to(here):
            continue  # the hook sees every collected item, not only this folder's
        name: Final = "::".join((path.relative_to(here).as_posix(), *item.nodeid.split("::")[1:]))
        if name in _RUST_PARITY_PENDING:
            item.add_marker(pytest.mark.xfail(reason="Rust core parity pending", strict=True))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, None]:
    """A test that reaches a member or request the Rust core does not serve yet is skipped with
    that member named, so the run separates "not served yet" from a wrong answer."""
    outcome: Final = yield
    excinfo: Final[object] = getattr(outcome, "excinfo", None)
    if not isinstance(excinfo, tuple):
        return
    info: Final[tuple[object, ...]] = excinfo  # pyright: ignore[reportUnknownVariableType]  # pluggy's (type, value, traceback)
    if len(info) < 2:
        return
    from litellm.router_rust_core import RustCoreUnsupported

    error: Final[object] = info[1]
    cause: Final[object] = getattr(error, "__cause__", None)
    force_exception: Final[object] = getattr(outcome, "force_exception", None)
    if (isinstance(error, RustCoreUnsupported) or isinstance(cause, RustCoreUnsupported)) and callable(force_exception):
        force_exception(pytest.skip.Exception(f"not served by the Rust core yet: {error}", _use_item_location=True))
