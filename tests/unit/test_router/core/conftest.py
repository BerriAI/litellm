from collections.abc import Callable

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
    builder = CORE_BUILDERS[request.param]

    def build(**arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
        try:
            return builder(host, **arguments)
        except CoreDeclined as declined:
            pytest.skip(f"{request.param} core declines this configuration: {declined}")

    return build
