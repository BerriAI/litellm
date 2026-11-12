from dataclasses import fields

from litellm.router_rust.surface.fallbacks import MOCK_FLAGS
from litellm.types.router import MockRouterTestingParams


def test_mock_flags_cover_every_mock_testing_param():
    """The Rust core serves exactly the ``mock_testing_*`` flags the Python core reads: the
    ``MockRouterTestingParams`` fields and ``mock_testing_rate_limit_error``. A new flag fails
    here until ``MOCK_FLAGS`` serves it."""
    assert set(MOCK_FLAGS) == {f.name for f in fields(MockRouterTestingParams)} | {"mock_testing_rate_limit_error"}
