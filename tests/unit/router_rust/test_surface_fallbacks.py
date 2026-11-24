from dataclasses import fields
from typing import Final

import pytest

from litellm.constants import SEMANTIC_CACHE_POLICY_KEY
from litellm.router_rust.surface.fallbacks import MOCK_FLAGS
from litellm.types.router import MockRouterTestingParams
from litellm.types.semantic_cache import SemanticCacheParams, SemanticCachePolicy
from tests.unit.test_router.core.fallbacks_helpers import build, complete
from tests.unit.test_router.core.scripted_host import ScriptedHost, rate_limit, rust_core


def test_mock_flags_cover_every_mock_testing_param():
    """The Rust core serves exactly the ``mock_testing_*`` flags the Python core reads: the
    ``MockRouterTestingParams`` fields and ``mock_testing_rate_limit_error``. A new flag fails
    here until ``MOCK_FLAGS`` serves it."""
    assert set(MOCK_FLAGS) == {f.name for f in fields(MockRouterTestingParams)} | {"mock_testing_rate_limit_error"}


@pytest.mark.asyncio
async def test_fallback_cache_controls_keep_the_authenticated_policy() -> None:
    host: Final = ScriptedHost()
    core: Final = build(lambda **kwargs: rust_core(host, **kwargs))
    host.answer("openai/a", rate_limit())
    policy: Final = SemanticCachePolicy(scope="key", params=SemanticCacheParams(enabled=False))

    await complete(
        core,
        cache={"ttl": 30, SEMANTIC_CACHE_POLICY_KEY: policy},
        fallbacks=[{"model": "b", "cache": {"ttl": 60}}],
    )

    primary, fallback = host.calls_to("acompletion")
    assert primary.kwargs["cache"] == {"ttl": 30, SEMANTIC_CACHE_POLICY_KEY: policy}
    assert fallback.kwargs["cache"] == {"ttl": 60, SEMANTIC_CACHE_POLICY_KEY: policy}
