"""The concurrency-ceiling gauge must never claim a limit nothing enforces."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.common_utils.request_pressure_metrics import (
    UNBOUNDED,
    effective_global_limit,
    is_global_limit_enforced,
    publish_global_max_parallel_requests,
)


def _with_limiter(handler):
    from litellm.proxy import hooks

    return patch.dict(hooks.PROXY_HOOKS, {"parallel_request_limiter": handler})


def test_the_v1_limiter_is_the_one_that_enforces_the_global_limit():
    from litellm.proxy.hooks.parallel_request_limiter import (
        _PROXY_MaxParallelRequestsHandler,
    )

    with _with_limiter(_PROXY_MaxParallelRequestsHandler):
        assert is_global_limit_enforced() is True


def test_the_default_v3_limiter_does_not_enforce_the_global_limit():
    """Verified live: with global_max_parallel_requests=3 and 20 concurrent
    requests, the v3 limiter returned 20x 200 and zero rejections. Tracked as
    LIT-5460."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,
    )

    with _with_limiter(_PROXY_MaxParallelRequestsHandler_v3):
        assert is_global_limit_enforced() is False


def test_an_enforced_limit_is_reported_as_the_configured_number():
    from litellm.proxy.hooks.parallel_request_limiter import (
        _PROXY_MaxParallelRequestsHandler,
    )

    logger = MagicMock()
    with (
        _with_limiter(_PROXY_MaxParallelRequestsHandler),
        patch("litellm.integrations.prometheus.PrometheusLogger.get_instance", return_value=logger),
    ):
        publish_global_max_parallel_requests(3)

    logger.set_global_max_parallel_requests_limit.assert_called_once_with(3.0)


@pytest.mark.parametrize("configured", [3, None])
def test_an_unenforced_limit_is_reported_as_unbounded(configured):
    """A registered gauge always exposes a value, so declining to set it renders
    as 0, which reads as "no requests allowed". Reporting +Inf says what is
    actually true: nothing bounds concurrency."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,
    )

    with _with_limiter(_PROXY_MaxParallelRequestsHandler_v3):
        assert effective_global_limit(configured) == UNBOUNDED


def test_no_configured_limit_is_unbounded_even_when_enforcement_is_on():
    from litellm.proxy.hooks.parallel_request_limiter import (
        _PROXY_MaxParallelRequestsHandler,
    )

    with _with_limiter(_PROXY_MaxParallelRequestsHandler):
        assert effective_global_limit(None) == UNBOUNDED


def test_the_gauge_is_never_left_at_a_bare_zero():
    """0 would be indistinguishable from a real ceiling of zero."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,
    )

    with _with_limiter(_PROXY_MaxParallelRequestsHandler_v3):
        assert effective_global_limit(3) != 0.0
        assert effective_global_limit(None) != 0.0


def test_publishing_never_blocks_startup():
    from litellm.proxy.hooks.parallel_request_limiter import (
        _PROXY_MaxParallelRequestsHandler,
    )

    with (
        _with_limiter(_PROXY_MaxParallelRequestsHandler),
        patch(
            "litellm.integrations.prometheus.PrometheusLogger.get_instance",
            side_effect=RuntimeError("metrics down"),
        ),
    ):
        publish_global_max_parallel_requests(3)
