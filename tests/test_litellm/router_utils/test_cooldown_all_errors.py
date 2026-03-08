"""Tests that all error status codes trigger cooldown."""
import pytest
from unittest.mock import MagicMock

from litellm.router_utils.cooldown_handlers import _is_cooldown_required


@pytest.mark.parametrize(
    "status_code",
    [400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503],
    ids=lambda s: f"status_{s}",
)
def test_all_errors_require_cooldown(status_code):
    """Every error status code should require cooldown."""
    router = MagicMock()
    result = _is_cooldown_required(
        litellm_router_instance=router,
        model_id="test-model-id",
        exception_status=status_code,
        exception_str="Test error",
    )
    assert result is True, f"Status {status_code} should require cooldown"


def test_api_connection_error_still_skips_cooldown():
    """APIConnectionError should still skip cooldown (transient network issue)."""
    router = MagicMock()
    result = _is_cooldown_required(
        litellm_router_instance=router,
        model_id="test-model-id",
        exception_status=500,
        exception_str="APIConnectionError: Connection refused",
    )
    assert result is False
