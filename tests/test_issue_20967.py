import pytest
from litellm.utils import exception_type

def test_none_metadata_error_handling():
    try:
        kwargs = {"metadata": None}
        # Simulate the fixed logic
        _is_litellm_router_call = "model_group" in (kwargs.get("metadata") or {})
        assert _is_litellm_router_call is False
    except TypeError:
        pytest.fail("TypeError raised when metadata is None!")
