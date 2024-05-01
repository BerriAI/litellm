import warnings
import pytest

def test_namespace_conflict_warning():
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")  # Capture all warnings
        import litellm

    # Check that no warning with the specific message was raised
    assert not any("conflict with protected namespace" in str(w.message) for w in recorded_warnings), "Test failed: 'conflict with protected namespace' warning was encountered!"
