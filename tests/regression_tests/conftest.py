"""Pytest configuration for regression tests."""

import pytest


def pytest_addoption(parser):
    """Add --update-baseline option."""
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Update performance baselines instead of comparing",
    )


@pytest.fixture
def update_baseline(request):
    """Check if we should update baselines."""
    return request.config.getoption("--update-baseline")


@pytest.fixture
def baselines():
    """Load existing baselines from JSON file."""
    import json
    from pathlib import Path
    
    baseline_file = Path(__file__).parent / "performance_baselines.json"
    if not baseline_file.exists():
        return {}
    
    with open(baseline_file, "r") as f:
        return json.load(f)

