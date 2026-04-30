"""
Test that max_budget from environment variable (string) is correctly
converted to float.
GitHub Issues: #23843, #26696
"""

import os
from unittest.mock import patch

import pytest

import litellm


@pytest.mark.asyncio
async def test_max_budget_string_converted_to_float():
    """
    When max_budget is set via os.environ/MAX_BUDGET, it arrives as a
    string. initialize() should convert it to float so the comparison
    `litellm.max_budget > 0` doesn't raise TypeError.
    """
    with (
        patch("litellm.proxy.common_utils.banner.show_banner"),
        patch("litellm.proxy.proxy_server.generate_feedback_box"),
    ):
        from litellm.proxy.proxy_server import initialize

        original = litellm.max_budget
        try:
            await initialize(max_budget="100.5")
            assert isinstance(litellm.max_budget, float)
            assert litellm.max_budget == 100.5
        finally:
            litellm.max_budget = original


@pytest.mark.asyncio
async def test_max_budget_float_stays_float():
    """max_budget as float should still work."""
    with (
        patch("litellm.proxy.common_utils.banner.show_banner"),
        patch("litellm.proxy.proxy_server.generate_feedback_box"),
    ):
        from litellm.proxy.proxy_server import initialize

        original = litellm.max_budget
        try:
            await initialize(max_budget=200.0)
            assert isinstance(litellm.max_budget, float)
            assert litellm.max_budget == 200.0
        finally:
            litellm.max_budget = original


@pytest.mark.asyncio
async def test_max_budget_from_config_yaml_env_var(tmp_path):
    """
    Regression test for GitHub issue #26696.

    When `litellm_settings.max_budget` is set via `os.environ/MAX_BUDGET`
    in config.yaml, ProxyConfig resolves the env var to a string, then
    applies the litellm_settings dict. Without coercion, `litellm.max_budget`
    becomes a str and `litellm.max_budget > 0` raises TypeError at startup
    (proxy_server.py around the `prisma_client is not None and
    litellm.max_budget > 0` check).
    """
    from litellm.proxy.proxy_server import ProxyConfig

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_list: []\n"
        "litellm_settings:\n"
        "  max_budget: os.environ/TEST_MAX_BUDGET_REGRESSION\n"
    )

    original_budget = litellm.max_budget
    original_env = os.environ.get("TEST_MAX_BUDGET_REGRESSION")
    os.environ["TEST_MAX_BUDGET_REGRESSION"] = "10"
    try:
        proxy_config = ProxyConfig()
        await proxy_config.load_config(router=None, config_file_path=str(config_path))

        assert isinstance(litellm.max_budget, float), (
            f"max_budget should be float after config load, got "
            f"{type(litellm.max_budget).__name__}"
        )
        assert litellm.max_budget == 10.0
        # The original failure mode: this comparison must not raise.
        assert litellm.max_budget > 0
    finally:
        litellm.max_budget = original_budget
        if original_env is None:
            os.environ.pop("TEST_MAX_BUDGET_REGRESSION", None)
        else:
            os.environ["TEST_MAX_BUDGET_REGRESSION"] = original_env
