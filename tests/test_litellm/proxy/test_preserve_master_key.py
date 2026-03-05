"""
Regression test for #22330:
A master_key set by initialize() (via config file) must be preserved when
LITELLM_MASTER_KEY is not set as an environment variable.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy import proxy_server


@pytest.mark.asyncio
async def test_master_key_preserved_when_env_var_absent():
    """proxy_startup_event must NOT overwrite a config-provided master_key."""
    config_key = "sk-from-config-file-1234"

    # Simulate initialize() having set master_key from a config file
    original_master_key = proxy_server.master_key
    proxy_server.master_key = config_key

    try:
        with patch(
            "litellm.proxy.proxy_server.get_secret_str", return_value=None
        ), patch(
            "litellm.proxy.proxy_server.get_secret", return_value=None
        ), patch(
            "litellm.proxy.proxy_server.init_verbose_loggers"
        ), patch(
            "litellm.proxy.proxy_server._license_check"
        ):
            # proxy_startup_event is an @asynccontextmanager, so we must
            # enter it with `async with` — a bare `await` would only create
            # the generator object without executing the function body.
            try:
                async with proxy_server.proxy_startup_event(app=AsyncMock()):
                    pass
            except Exception:
                # Startup will fail on DB/router setup — we only care about
                # the master_key guard at the top of the function.
                pass

        assert proxy_server.master_key == config_key, (
            f"master_key was overwritten: expected {config_key!r}, "
            f"got {proxy_server.master_key!r}"
        )
    finally:
        proxy_server.master_key = original_master_key
