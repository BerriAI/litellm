# What this tests
## This tests the proxy server startup
import json
import os
import sys
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm.proxy.proxy_server import (
    proxy_startup_event,
)


@pytest.mark.asyncio
async def test_proxy_gunicorn_startup_direct_config():
    """
    gunicorn startup requires the config to be passed in via environment variables

    We support saving either the config or the dict as an environment variable.

    Test both approaches
    """
    try:
        from litellm._logging import verbose_proxy_logger, verbose_router_logger
        import logging

        # unset set DATABASE_URL in env for this test
        # set prisma client to None
        setattr(litellm.proxy.proxy_server, "prisma_client", None)
        database_url = os.environ.pop("DATABASE_URL", None)

        verbose_proxy_logger.setLevel(level=logging.DEBUG)
        verbose_router_logger.setLevel(level=logging.DEBUG)
        filepath = os.path.dirname(os.path.abspath(__file__))
        # test with worker_config = config yaml
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        os.environ["WORKER_CONFIG"] = config_fp
        async with proxy_startup_event(app=None) as _:
            pass
    except Exception as e:
        if "Already connected to the query engine" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")
    finally:
        # restore DATABASE_URL after the test
        if database_url is not None:
            os.environ["DATABASE_URL"] = database_url


@pytest.mark.asyncio
async def test_proxy_gunicorn_startup_config_dict():
    try:
        from litellm._logging import verbose_proxy_logger, verbose_router_logger
        import logging

        verbose_proxy_logger.setLevel(level=logging.DEBUG)
        verbose_router_logger.setLevel(level=logging.DEBUG)
        # unset set DATABASE_URL in env for this test
        # set prisma client to None
        setattr(litellm.proxy.proxy_server, "prisma_client", None)
        database_url = os.environ.pop("DATABASE_URL", None)

        filepath = os.path.dirname(os.path.abspath(__file__))
        # test with worker_config = config yaml
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        # test with worker_config = dict
        worker_config = {"config": config_fp}
        os.environ["WORKER_CONFIG"] = json.dumps(worker_config)
        async with proxy_startup_event(app=None) as _:
            pass
    except Exception as e:
        if "Already connected to the query engine" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")
    finally:
        # restore DATABASE_URL after the test
        if database_url is not None:
            os.environ["DATABASE_URL"] = database_url


# test_proxy_gunicorn_startup()


def test_proxy_cli_azure_postgresql_auth_sets_token_database_url(monkeypatch):
    from litellm.proxy.db.prisma_client import (
        AZURE_POSTGRESQL_AUTH_MARKER_ENV,
        IAMEndpoint,
    )
    from litellm.proxy.proxy_cli import run_server

    endpoint = IAMEndpoint(
        host="server.postgres.database.azure.com",
        port="5432",
        user="managed-identity",
        name="litellm",
    )
    token_url = (
        "postgresql://managed-identity:token@"
        "server.postgres.database.azure.com:5432/litellm"
    )
    mock_proxy_module = MagicMock(
        app=MagicMock(),
        ProxyConfig=MagicMock(),
        KeyManagementSettings=MagicMock(),
        save_worker_config=MagicMock(),
    )

    for key in (
        "DATABASE_URL",
        "DIRECT_URL",
        "IAM_TOKEN_DB_AUTH",
        AZURE_POSTGRESQL_AUTH_MARKER_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    with (
        patch.dict(
            "sys.modules",
            {
                "proxy_server": mock_proxy_module,
                "litellm.proxy.proxy_server": mock_proxy_module,
            },
        ),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("atexit.register"),
        patch(
            "litellm.proxy.db.prisma_client.get_database_auth_endpoint_from_env",
            return_value=endpoint,
        ) as mock_get_endpoint,
        patch(
            "litellm.proxy.db.prisma_client.build_database_token_auth_url",
            return_value=token_url,
        ) as mock_build_url,
        patch(
            "litellm.proxy.db.prisma_client.should_update_prisma_schema",
            return_value=False,
        ),
        patch("litellm.proxy.db.check_migration.check_prisma_schema_diff"),
    ):
        run_server.main(
            ["--local", "--skip_server_startup", "--azure_postgresql_auth"],
            standalone_mode=False,
        )

    assert os.environ["DATABASE_URL"].startswith(token_url)
    assert os.environ["IAM_TOKEN_DB_AUTH"] == "True"
    assert os.environ[AZURE_POSTGRESQL_AUTH_MARKER_ENV] == "True"
    mock_get_endpoint.assert_called_once()
    mock_build_url.assert_called_once_with(endpoint, azure_postgresql_auth=True)
