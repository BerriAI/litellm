# What this tests
## This tests the proxy server startup
import sys, os, json
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm
from litellm.proxy.proxy_server import (
    router,
    save_worker_config,
    initialize,
    proxy_startup_event,
    llm_model_list,
    proxy_shutdown_event,
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


@pytest.mark.asyncio
async def test_proxy_shutdown_stops_scheduler_before_prisma_disconnect(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server

    events = []

    class MockScheduler:
        def shutdown(self, wait=True):
            assert wait is True
            events.append("scheduler_shutdown")

    class MockPrismaClient:
        async def disconnect(self):
            events.append("prisma_disconnect")

    mock_jwt_handler = MagicMock()
    mock_jwt_handler.close = AsyncMock()

    monkeypatch.setattr(proxy_server, "scheduler", MockScheduler())
    monkeypatch.setattr(proxy_server, "spend_logs_queue_monitor_task", None)
    monkeypatch.setattr(proxy_server, "prisma_client", MockPrismaClient())
    monkeypatch.setattr(proxy_server, "jwt_handler", mock_jwt_handler)
    monkeypatch.setattr(proxy_server, "db_writer_client", None)
    monkeypatch.setattr(litellm, "cache", None)

    await proxy_server.proxy_shutdown_event()

    assert events == ["scheduler_shutdown", "prisma_disconnect"]
    assert proxy_server.scheduler is None
    mock_jwt_handler.close.assert_awaited_once()
