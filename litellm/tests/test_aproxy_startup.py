# What this tests
## This tests the proxy server startup
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

import pytest

import litellm
from litellm.proxy.proxy_server import (
    initialize,
    llm_model_list,
    router,
    save_worker_config,
    shutdown_event,
    startup_event,
)


def test_proxy_gunicorn_startup_direct_config():
    """
    gunicorn startup requires the config to be passed in via environment variables

    We support saving either the config or the dict as an environment variable.

    Test both approaches
    """
    try:
        import logging

        from litellm._logging import verbose_proxy_logger, verbose_router_logger

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
        asyncio.run(startup_event())
        asyncio.run(shutdown_event())
    except Exception as e:
        if "Already connected to the query engine" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")
    finally:
        # restore DATABASE_URL after the test
        if database_url is not None:
            os.environ["DATABASE_URL"] = database_url


def test_proxy_gunicorn_startup_config_dict():
    try:
        import logging

        from litellm._logging import verbose_proxy_logger, verbose_router_logger

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
        asyncio.run(startup_event())
        asyncio.run(shutdown_event())
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
