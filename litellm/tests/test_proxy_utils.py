import sys, os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

load_dotenv()
import os, io, time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import update_config
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import (
    ConfigYAML,
    ConfigGeneralSettings,
)
from litellm.proxy.utils import DBClient
from starlette.datastructures import URL
from litellm.caching import DualCache

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.mark.asyncio
async def test_config_update_master_key():
    # this should raise an Exception - we never want to store a master key in the DB
    try:
        config = ConfigYAML(
            general_settings=ConfigGeneralSettings(master_key="test_master_key")
        )
        await update_config(config_info=config)
        pytest.fail(
            "Should have raised an Exception, user tried storing master key in the DB"
        )
    except Exception as e:
        print("Got Exception", str(e))
        print(e.message)
        assert (
            "master_key cannot be stored in the DB, please store it on your machines env or use a secret manager"
            in e.message
        )
