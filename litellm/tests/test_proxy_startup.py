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
    startup_event,
    llm_model_list
)

def test_proxy_gunicorn_startup():
    """
    gunicorn startup requires the config to be passed in via environment variables

    We support saving either the config or the dict as an environment variable. 

    Test both approaches
    """

    filepath = os.path.dirname(os.path.abspath(__file__))
    # test with worker_config = config yaml 
    config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
    os.environ["WORKER_CONFIG"] = config_fp
    asyncio.run(startup_event())
    # test with worker_config = dict 
    worker_config = {"config": config_fp}
    os.environ["WORKER_CONFIG"] = json.dumps(worker_config)
    asyncio.run(startup_event())


# test_proxy_gunicorn_startup()