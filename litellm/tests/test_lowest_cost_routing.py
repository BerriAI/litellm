#### What this tests ####
#    This tests the router's ability to pick deployment with lowest latency

import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, copy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import Router
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.caching import DualCache

### UNIT TESTS FOR LATENCY ROUTING ###


def test_get_available_deployments():
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "openai-gpt-4"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "groq/llama3-8b-8192"},
            "model_info": {"id": "groq-llama"},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    model_group = "gpt-3.5-turbo"

    ## CHECK WHAT'S SELECTED ##
    selected_model = lowest_cost_logger.get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print("selected model: ", selected_model)

    assert selected_model["model_info"]["id"] == "groq-llama"


async def _deploy(lowest_latency_logger, deployment_id, tokens_used, duration):
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/chatgpt-v-2",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": tokens_used}}
    time.sleep(duration)
    end_time = time.time()
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
