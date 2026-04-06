import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Dict, List, Optional, Union

import pytest

import litellm
from litellm import Router
from litellm.router import CustomRoutingStrategyBase


def _create_router():
    return Router(
        model_list=[
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/very-special-endpoint",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "very-special-endpoint"},
            },
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/fast-endpoint",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "fast-endpoint"},
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )


class CustomRoutingStrategy(CustomRoutingStrategyBase):
    def __init__(self, router_instance: Router):
        self._router = router_instance

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        print("In CUSTOM async get available deployment")
        model_list = self._router.model_list
        print("router model list=", model_list)
        for model in model_list:
            if isinstance(model, dict):
                if model["litellm_params"]["model"] == "openai/very-special-endpoint":
                    return model
        pass

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        pass


@pytest.mark.asyncio
async def test_custom_routing():
    litellm.set_verbose = True

    router = _create_router()
    router.set_custom_routing_strategy(CustomRoutingStrategy(router))

    # make 4 requests
    for _ in range(4):
        try:
            response = await router.acompletion(
                model="azure-model", messages=[{"role": "user", "content": "hello"}]
            )
            print(response)
        except Exception as e:
            print("got exception", e)

    await asyncio.sleep(1)
    print("done sending initial requests to collect latency")

    deployments = {}
    # make 10 requests
    for _ in range(10):
        response = await router.acompletion(
            model="azure-model", messages=[{"role": "user", "content": "hello"}]
        )
        print(response)
        _picked_model_id = response._hidden_params["model_id"]
        if _picked_model_id not in deployments:
            deployments[_picked_model_id] = 1
        else:
            deployments[_picked_model_id] += 1
    print("deployments", deployments)
