import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()
import copy
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Dict, List, Optional, Union

import pytest

import litellm
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/very-special-endpoint",
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",  # If you are Krrish, this is OpenAI Endpoint3 on our Railway endpoint :)
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

from litellm.router import CustomRoutingStrategyBase


class CustomRoutingStrategy(CustomRoutingStrategyBase):
    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Asynchronously retrieves the available deployment based on the given parameters.

        Args:
            model (str): The name of the model.
            messages (Optional[List[Dict[str, str]]], optional): The list of messages for a given request. Defaults to None.
            input (Optional[Union[str, List]], optional): The input for a given embedding request. Defaults to None.
            specific_deployment (Optional[bool], optional): Whether to retrieve a specific deployment. Defaults to False.
            request_kwargs (Optional[Dict], optional): Additional request keyword arguments. Defaults to None.

        Returns:
            Returns an element from litellm.router.model_list

        """
        print("In CUSTOM async get available deployment")
        model_list = router.model_list
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
        """
        Synchronously retrieves the available deployment based on the given parameters.

        Args:
            model (str): The name of the model.
            messages (Optional[List[Dict[str, str]]], optional): The list of messages for a given request. Defaults to None.
            input (Optional[Union[str, List]], optional): The input for a given embedding request. Defaults to None.
            specific_deployment (Optional[bool], optional): Whether to retrieve a specific deployment. Defaults to False.
            request_kwargs (Optional[Dict], optional): Additional request keyword arguments. Defaults to None.

        Returns:
            Returns an element from litellm.router.model_list

        """
        pass


@pytest.mark.asyncio
async def test_custom_routing():
    import litellm

    litellm.set_verbose = True
    router.set_custom_routing_strategy(CustomRoutingStrategy())

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
    """
    Note: for debugging
    - By this point: slow-endpoint should have timed out 3-4 times and should be heavily penalized :)
    - The next 10 requests should all be routed to the fast-endpoint
    """

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

    # ALL the Requests should have been routed to the fast-endpoint
    # assert deployments["fast-endpoint"] == 10
