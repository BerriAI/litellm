import sys
import os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
import pytest


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ]


def test_validate_fallbacks(model_list):
    router = Router(model_list=model_list, fallbacks=[{"gpt-4o": "gpt-3.5-turbo"}])
    router.validate_fallbacks(fallback_param=[{"gpt-4o": "gpt-3.5-turbo"}])


def test_routing_strategy_init(model_list):
    """Test if all routing strategies are initialized correctly"""
    from litellm.types.router import RoutingStrategy

    router = Router(model_list=model_list)
    for strategy in RoutingStrategy._member_names_:
        router.routing_strategy_init(
            routing_strategy=strategy, routing_strategy_args={}
        )


def test_print_deployment(model_list):
    """Test if the api key is masked correctly"""

    router = Router(model_list=model_list)
    deployment = {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    }
    printed_deployment = router.print_deployment(deployment)
    assert 10 * "*" in printed_deployment["litellm_params"]["api_key"]
