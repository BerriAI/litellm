# What is this?
## Tests if proxy/auth/auth_utils.py works as expected

import sys, os, asyncio, time, random, uuid
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.auth.auth_utils import (
    _allow_model_level_clientside_configurable_parameters,
)
from litellm.router import Router


@pytest.mark.parametrize(
    "allowed_param, input_value, should_return_true",
    [
        ("api_base", {"api_base": "http://dummy.com"}, True),
        (
            {"api_base": "https://api.openai.com/v1"},
            {"api_base": "https://api.openai.com/v1"},
            True,
        ),  # should return True
        (
            {"api_base": "https://api.openai.com/v1"},
            {"api_base": "https://api.anthropic.com/v1"},
            False,
        ),  # should return False
        (
            {"api_base": "^https://litellm.*direct\.fireworks\.ai/v1$"},
            {"api_base": "https://litellm-dev.direct.fireworks.ai/v1"},
            True,
        ),
        (
            {"api_base": "^https://litellm.*novice\.fireworks\.ai/v1$"},
            {"api_base": "https://litellm-dev.direct.fireworks.ai/v1"},
            False,
        ),
    ],
)
def test_configurable_clientside_parameters(
    allowed_param, input_value, should_return_true
):
    router = Router(
        model_list=[
            {
                "model_name": "dummy-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "dummy-key",
                    "configurable_clientside_auth_params": [allowed_param],
                },
            }
        ]
    )
    resp = _allow_model_level_clientside_configurable_parameters(
        model="dummy-model",
        param="api_base",
        request_body_value=input_value["api_base"],
        llm_router=router,
    )
    print(resp)
    assert resp == should_return_true


def test_get_end_user_id_from_request_body_always_returns_str():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    request_body = {"user": 123}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id == "123"
    assert isinstance(end_user_id, str)
