import os
import sys
from fastapi.exceptions import HTTPException
from unittest.mock import patch
from httpx import Response, Request

import pytest

from litellm import DualCache
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.acuvity import AcuvityGuardrail

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
import litellm
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


@pytest.mark.asyncio
async def test_aim_guard_config():
    litellm.set_verbose = True
    litellm.guardrail_name_config_map = {}

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "gibberish-guard",
                "litellm_params": {
                    "guardrail": "acuvity",
                    "guard_name": "gibberish_guard",
                    "mode": "pre_call",
                    "api_key": "",
                    "vendor_params": {
                        "guardrails": [
                            {
                            "name": "pii_detector",
                            "matches": {
                                "email_address": {
                                "redact": True
                                }
                            }
                            }
                        ]
                    }
                },
            }
        ],
        config_file_path="",
    )
    acuvity_guardrails = [callback for callback in litellm.callbacks if isinstance(callback, AcuvityGuardrail)]
    assert len(acuvity_guardrails) == 1
    acuvity_guardrails = acuvity_guardrails[0]

    data = {
        "messages": [
            {"role": "user", "content": "my email is abcd.12@gmail.com"},
        ]
    }

    resp = await acuvity_guardrails.async_pre_call_hook(
                data=data, cache=DualCache(), user_api_key_dict=UserAPIKeyAuth(), call_type="completion"
            )
    print(resp)
    assert resp['messages'][0]['content'] == "my email is XXXXXXXXXXXXXXXXX"
