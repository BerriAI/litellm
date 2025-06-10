# litellm/proxy/guardrails/guardrail_hooks/prisma_airs_guardrail.py

import requests
import os
#from typing import Any, Dict, List, Literal, Optional, Union
from typing import Literal, Optional, Union
#import litellm
#from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
#from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
#from litellm.types.guardrails import GuardrailEventHooks


class prisma_airs_guardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank"
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        try:
            user_prompt = data["messages"][-1]["content"]
        except (AttributeError, IndexError, KeyError):
            return "Invalid input: 'messages' missing or improperly formatted."
        try:
            # Call AIRS service to scan the user prompt
            airs_response = call_airs_api(user_prompt)

            if airs_response.status_code != 200:
                return f"airs call failed (HTTP {airs_response.status_code})."
            if airs_response.json().get("action","") == "block":
                return "Request blocked by security policy."
                 #return airs_response
        except Exception as e:
            return f"Error calling AIRS {e}"

def call_airs_api(data):
  airs_response = requests.post(os.environ.get("PRISMA_AIRS_API_BASE"),
    # "<PRISMA_AIRS_API_BASE>", 
    headers={
        "x-pan-token": os.environ.get("PRISMA_AIRS_API_KEY"), 
        "Content-Type": "application/json"
    },
    json={
        "metadata": {
            "ai_model": "Test AI model",
            "app_name": "Google AI",
            "app_user": "test-user-1"
        },
        "contents": [
            {
                "prompt": data
            }
        ],
        #"tr_id": "1234",
        "ai_profile": {
            "profile_name": os.environ.get("PRISMA_AIRS_PROFILE_NAME")
        }
    },
    timeout=5,
    verify=False
  )
  return airs_response
return data
