from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache
from typing import Optional, Literal


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(
    CustomLogger
):  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        pass

    #### CALL HOOKS - proxy only ####

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
        ],
    ):
        return data

    async def async_post_call_failure_hook(
        self, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth
    ):
        pass

    async def async_post_call_success_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        # print("in async_post_call_success_hook")
        pass

    async def async_moderation_hook(  # call made in parallel to llm api call
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation"],
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        # print("in async_post_call_streaming_hook")
        pass


proxy_handler_instance = MyCustomHandler()
