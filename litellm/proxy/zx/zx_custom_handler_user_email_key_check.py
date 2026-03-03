import os
from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache
from litellm.types.utils import (
    CallTypesLiteral,
    ModelResponseStream,
)
from typing import Any, AsyncGenerator, Optional, Dict
from fastapi import HTTPException


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger): # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        super().__init__()

    #### CALL HOOKS - proxy only #### 

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: CallTypesLiteral): 
        if user_api_key_dict.key_alias is not None and '@' in user_api_key_dict.key_alias:
            text = None
            system_msg = data.get('system', [])
            if system_msg and len(system_msg) > 0:
                text = system_msg[0].get('text')
            if text and 'running inside OpenClaw' in text:
                return HTTPException(
                    status_code=403,
                    detail="当前Key无法在 OpenClaw 中使用，请在配置程序中添加 --assistant-openclaw 参数生成新的 Key。"
                )
        return data


proxy_handler_instance = MyCustomHandler()