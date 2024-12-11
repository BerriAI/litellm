"""
Translate from OpenAI's `/v1/chat/completions` to Friendliai's `/v1/chat/completions`
"""

import json
import types
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

from ...openai_like.chat.handler import OpenAILikeChatConfig


class FriendliaiChatConfig(OpenAILikeChatConfig):
    pass
