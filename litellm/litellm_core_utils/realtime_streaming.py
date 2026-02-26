import asyncio
import concurrent.futures
import json
from typing import Any, Dict, List, Optional, Union, Protocol, cast  # noqa: F401

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamSessionEvents,
)
from litellm.types.realtime import ALL_DELTA_TYPES

from .litellm_logging import Logging as LiteLLMLogging

class WSProtocol(Protocol):
    async def send(self, data: str) -> None: ...
    async def recv(self, **kwargs: Any) -> Any: ...
    async def close(self) -> None: ...

CLIENT_CONNECTION_CLASS = WSProtocol

# Create a thread pool with a maximum of 10 threads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)