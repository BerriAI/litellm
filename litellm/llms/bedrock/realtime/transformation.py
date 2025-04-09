from typing import Any, Optional

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.utils import LlmProviders


class BedrockRealtimeConfig(BaseRealtimeConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return f"{api_base}/model/{model}/invoke-with-bidirectional-stream"

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        litellm_params: dict,
        client: Optional[Any] = None,
        logging_obj: Optional[LiteLLMLogging] = None,
        timeout: Optional[float] = None,
    ):
        pass
