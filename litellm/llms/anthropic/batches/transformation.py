from typing import TYPE_CHECKING, Any, Dict, List, Optional

from httpx import Response

from litellm.types.llms.openai import AllMessageValues
from litellm.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


ANTHROPIC_HOSTED_TOOLS = ["web_search", "bash", "text_editor", "code_execution"]


class AnthropicBatchesConfig:
    def __init__(self):
        pass

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        return model_response
