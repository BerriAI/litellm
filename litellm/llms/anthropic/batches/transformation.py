import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from httpx import Response

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class AnthropicBatchesConfig:
    def __init__(self):
        from ..chat.transformation import AnthropicConfig

        self.anthropic_chat_config = AnthropicConfig()  # initialize once

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
        from litellm.cost_calculator import BaseTokenUsageProcessor
        from litellm.types.utils import Usage

        response_text = raw_response.text.strip()
        all_usage: List[Usage] = []

        try:
            # Split by newlines and try to parse each line as JSON
            lines = response_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    response_json = json.loads(line)
                    # Update model_response with the parsed JSON
                    completion_response = response_json["result"]["message"]
                    transformed_response = (
                        self.anthropic_chat_config.transform_parsed_response(
                            completion_response=completion_response,
                            raw_response=raw_response,
                            model_response=model_response,
                        )
                    )

                    transformed_response_usage = getattr(
                        transformed_response, "usage", None
                    )
                    if transformed_response_usage:
                        all_usage.append(cast(Usage, transformed_response_usage))
                except json.JSONDecodeError:
                    continue

            ## SUM ALL USAGE
            combined_usage = BaseTokenUsageProcessor.combine_usage_objects(all_usage)
            setattr(model_response, "usage", combined_usage)

            return model_response
        except Exception as e:
            raise e
