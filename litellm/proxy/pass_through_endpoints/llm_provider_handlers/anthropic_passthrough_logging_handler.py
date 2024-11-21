import json
from datetime import datetime
from typing import Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class AnthropicPassthroughLoggingHandler:

    @staticmethod
    async def anthropic_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """
        Transforms Anthropic response to OpenAI response, generates a standard logging object so downstream logging can be handled
        """
        model = response_body.get("model", "")
        litellm_model_response: litellm.ModelResponse = (
            AnthropicConfig._process_response(
                response=httpx_response,
                model_response=litellm.ModelResponse(),
                model=model,
                stream=False,
                messages=[],
                logging_obj=logging_obj,
                optional_params={},
                api_key="",
                data={},
                print_verbose=litellm.print_verbose,
                encoding=None,
                json_mode=False,
            )
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=litellm_model_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        await logging_obj.async_success_handler(
            result=litellm_model_response,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            **kwargs,
        )

        pass

    @staticmethod
    def _create_anthropic_response_logging_payload(
        litellm_model_response: Union[
            litellm.ModelResponse, litellm.TextCompletionResponse
        ],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Create the standard logging object for Anthropic passthrough

        handles streaming and non-streaming responses
        """
        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
        )
        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # Make standard logging object for Vertex AI
        standard_logging_object = get_standard_logging_object_payload(
            kwargs=kwargs,
            init_response_obj=litellm_model_response,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )

        # pretty print standard logging object
        verbose_proxy_logger.debug(
            "standard_logging_object= %s", json.dumps(standard_logging_object, indent=4)
        )
        kwargs["standard_logging_object"] = standard_logging_object
        return kwargs
