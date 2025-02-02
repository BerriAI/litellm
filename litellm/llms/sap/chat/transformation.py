"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration  `/completion`
"""

from typing import List, Optional, Tuple, Union, Dict
from dataclasses import asdict
from pydantic import BaseModel

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


from litellm.types.utils import ModelResponse, Usage
from litellm.utils import CustomStreamWrapper
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.message import Message
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.response_format import ResponseFormatJsonSchema



class GenAIHubOrchestrationConfig(OpenAIGPTConfig):

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None#
    model_version: str = "latest"

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # groq is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.groq.com/openai/v1
        api_base = (
            api_base
            or get_secret_str("GROQ_API_BASE")
            or "https://api.groq.com/openai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("GROQ_API_KEY")
        return api_base, dynamic_api_key

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Groq doesn't support 'response_format' while streaming
        """
        if optional_params.get("response_format") is not None:
            return True

        return False


    def _transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: dict,
        litellm_params: dict,
    ) -> OrchestrationConfig:
        messages_ = [Message(**kwargs) for kwargs in messages]
        model_version = optional_params.pop('model_version', 'latest')
        tools = optional_params.pop("tools", None)
        response_format = optional_params.pop("response_format", None)
        if isinstance(response_format, dict):
            print(response_format)
            response_format = ResponseFormatJsonSchema(**response_format['json_schema'])
            print([*response_format.to_dict()['json_schema'].keys()])
        return OrchestrationConfig(
            template=Template(messages=messages_, response_format=response_format),
            llm=LLM(name=model, parameters=optional_params, version=model_version),
        )

    def _transform_response(
        self,
        model: str,
        response,
        model_response: ModelResponse,
        config: OrchestrationConfig,
        stream: bool,
        logging_obj: Optional[LiteLLMLoggingObject],
        optional_params: dict,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        llm_response = response.module_results.llm
        obj = model_response.model_dump()# response.module_results.llm
        obj['choices'] = [asdict(c) for c in llm_response.choices]
        for c in obj['choices']:
            c["message"]["role"] = c["message"]["role"].value
        obj['created'] = llm_response.created
        obj['model'] = llm_response.model
        obj["usage"] = Usage(**asdict(llm_response.usage))
        return ModelResponse.model_validate(obj)