import copy
import json
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    get_args,
)

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.prompt_templates.factory import (
    cohere_message_pt,
    custom_prompt,
    deepseek_r1_pt,
    prompt_factory,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.bedrock.chat.invoke_handler import make_call, make_sync_call
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.types.llms.bedrock import (
    BedrockInvokeAI21Request,
    BedrockInvokeCohereChatRequest,
    BedrockInvokeCohereCompletionRequest,
    BedrockInvokeLlamaRequest,
    BedrockInvokeMistralRequest,
    BedrockInvokeTitanInferenceParams,
    BedrockInvokeTitanRequest,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import CustomStreamWrapper

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM


@dataclass(frozen=True, slots=True)
class BedrockInvokeCohereParams:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    return_likelihood: Optional[str] = None
    p: Optional[float] = None
    k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    num_generations: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    truncate: Optional[str] = None
    stream: Optional[bool] = None
    tools: Optional[List[Dict[str, object]]] = None
    tool_results: Optional[List[Dict[str, object]]] = None
    seed: Optional[int] = None
    force_single_step: Optional[bool] = None


@dataclass(frozen=True, slots=True)
class BedrockInvokeAI21Params:
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[List[str]] = None
    frequencyPenalty: Optional[Dict[str, object]] = None
    frequencePenalty: Optional[Dict[str, object]] = None
    presencePenalty: Optional[Dict[str, object]] = None
    countPenalty: Optional[Dict[str, object]] = None


@dataclass(frozen=True, slots=True)
class BedrockInvokeMistralParams:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stop: Optional[List[str]] = None


@dataclass(frozen=True, slots=True)
class BedrockInvokeTitanParams:
    maxTokenCount: Optional[int] = None
    stopSequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    topP: Optional[int] = None


@dataclass(frozen=True, slots=True)
class BedrockInvokeLlamaParams:
    max_gen_len: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    topP: Optional[float] = None


class AmazonInvokeConfig(BaseConfig, BaseAWSLLM):
    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        This is a base invoke model mapping. For Invoke - define a bedrock provider specific config that extends this class.
        """
        return [
            "max_tokens",
            "max_completion_tokens",
            "stream",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        This is a base invoke model mapping. For Invoke - define a bedrock provider specific config that extends this class.
        """
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "stream":
                optional_params["stream"] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        provider = self.get_bedrock_invoke_provider(model)
        modelId = self.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params=optional_params,
        )
        ### SET RUNTIME ENDPOINT ###
        aws_bedrock_runtime_endpoint = optional_params.get(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=self._get_aws_region_name(
                optional_params=optional_params, model=model
            ),
        )

        if (stream is not None and stream is True) and provider != "ai21":
            endpoint_url = f"{endpoint_url}/model/{modelId}/invoke-with-response-stream"
            proxy_endpoint_url = (
                f"{proxy_endpoint_url}/model/{modelId}/invoke-with-response-stream"
            )
        else:
            endpoint_url = f"{endpoint_url}/model/{modelId}/invoke"
            proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/invoke"

        return endpoint_url

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    @staticmethod
    def _get_param_value(optional_params: dict, config: dict, key: str) -> object:
        if key in optional_params:
            return copy.deepcopy(optional_params[key])
        return copy.deepcopy(config.get(key))

    @staticmethod
    def _drop_none(data: Dict[str, object]) -> Dict[str, object]:
        return {key: value for key, value in data.items() if value is not None}

    def _parse_cohere_params(
        self, optional_params: dict, config: dict
    ) -> BedrockInvokeCohereParams:
        return BedrockInvokeCohereParams(
            max_tokens=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "max_tokens"),
            ),
            temperature=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "temperature"),
            ),
            return_likelihood=cast(
                Optional[str],
                self._get_param_value(optional_params, config, "return_likelihood"),
            ),
            p=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "p"),
            ),
            k=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "k"),
            ),
            stop_sequences=cast(
                Optional[List[str]],
                self._get_param_value(optional_params, config, "stop_sequences"),
            ),
            num_generations=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "num_generations"),
            ),
            frequency_penalty=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "frequency_penalty"),
            ),
            presence_penalty=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "presence_penalty"),
            ),
            truncate=cast(
                Optional[str],
                self._get_param_value(optional_params, config, "truncate"),
            ),
            stream=cast(
                Optional[bool],
                self._get_param_value(optional_params, config, "stream"),
            ),
            tools=cast(
                Optional[List[Dict[str, object]]],
                self._get_param_value(optional_params, config, "tools"),
            ),
            tool_results=cast(
                Optional[List[Dict[str, object]]],
                self._get_param_value(optional_params, config, "tool_results"),
            ),
            seed=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "seed"),
            ),
            force_single_step=cast(
                Optional[bool],
                self._get_param_value(optional_params, config, "force_single_step"),
            ),
        )

    def _parse_ai21_params(
        self, optional_params: dict, config: dict
    ) -> BedrockInvokeAI21Params:
        return BedrockInvokeAI21Params(
            maxTokens=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "maxTokens"),
            ),
            temperature=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "temperature"),
            ),
            topP=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "topP"),
            ),
            stopSequences=cast(
                Optional[List[str]],
                self._get_param_value(optional_params, config, "stopSequences"),
            ),
            frequencyPenalty=cast(
                Optional[Dict[str, object]],
                self._get_param_value(optional_params, config, "frequencyPenalty"),
            ),
            frequencePenalty=cast(
                Optional[Dict[str, object]],
                self._get_param_value(optional_params, config, "frequencePenalty"),
            ),
            presencePenalty=cast(
                Optional[Dict[str, object]],
                self._get_param_value(optional_params, config, "presencePenalty"),
            ),
            countPenalty=cast(
                Optional[Dict[str, object]],
                self._get_param_value(optional_params, config, "countPenalty"),
            ),
        )

    def _parse_mistral_params(
        self, optional_params: dict, config: dict
    ) -> BedrockInvokeMistralParams:
        return BedrockInvokeMistralParams(
            max_tokens=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "max_tokens"),
            ),
            temperature=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "temperature"),
            ),
            top_p=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "top_p"),
            ),
            top_k=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "top_k"),
            ),
            stop=cast(
                Optional[List[str]],
                self._get_param_value(optional_params, config, "stop"),
            ),
        )

    def _parse_titan_params(
        self, optional_params: dict, config: dict
    ) -> BedrockInvokeTitanParams:
        return BedrockInvokeTitanParams(
            maxTokenCount=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "maxTokenCount"),
            ),
            stopSequences=cast(
                Optional[List[str]],
                self._get_param_value(optional_params, config, "stopSequences"),
            ),
            temperature=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "temperature"),
            ),
            topP=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "topP"),
            ),
        )

    def _parse_llama_params(
        self, optional_params: dict, config: dict
    ) -> BedrockInvokeLlamaParams:
        return BedrockInvokeLlamaParams(
            max_gen_len=cast(
                Optional[int],
                self._get_param_value(optional_params, config, "max_gen_len"),
            ),
            temperature=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "temperature"),
            ),
            top_p=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "top_p"),
            ),
            topP=cast(
                Optional[float],
                self._get_param_value(optional_params, config, "topP"),
            ),
        )

    @staticmethod
    def _build_cohere_chat_request(
        prompt: str,
        inference_params: BedrockInvokeCohereParams,
        chat_history: Optional[List[Dict[str, object]]],
    ) -> BedrockInvokeCohereChatRequest:
        request: Dict[str, object] = {
            "message": prompt,
            **AmazonInvokeConfig._cohere_params_to_wire_dict(inference_params),
        }
        if chat_history is not None:
            request["chat_history"] = chat_history
        return cast(BedrockInvokeCohereChatRequest, request)

    @staticmethod
    def _build_cohere_completion_request(
        prompt: str,
        inference_params: BedrockInvokeCohereParams,
    ) -> BedrockInvokeCohereCompletionRequest:
        return cast(
            BedrockInvokeCohereCompletionRequest,
            {
                "prompt": prompt,
                **AmazonInvokeConfig._cohere_params_to_wire_dict(inference_params),
            },
        )

    @staticmethod
    def _build_ai21_request(
        prompt: str,
        inference_params: BedrockInvokeAI21Params,
    ) -> BedrockInvokeAI21Request:
        return cast(
            BedrockInvokeAI21Request,
            {
                "prompt": prompt,
                **AmazonInvokeConfig._drop_none(
                    {
                        "maxTokens": inference_params.maxTokens,
                        "temperature": inference_params.temperature,
                        "topP": inference_params.topP,
                        "stopSequences": inference_params.stopSequences,
                        "frequencyPenalty": inference_params.frequencyPenalty,
                        "frequencePenalty": inference_params.frequencePenalty,
                        "presencePenalty": inference_params.presencePenalty,
                        "countPenalty": inference_params.countPenalty,
                    }
                ),
            },
        )

    @staticmethod
    def _build_mistral_request(
        prompt: str,
        inference_params: BedrockInvokeMistralParams,
    ) -> BedrockInvokeMistralRequest:
        return cast(
            BedrockInvokeMistralRequest,
            {
                "prompt": prompt,
                **AmazonInvokeConfig._drop_none(
                    {
                        "max_tokens": inference_params.max_tokens,
                        "temperature": inference_params.temperature,
                        "top_p": inference_params.top_p,
                        "top_k": inference_params.top_k,
                        "stop": inference_params.stop,
                    }
                ),
            },
        )

    @staticmethod
    def _build_titan_request(
        prompt: str,
        inference_params: BedrockInvokeTitanParams,
    ) -> BedrockInvokeTitanRequest:
        return BedrockInvokeTitanRequest(
            inputText=prompt,
            textGenerationConfig=cast(
                BedrockInvokeTitanInferenceParams,
                AmazonInvokeConfig._drop_none(
                    {
                        "maxTokenCount": inference_params.maxTokenCount,
                        "stopSequences": inference_params.stopSequences,
                        "temperature": inference_params.temperature,
                        "topP": inference_params.topP,
                    }
                ),
            ),
        )

    @staticmethod
    def _build_llama_request(
        prompt: str,
        inference_params: BedrockInvokeLlamaParams,
    ) -> BedrockInvokeLlamaRequest:
        return cast(
            BedrockInvokeLlamaRequest,
            {
                "prompt": prompt,
                **AmazonInvokeConfig._drop_none(
                    {
                        "max_gen_len": inference_params.max_gen_len,
                        "temperature": inference_params.temperature,
                        "top_p": inference_params.top_p,
                        "topP": inference_params.topP,
                    }
                ),
            },
        )

    @staticmethod
    def _cohere_params_to_wire_dict(
        inference_params: BedrockInvokeCohereParams,
    ) -> Dict[str, object]:
        return AmazonInvokeConfig._drop_none(
            {
                "max_tokens": inference_params.max_tokens,
                "temperature": inference_params.temperature,
                "return_likelihood": inference_params.return_likelihood,
                "p": inference_params.p,
                "k": inference_params.k,
                "stop_sequences": inference_params.stop_sequences,
                "num_generations": inference_params.num_generations,
                "frequency_penalty": inference_params.frequency_penalty,
                "presence_penalty": inference_params.presence_penalty,
                "truncate": inference_params.truncate,
                "stream": inference_params.stream,
                "tools": inference_params.tools,
                "tool_results": inference_params.tool_results,
                "seed": inference_params.seed,
                "force_single_step": inference_params.force_single_step,
            }
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        ## SETUP ##
        stream = optional_params.pop("stream", None)
        optional_params.pop("stream_chunk_size", None)
        custom_prompt_dict: dict = litellm_params.pop("custom_prompt_dict", None) or {}
        hf_model_name = litellm_params.get("hf_model_name", None)

        provider = self.get_bedrock_invoke_provider(model)

        prompt, chat_history = self.convert_messages_to_prompt(
            model=hf_model_name or model,
            messages=messages,
            provider=provider,
            custom_prompt_dict=custom_prompt_dict,
        )
        if provider == "cohere":
            if model.startswith("cohere.command-r"):
                config = litellm.AmazonCohereChatConfig().get_config()
                cohere_inference_params = self._parse_cohere_params(
                    optional_params=optional_params,
                    config=config,
                )
                return cast(
                    dict,
                    self._build_cohere_chat_request(
                        prompt=prompt,
                        inference_params=cohere_inference_params,
                        chat_history=chat_history,
                    ),
                )
            else:
                config = litellm.AmazonCohereConfig.get_config()
                cohere_inference_params = self._parse_cohere_params(
                    optional_params=optional_params,
                    config=config,
                )
                if stream is True:
                    cohere_inference_params = replace(
                        cohere_inference_params,
                        stream=True,
                    )
                return cast(
                    dict,
                    self._build_cohere_completion_request(
                        prompt=prompt,
                        inference_params=cohere_inference_params,
                    ),
                )
        elif provider == "anthropic":
            transformed_request = (
                litellm.AmazonAnthropicClaudeConfig().transform_request(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    headers=headers,
                )
            )

            return transformed_request
        elif provider == "nova":
            return litellm.AmazonInvokeNovaConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        elif provider == "ai21":
            config = litellm.AmazonAI21Config.get_config()
            ai21_inference_params = self._parse_ai21_params(
                optional_params=optional_params,
                config=config,
            )
            return cast(
                dict,
                self._build_ai21_request(
                    prompt=prompt,
                    inference_params=ai21_inference_params,
                ),
            )
        elif provider == "mistral":
            config = litellm.AmazonMistralConfig.get_config()
            mistral_inference_params = self._parse_mistral_params(
                optional_params=optional_params,
                config=config,
            )
            return cast(
                dict,
                self._build_mistral_request(
                    prompt=prompt,
                    inference_params=mistral_inference_params,
                ),
            )
        elif provider == "amazon":  # amazon titan
            config = litellm.AmazonTitanConfig.get_config()
            titan_inference_params = self._parse_titan_params(
                optional_params=optional_params,
                config=config,
            )
            return cast(
                dict,
                self._build_titan_request(
                    prompt=prompt,
                    inference_params=titan_inference_params,
                ),
            )
        elif provider == "meta" or provider == "llama" or provider == "deepseek_r1":
            config = litellm.AmazonLlamaConfig.get_config()
            llama_inference_params = self._parse_llama_params(
                optional_params=optional_params,
                config=config,
            )
            return cast(
                dict,
                self._build_llama_request(
                    prompt=prompt,
                    inference_params=llama_inference_params,
                ),
            )
        elif provider == "twelvelabs":
            return litellm.AmazonTwelveLabsPegasusConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        elif provider == "openai":
            # OpenAI imported models use OpenAI Chat Completions format
            return litellm.AmazonBedrockOpenAIConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        else:
            raise BedrockError(
                status_code=404,
                message="Bedrock Invoke HTTPX: Unknown provider={}, model={}. Try calling via converse route - `bedrock/converse/<model>`.".format(
                    provider, model
                ),
            )

    def transform_response(  # noqa: PLR0915
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            completion_response = raw_response.json()
        except Exception:
            raise BedrockError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        verbose_logger.debug(
            "bedrock invoke response % s",
            json.dumps(completion_response, indent=4, default=str),
        )
        provider = self.get_bedrock_invoke_provider(model)
        outputText: Optional[str] = None
        try:
            if provider == "cohere":
                if "text" in completion_response:
                    outputText = completion_response["text"]  # type: ignore
                elif "generations" in completion_response:
                    outputText = completion_response["generations"][0]["text"]
                    model_response.choices[0].finish_reason = map_finish_reason(
                        completion_response["generations"][0]["finish_reason"]
                    )
            elif provider == "anthropic":
                return litellm.AmazonAnthropicClaudeConfig().transform_response(
                    model=model,
                    raw_response=raw_response,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    request_data=request_data,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    encoding=encoding,
                    api_key=api_key,
                    json_mode=json_mode,
                )
            elif provider == "nova":
                return litellm.AmazonInvokeNovaConfig().transform_response(
                    model=model,
                    raw_response=raw_response,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    request_data=request_data,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    encoding=encoding,
                )
            elif provider == "twelvelabs":
                return litellm.AmazonTwelveLabsPegasusConfig().transform_response(
                    model=model,
                    raw_response=raw_response,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    request_data=request_data,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    encoding=encoding,
                    api_key=api_key,
                    json_mode=json_mode,
                )
            elif provider == "ai21":
                outputText = (
                    completion_response.get("completions")[0].get("data").get("text")
                )
            elif provider == "meta" or provider == "llama" or provider == "deepseek_r1":
                outputText = completion_response["generation"]
            elif provider == "mistral":
                outputText = litellm.AmazonMistralConfig.get_outputText(
                    completion_response, model_response
                )
            else:  # amazon titan
                outputText = completion_response.get("results")[0].get("outputText")
        except Exception as e:
            raise BedrockError(
                message="Error processing={}, Received error={}".format(
                    raw_response.text, str(e)
                ),
                status_code=422,
            )

        try:
            if (
                outputText is not None
                and len(outputText) > 0
                and hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None)  # type: ignore
                is None
            ):
                model_response.choices[0].message.content = outputText  # type: ignore
            elif (
                hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None)  # type: ignore
                is not None
            ):
                pass
            else:
                raise Exception()
        except Exception as e:
            raise BedrockError(
                message="Error parsing received text={}.\nError-{}".format(
                    outputText, str(e)
                ),
                status_code=raw_response.status_code,
            )

        ## CALCULATING USAGE - bedrock returns usage in the headers
        bedrock_input_tokens = raw_response.headers.get(
            "x-amzn-bedrock-input-token-count", None
        )
        bedrock_output_tokens = raw_response.headers.get(
            "x-amzn-bedrock-output-token-count", None
        )

        prompt_tokens = int(
            bedrock_input_tokens or litellm.token_counter(messages=messages)
        )

        completion_tokens = int(
            bedrock_output_tokens
            or litellm.token_counter(
                text=model_response.choices[0].message.content,  # type: ignore
                count_response_tokens=True,
            )
        )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message)

    @track_llm_api_timing()
    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[AsyncHTTPHandler] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> CustomStreamWrapper:
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=json.dumps(data),
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                fake_stream=True if "ai21" in api_base else False,
                bedrock_invoke_provider=self.get_bedrock_invoke_provider(model),
                json_mode=json_mode,
            ),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    @track_llm_api_timing()
    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> CustomStreamWrapper:
        if client is None or isinstance(client, AsyncHTTPHandler):
            client = _get_httpx_client(params={})
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_sync_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=json.dumps(data),
                signed_json_body=signed_json_body,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                fake_stream=True if "ai21" in api_base else False,
                bedrock_invoke_provider=self.get_bedrock_invoke_provider(model),
                json_mode=json_mode,
            ),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )
        return streaming_response

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """
        Bedrock invoke does not allow passing `stream` in the request body.
        """
        return False

    @staticmethod
    def get_bedrock_invoke_provider(
        model: str,
    ) -> Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL]:
        """
        Helper function to get the bedrock provider from the model

        handles 4 scenarios:
        1. model=invoke/anthropic.claude-3-5-sonnet-20240620-v1:0 -> Returns `anthropic`
        2. model=anthropic.claude-3-5-sonnet-20240620-v1:0 -> Returns `anthropic`
        3. model=llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n -> Returns `llama`
        4. model=us.amazon.nova-pro-v1:0 -> Returns `nova`
        """
        if model.startswith("invoke/"):
            model = model.replace("invoke/", "", 1)

        # Special case: Check for "nova" in model name first (before "amazon")
        # This handles amazon.nova-* models which would otherwise match "amazon" (Titan)
        if "nova" in model.lower():
            if "nova" in get_args(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL):
                return cast(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL, "nova")

        _split_model = model.split(".")[0]
        if _split_model in get_args(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL):
            return cast(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL, _split_model)

        # If not a known provider, check for pattern with two slashes
        provider = AmazonInvokeConfig._get_provider_from_model_path(model)
        if provider is not None:
            return provider

        for provider in get_args(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL):
            if provider in model:
                return provider
        return None

    @staticmethod
    def _get_provider_from_model_path(
        model_path: str,
    ) -> Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL]:
        """
        Helper function to get the provider from a model path with format: provider/model-name

        Args:
            model_path (str): The model path (e.g., 'llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n' or 'anthropic/model-name')

        Returns:
            Optional[str]: The provider name, or None if no valid provider found
        """
        parts = model_path.split("/")
        if len(parts) >= 1:
            provider = parts[0]
            if provider in get_args(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL):
                return cast(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL, provider)
        return None

    def convert_messages_to_prompt(
        self, model, messages, provider, custom_prompt_dict
    ) -> Tuple[str, Optional[list]]:
        # handle anthropic prompts and amazon titan prompts
        prompt = ""
        chat_history: Optional[list] = None
        ## CUSTOM PROMPT
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                messages=messages,
            )
            return prompt, None
        ## ELSE
        if provider == "anthropic" or provider == "amazon":
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="bedrock"
            )
        elif provider == "mistral":
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="bedrock"
            )
        elif provider == "meta" or provider == "llama":
            prompt = prompt_factory(
                model=model, messages=messages, custom_llm_provider="bedrock"
            )
        elif provider == "cohere":
            prompt, chat_history = cohere_message_pt(messages=messages)
        elif provider == "deepseek_r1":
            prompt = deepseek_r1_pt(messages=messages)
        else:
            prompt = ""
            for message in messages:
                if "role" in message:
                    if message["role"] == "user":
                        prompt += f"{message['content']}"
                    else:
                        prompt += f"{message['content']}"
                else:
                    prompt += f"{message['content']}"
        return prompt, chat_history  # type: ignore
