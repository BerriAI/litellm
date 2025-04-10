import copy
import json
import time
import urllib.parse
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast, get_args

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
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import CustomStreamWrapper

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM


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
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> dict:
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_session_token, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.get("aws_secret_access_key", None)
        aws_access_key_id = optional_params.get("aws_access_key_id", None)
        aws_session_token = optional_params.get("aws_session_token", None)
        aws_role_name = optional_params.get("aws_role_name", None)
        aws_session_name = optional_params.get("aws_session_name", None)
        aws_profile_name = optional_params.get("aws_profile_name", None)
        aws_web_identity_token = optional_params.get("aws_web_identity_token", None)
        aws_sts_endpoint = optional_params.get("aws_sts_endpoint", None)
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params, model=model
        )

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
        )

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
        if headers is not None:
            headers = {"Content-Type": "application/json", **headers}
        else:
            headers = {"Content-Type": "application/json"}

        request = AWSRequest(
            method="POST",
            url=api_base,
            data=json.dumps(request_data),
            headers=headers,
        )
        sigv4.add_auth(request)

        request_headers_dict = dict(request.headers)
        if (
            headers is not None and "Authorization" in headers
        ):  # prevent sigv4 from overwriting the auth header
            request_headers_dict["Authorization"] = headers["Authorization"]
        return request_headers_dict

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
        custom_prompt_dict: dict = litellm_params.pop("custom_prompt_dict", None) or {}
        hf_model_name = litellm_params.get("hf_model_name", None)

        provider = self.get_bedrock_invoke_provider(model)

        prompt, chat_history = self.convert_messages_to_prompt(
            model=hf_model_name or model,
            messages=messages,
            provider=provider,
            custom_prompt_dict=custom_prompt_dict,
        )
        inference_params = copy.deepcopy(optional_params)
        inference_params = {
            k: v
            for k, v in inference_params.items()
            if k not in self.aws_authentication_params
        }
        request_data: dict = {}
        if provider == "cohere":
            if model.startswith("cohere.command-r"):
                ## LOAD CONFIG
                config = litellm.AmazonCohereChatConfig().get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                _data = {"message": prompt, **inference_params}
                if chat_history is not None:
                    _data["chat_history"] = chat_history
                request_data = _data
            else:
                ## LOAD CONFIG
                config = litellm.AmazonCohereConfig.get_config()
                for k, v in config.items():
                    if (
                        k not in inference_params
                    ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                        inference_params[k] = v
                if stream is True:
                    inference_params[
                        "stream"
                    ] = True  # cohere requires stream = True in inference params
                request_data = {"prompt": prompt, **inference_params}
        elif provider == "anthropic":
            return litellm.AmazonAnthropicClaude3Config().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        elif provider == "nova":
            return litellm.AmazonInvokeNovaConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        elif provider == "ai21":
            ## LOAD CONFIG
            config = litellm.AmazonAI21Config.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            request_data = {"prompt": prompt, **inference_params}
        elif provider == "mistral":
            ## LOAD CONFIG
            config = litellm.AmazonMistralConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            request_data = {"prompt": prompt, **inference_params}
        elif provider == "amazon":  # amazon titan
            ## LOAD CONFIG
            config = litellm.AmazonTitanConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v

            request_data = {
                "inputText": prompt,
                "textGenerationConfig": inference_params,
            }
        elif provider == "meta" or provider == "llama" or provider == "deepseek_r1":
            ## LOAD CONFIG
            config = litellm.AmazonLlamaConfig.get_config()
            for k, v in config.items():
                if (
                    k not in inference_params
                ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                    inference_params[k] = v
            request_data = {"prompt": prompt, **inference_params}
        else:
            raise BedrockError(
                status_code=404,
                message="Bedrock Invoke HTTPX: Unknown provider={}, model={}. Try calling via converse route - `bedrock/converse/<model>`.".format(
                    provider, model
                ),
            )

        return request_data

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
                return litellm.AmazonAnthropicClaude3Config().transform_response(
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
            elif provider == "ai21":
                outputText = (
                    completion_response.get("completions")[0].get("data").get("text")
                )
            elif provider == "meta" or provider == "llama" or provider == "deepseek_r1":
                outputText = completion_response["generation"]
            elif provider == "mistral":
                outputText = completion_response["outputs"][0]["text"]
                model_response.choices[0].finish_reason = completion_response[
                    "outputs"
                ][0]["stop_reason"]
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
    def get_async_custom_stream_wrapper(
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

        _split_model = model.split(".")[0]
        if _split_model in get_args(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL):
            return cast(litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL, _split_model)

        # If not a known provider, check for pattern with two slashes
        provider = AmazonInvokeConfig._get_provider_from_model_path(model)
        if provider is not None:
            return provider

        # check if provider == "nova"
        if "nova" in model:
            return "nova"

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

    def get_bedrock_model_id(
        self,
        optional_params: dict,
        provider: Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL],
        model: str,
    ) -> str:
        modelId = optional_params.pop("model_id", None)
        if modelId is not None:
            modelId = self.encode_model_id(model_id=modelId)
        else:
            modelId = model

        modelId = modelId.replace("invoke/", "", 1)
        if provider == "llama" and "llama/" in modelId:
            modelId = self._get_model_id_from_model_with_spec(modelId, spec="llama")
        elif provider == "deepseek_r1" and "deepseek_r1/" in modelId:
            modelId = self._get_model_id_from_model_with_spec(
                modelId, spec="deepseek_r1"
            )
        return modelId

    def _get_model_id_from_model_with_spec(
        self,
        model: str,
        spec: str,
    ) -> str:
        """
        Remove `llama` from modelID since `llama` is simply a spec to follow for custom bedrock models
        """
        model_id = model.replace(spec + "/", "")
        return self.encode_model_id(model_id=model_id)

    def encode_model_id(self, model_id: str) -> str:
        """
        Double encode the model ID to ensure it matches the expected double-encoded format.
        Args:
            model_id (str): The model ID to encode.
        Returns:
            str: The double-encoded model ID.
        """
        return urllib.parse.quote(model_id, safe="")

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
