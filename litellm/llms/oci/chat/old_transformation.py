import base64
import datetime
import hashlib
import time
from urllib.parse import urlparse
import litellm
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from enum import Enum

import httpx

from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
    version,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage, ChatCompletionTextObject
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from ..common_utils import API_BASE, OCIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def sha256_base64(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.b64encode(digest).decode()


def build_signature_string(method, path, headers, signed_headers):
    lines = []
    for header in signed_headers:
        if header == "(request-target)":
            value = f"{method.lower()} {path}"
        else:
            value = headers[header]
        lines.append(f"{header}: {value}")
    return "\n".join(lines)


def load_private_key_from_str(key_str: str):
    key = serialization.load_pem_private_key(
        key_str.encode("utf-8"),
        password=None,
    )
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError("The provided private key is not an RSA key, which is required for OCI signing.")
    return key

# 5 minute timeout (models may need to load)
STREAMING_TIMEOUT = 60 * 5

class OCIChatConfig(BaseConfig):
    """
    Configuration class for OCI's API interface.
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        # mark the class as using a custom stream wrapper because the default only iterates on lines
        setattr(self.__class__, "has_custom_stream_wrapper", True)

        self.openai_to_oci_generic_param_map = {
            "stream": "isStream",
            "max_tokens": "maxTokens",
            "max_completion_tokens": "maxTokens",
            "temperature": "temperature",
            "tools": "tools",
            "frequency_penalty": "frequencyPenalty",
            "logprobs": "logProbs",
            "logit_bias": "logitBias",
            "n": "numGenerations",
            "presence_penalty": "presencePenalty",
            "seed": "seed",
            "stop": "stop",
            "tool_choice": "toolChoice",
            "top_p": "topP",
            "max_retries": False,
            "top_logprobs": False,
            "modalities": False,
            "prediction": False,
            "stream_options": False,
            "function_call": False,
            "functions": False,
            "extra_headers": False,
            "parallel_tool_calls": False,
            "audio": False,
            "web_search_options": False,
        }

        self.openai_to_oci_cohere_param_map = {
            "stream": "isStream",
            "max_tokens": "maxTokens",
            "max_completion_tokens": "maxTokens",
            "temperature": "temperature",
            "tools": "tools",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
            "seed": "seed",
            "stop": "stopSequences",
            "top_p": "topP",
            "stream_options": "streamOptions",
            "max_retries": False,
            "top_logprobs": False,
            "modalities": False,
            "prediction": False,
            "function_call": False,
            "functions": False,
            "extra_headers": False,
            "parallel_tool_calls": False,
            "audio": False,
            "web_search_options": False,
        }


    def _get_vendor_from_model(self, model: str) -> OCIVendors:
        """
        Extracts the vendor from the model name.
        Args:
            model (str): The model name.
        Returns:
            str: The vendor name.
        """
        vendor = model.split(".")[0].lower()
        if vendor == "cohere":
            return OCIVendors.COHERE
        else:
            return OCIVendors.GENERIC

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = []
        vendor = self._get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            open_ai_to_oci_param_map = self.openai_to_oci_cohere_param_map
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map
        for key, value in open_ai_to_oci_param_map.items():
            if value:
                supported_params.append(key)

        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:

        adapted_params = {}
        vendor = self._get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            open_ai_to_oci_param_map = self.openai_to_oci_cohere_param_map
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        all_params = {**non_default_params, **optional_params}

        for key, value in all_params.items():

            alias = open_ai_to_oci_param_map.get(key)

            if alias is False:
                if drop_params:
                    continue

                raise Exception(f"param `{key}` is not supported on OCI")

            if alias is None:
                adapted_params[key] = value
                continue

            adapted_params[alias] = value

        return adapted_params

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
        """
        Some providers like Bedrock require signing the request. The sign request funtion needs access to `request_data` and `complete_url`
        Args:
            headers: dict
            optional_params: dict
            request_data: dict - the request body being sent in http request
            api_base: str - the complete url being sent in http request
        Returns:
            dict - the signed headers
        """
        import json

        oci_region = optional_params.get("oci_region", "us-ashburn-1")
        api_base = (
            api_base
            or litellm.api_base
            or f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com"
        )
        oci_user = optional_params.get("oci_user")
        oci_fingerprint = optional_params.get("oci_fingerprint")
        oci_tenancy = optional_params.get("oci_tenancy")
        oci_key = optional_params.get("oci_key")

        if not oci_user or not oci_fingerprint or not oci_tenancy or not oci_key:
            raise Exception(
                "Missing one of the following parameters: oci_user, oci_fingerprint, oci_tenancy, oci_key"
            )

        method = str(optional_params.get("method", "POST")).upper()
        body = json.dumps(request_data).encode("utf-8")
        parsed = urlparse(api_base)
        path = parsed.path or "/"
        host = parsed.netloc

        date = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        content_type = headers.get("content-type", "application/json")
        content_length = str(len(body))
        x_content_sha256 = sha256_base64(body)

        headers_to_sign = {
            "date": date,
            "host": host,
            "content-type": content_type,
            "content-length": content_length,
            "x-content-sha256": x_content_sha256,
        }

        signed_headers = ["date", "(request-target)", "host", "content-length", "content-type", "x-content-sha256"]
        signing_string = build_signature_string(method, path, headers_to_sign, signed_headers)

        private_key = load_private_key_from_str(oci_key)
        signature = private_key.sign(
            signing_string.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature).decode()

        key_id = f"{oci_tenancy}/{oci_user}/{oci_fingerprint}"

        authorization = (
            'Signature version="1",'
            f'keyId="{key_id}",'
            'algorithm="rsa-sha256",'
            f'headers="{" ".join(signed_headers)}",'
            f'signature="{signature_b64}"'
        )

        headers.update({
            "authorization": authorization,
            "date": date,
            "host": host,
            "content-type": content_type,
            "content-length": content_length,
            "x-content-sha256": x_content_sha256,
        })

        return headers, None

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

        oci_region = optional_params.get(
            "oci_region", "us-ashburn-1"
        )
        api_base = (
            api_base
            or litellm.api_base
            or f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com"
        )

        headers.update(
            {
                "content-type": "application/json",
                "user-agent": f"litellm/{version}",
            }
        )

        if not messages:
            raise Exception(
                "kwarg `messages` must be an array of messages that follow the openai chat standard"
            )

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return f"{API_BASE}/{model}"

    def _get_optional_params(
        self, vendor: OCIVendors, optional_params: dict
    ) -> Dict:
        selected_params = {}
        if vendor == OCIVendors.COHERE:
            open_ai_to_oci_param_map = self.openai_to_oci_cohere_param_map
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        for value in open_ai_to_oci_param_map.values():
            if value in optional_params:
                selected_params[value] = optional_params[value]
        if "tools" in selected_params:
            selected_params["tools"] = adapt_tools_to_oci_standard(selected_params["tools"], vendor)
        return selected_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        oci_compartment_id = optional_params.get("oci_compartment_id", None)
        if not oci_compartment_id:
            raise Exception(
                "kwarg `oci_compartment_id` is required for OCI requests"
            )

        vendor = self._get_vendor_from_model(model)

        data = {
            "compartmentId": oci_compartment_id,
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model,
            },
            "chatRequest": {
                "apiFormat": vendor.value,
            },
        }

        if vendor == OCIVendors.COHERE:
            raise Exception(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
        
            # TODO: Continue the implementation
            last_message = messages[-1]
            if last_message["role"] != "user":
                raise Exception(
                    "Last message must be from the USER when using Cohere models"
                )
            cast(ChatCompletionUserMessage, last_message)
            if isinstance(last_message["content"], str):
                data["chatRequest"]["message"] = last_message["content"]
            elif isinstance(last_message["content"], list) and len(last_message["content"]) > 0:
                if type(last_message["content"][0]) is not ChatCompletionTextObject:
                    raise Exception(
                        "Message content must be a list of ChatCompletionTextObject when using Cohere models"
                    )
                data["chatRequest"]["message"] = last_message["content"][0]["text"]
            data["chatRequest"]["chatHistory"] = adapt_messages_to_oci_standard(messages[:-1], vendor)  # type: ignore
        else:
            messages = adapt_messages_to_oci_standard(messages, vendor)  # type: ignore
            data["chatRequest"]["messages"] = messages
        
        data["chatRequest"] = {**data["chatRequest"], **self._get_optional_params(vendor, optional_params)}

        return data

    def transform_response(
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

        json = raw_response.json()  # noqa: F811

        error = json.get("error")

        if error is not None:
            raise OCIError(
                message=str(json["error"]),
                status_code=raw_response.status_code,
            )
        output = json.get("chatResponse")
        if not output:
            raise OCIError(
                message="Invalid response format from OCI",
                status_code=raw_response.status_code,
            )

        vendor = self._get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            model_response.created = int(time.time())
        else:
            iso_str = output["timeCreated"]
            dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            model_response.created = int(dt.timestamp())

        model_response.model = json.get("modelId", model)

        # Add the output
        if not output or not isinstance(output, dict):
            raise OCIError(
                message="Invalid response format from OCI",
                status_code=raw_response.status_code,
            )

        message = model_response.choices[0].message  # type: ignore
        if vendor == OCIVendors.COHERE:
            message.content = output["text"] 
        else:
            response_message = output["choices"][0]["message"]
            if "content" in response_message and isinstance(response_message["content"], list):
                message.content = response_message["content"][0]["text"]
            if "toolCalls" in response_message:
                message.tool_calls = adapt_tools_to_openai_standard(response_message["toolCalls"])

        usage = Usage(
            prompt_tokens=output["usage"]["promptTokens"],
            completion_tokens=output["usage"]["completionTokens"],
            total_tokens=output["usage"]["totalTokens"],
        )
        model_response.usage = usage  # type: ignore

        model_response._hidden_params["additional_headers"] = raw_response.headers

        return model_response

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
    ) -> "CustomStreamWrapper":
        if client is None or isinstance(client, AsyncHTTPHandler):
            client = _get_httpx_client(params={})

        try:
            response = client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        completion_stream = response.iter_text()

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streaming_response

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
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> "CustomStreamWrapper":
        if client is None or isinstance(client, HTTPHandler):
            client = get_async_httpx_client(llm_provider=LlmProviders.BYTEZ, params={})

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        completion_stream = response.aiter_text()

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streaming_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OCIError(status_code=status_code, message=error_message)


open_ai_to_generic_oci_role_map = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
}

# TODO: Implement the Cohere role mapping if needed
# open_ai_to_cohere_role_map = {
#     "system": "SYSTEM",
#     "user": "USER",
#     "assistant": "CHATBOT",
# }

def adapt_messages_to_oci_standard(messages: List[Dict], vendor: OCIVendors) -> List[Dict]:
    """
    Converts the message history to the standard expected by OCI models.
    This is a specific transformation for OCI models.
    Args:
        messages (List[AllMessageValues]): The list of messages to convert.
        vendor (OCIVendors): The vendor of the model.
    Returns:
        List[Dict]: The converted message history.
    """
    if vendor == OCIVendors.COHERE:
        # TODO: Implement the conversion for Cohere models
        # new_messages = adapt_messages_to_cohere_standard(messages)
        raise Exception(
            "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
        )
    else:
        new_messages = adapt_messages_to_generic_oci_standard(messages)

    return new_messages

def adapt_messages_to_generic_oci_standard(messages: List[Dict]) -> List[Dict]:
    new_messages = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        new_content = []

        if role in ["system", "user", "assistant"] and content is not None:
            if isinstance(content, str):
                new_content.append({"type": "TEXT", "text": content})
                new_messages.append({"role": open_ai_to_generic_oci_role_map[role], "content": new_content})
                continue

            # content is a list of content items:
            # [
            #     {"type": "text", "text": "Hello"},
            #     {"type": "image_url", "image_url": "https://example.com/image.png"}
            # ]
            if not isinstance(content, list):
                raise Exception("Prop `content` must be a list of content items")

            for content_item in content:
                type = content_item.get("type")
                if not isinstance(type, str):
                    raise Exception("Prop `type` is not a string")

                if type not in ["text", "image_url"]:
                    raise Exception(f"Prop `{type}` is not supported")

                if type == "text":
                    text = content_item.get("text")
                    if not isinstance(text, str):
                        raise Exception("Prop `text` is not a string")
                    new_content.append({
                        "type": "TEXT",
                        "text": text
                    })
                
                elif type == "image_url":
                    image_url = content_item.get("image_url")
                    if not isinstance(image_url, str):
                        raise Exception("Prop `image_url` is not a string")
                    new_content.append({
                        "type": "IMAGE",
                        "imageUrl": image_url
                    })
            new_messages.append({"role": open_ai_to_generic_oci_role_map[role], "content": new_content})

        elif role == "assistant" and tool_calls is not None:
            tool_calls_formated = []
            if not isinstance(tool_calls, list):
                raise Exception("Prop `tool_calls` must be a list of tool calls")
            for tool_call in tool_calls:
                if tool_call["type"] != "function":
                    raise Exception("OCI only supports function tools")

                tool_call_id = tool_call.get("id")
                if not isinstance(tool_call_id, str):
                    raise Exception("Prop `id` is not a string")
                
                function_name = tool_call["function"]["name"]
                if not isinstance(function_name, str):
                    raise Exception("Prop `name` is not a string")

                arguments = tool_call["function"].get("arguments", "{}")
                if not isinstance(arguments, str):
                    raise Exception("Prop `arguments` is not a string")

                tool_calls_formated.append({
                    "id": tool_call_id,
                    "type": "FUNCTION",
                    "name": function_name,
                    "arguments": arguments
                })
            
            new_messages.append({
                "role": open_ai_to_generic_oci_role_map[role],
                # "content": None,
                "toolCalls": tool_calls_formated
            })
        
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                raise Exception("Prop `tool_call_id` is not a string")

            content = message.get("content")
            if not isinstance(content, str):
                raise Exception("Prop `content` is not a string")

            new_messages.append({
                "role": "TOOL",
                "toolCallId": tool_call_id,
                "content": [{
                    "type": "TEXT",
                    "text": content
                }]
            })

    return new_messages

def adapt_tools_to_oci_standard(tools: List[Dict], vendor: OCIVendors):
    new_tools = []
    if vendor == OCIVendors.COHERE:
        for tool in tools:
            if tool["type"] != "function":
                raise Exception("OCI only supports function tools")

            parameters = {}
            for key, value in tool["function"]["parameters"]["properties"].items():
                parameters[key] = {
                    "description": value.get("description", ""),
                    "type": value.get("type", "string"),
                }
            if "required" in tool["function"]["parameters"]:
                for required_key in tool["function"]["parameters"]["required"]:
                    if required_key not in parameters:
                        raise Exception(f"Required key `{required_key}` not found in parameters")
                    parameters[required_key]["isRequired"] = True

            new_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameterDefinitions": parameters
            }
            new_tools.append(new_tool)
    else:
        for tool in tools:
            if tool["type"] != "function":
                raise Exception("OCI only supports function tools")
            
            new_tool = {
                "type": "FUNCTION",
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"]["parameters"],
            }
            new_tools.append(new_tool)
    
    return new_tools

def adapt_tools_to_openai_standard(tools: List[Dict]):
    new_tools = []
    for tool in tools:
        if tool["type"] != "FUNCTION":
            raise Exception("OCI only supports function tools")
        
        new_tool = {
            "type": "function",
            "id": tool["id"],
            "function": {
                "name": tool["name"],
                "arguments": tool.get("arguments", ""),
            }
        }

        new_tools.append(new_tool)
    
    return new_tools
