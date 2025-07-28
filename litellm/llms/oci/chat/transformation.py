import base64
import datetime
import hashlib
from urllib.parse import urlparse
import litellm
import json
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import base64
import hashlib
import datetime
from urllib.parse import urlparse
from typing import Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import httpx

from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
    version,
)
from litellm.types.llms.openai import AllMessageValues
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

        self.openai_to_oci_param_map = {
            "stream": "stream",
            "max_tokens": "max_tokens",
            "max_completion_tokens": "max_tokens",
            "temperature": "temperature",
            "tools": "tools",
            # "top_p": "top_p",
            # "n": "num_return_sequences",
            # "max_retries": "max_retries",
            "top_p": False,
            "n": False,
            "max_retries": False,
            "seed": False,  # TODO requires backend changes
            "stop": False,  # TODO requires backend changes
            "logit_bias": False,  # TODO requires backend changes
            "logprobs": False,  # TODO requires backend changes
            "frequency_penalty": False,
            "presence_penalty": False,
            "top_logprobs": False,
            "modalities": False,
            "prediction": False,
            "stream_options": False,
            "tool_choice": False,
            "function_call": False,
            "functions": False,
            "extra_headers": False,
            "parallel_tool_calls": False,
            "audio": False,
            "web_search_options": False,
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = []
        for key, value in self.openai_to_oci_param_map.items():
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

        all_params = {**non_default_params, **optional_params}

        for key, value in all_params.items():

            alias = self.openai_to_oci_param_map.get(key)

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

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream = optional_params.get("stream", False)
        oci_compartment_id = optional_params.get("oci_compartment_id", None)
        if not oci_compartment_id:
            raise Exception(
                "kwarg `oci_compartment_id` is required for OCI requests"
            )
        temperature = optional_params.get("temperature", None)
        max_tokens = optional_params.get("max_tokens", None)
        tools = optional_params.get("tools", None)

        # we add stream not as an additional param, but as a primary prop on the request body, this is always defined if stream == True
        if optional_params.get("stream"):
            del optional_params["stream"]

        messages = adapt_messages_to_oci_standard(messages=messages)  # type: ignore

        data = {
            "compartmentId": oci_compartment_id,
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model,
            },
            "chatRequest": {
                "apiFormat": "GENERIC",
                "messages": messages,
            },
        }
        if temperature:
            data["chatRequest"]["temperature"] = temperature
        if stream:
            data["chatRequest"]["isStream"] = True
        if max_tokens:
            data["chatRequest"]["maxTokens"] = max_tokens
        if tools:
            data["chatRequest"]["tools"] = adapt_tools_to_oci_standard(tools)

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

        # set meta data here
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
    ) -> "BytezCustomStreamWrapper":
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

        streaming_response = BytezCustomStreamWrapper(
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
    ) -> "BytezCustomStreamWrapper":
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

        streaming_response = BytezCustomStreamWrapper(
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


class BytezCustomStreamWrapper(CustomStreamWrapper):
    def chunk_creator(self, chunk: Any):
        try:
            model_response = self.model_response_creator()
            response_obj: Dict[str, Any] = {}

            response_obj = {
                "text": chunk,
                "is_finished": False,
                "finish_reason": "",
            }

            completion_obj: Dict[str, Any] = {"content": chunk}

            return self.return_processed_chunk_logic(
                completion_obj=completion_obj,
                model_response=model_response,  # type: ignore
                response_obj=response_obj,
            )

        except StopIteration:
            raise StopIteration
        except Exception as e:
            traceback.format_exc()
            setattr(e, "message", str(e))
            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                original_exception=e,
            )


# litellm/types/llms/openai.py is a good reference for what is supported
open_ai_to_oci_content_item_map = {
    "text": {"type": "TEXT", "value_name": "text"},
    "image_url": {"type": "image", "value_name": "url"},
    "input_audio": {"type": "audio", "value_name": "url"},
    "video_url": {"type": "video", "value_name": "url"},
    "document": None,
    "file": None,
}

open_ai_to_oci_role_map = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
}

def adapt_messages_to_oci_standard(messages: List[Dict]):
    messages = _adapt_string_only_content_to_lists(messages)
    new_messages = []

    for message in messages:
        role = message["role"]
        content: list = message["content"]
        new_content = []

        for content_item in content:
            type: Union[str, None] = content_item.get("type")

            if not type:
                raise Exception("Prop `type` is not a string")

            content_item_map = open_ai_to_oci_content_item_map[type]

            if not content_item_map:
                raise Exception(f"Prop `{type}` is not supported")

            new_type = content_item_map["type"]

            value_name = content_item_map["value_name"]

            value: Union[str, None] = content_item.get(value_name)

            if not value:
                raise Exception(f"Prop `{value_name}` is not a string")

            new_content.append({"type": new_type, value_name: value})

        new_messages.append({"role": open_ai_to_oci_role_map[role], "content": new_content})

    return new_messages


# "content": "The cat ran so fast"
# becomes
# "content": [{"type": "text", "text": "The cat ran so fast"}]
def _adapt_string_only_content_to_lists(messages: List[Dict]):
    new_messages = []

    for message in messages:

        role = message.get("role")
        content = message.get("content")

        new_content = []

        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})

        elif isinstance(content, dict):
            new_content.append(content)

        elif isinstance(content, list):

            new_content_items = []
            for content_item in content:
                if isinstance(content_item, str):
                    new_content_items.append({"type": "text", "text": content_item})
                elif isinstance(content_item, dict):
                    new_content_items.append(content_item)
                else:
                    raise Exception(
                        "`content` can only contain strings or openai content dicts"
                    )

            new_content += new_content_items
        else:
            raise Exception("Content must be a string")

        new_messages.append({"role": role, "content": new_content})

    return new_messages


def adapt_tools_to_oci_standard(tools: List[Dict]):
    new_tools = []
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
