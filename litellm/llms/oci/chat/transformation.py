import base64
import datetime
import hashlib
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx

import litellm
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
    version,
)
from litellm.llms.oci.common_utils import OCIError
from litellm.types.llms.oci import (
    OCIChatRequestPayload,
    OCICompletionPayload,
    OCICompletionResponse,
    OCIContentPartUnion,
    OCIImageContentPart,
    OCIMessage,
    OCIRoles,
    OCIServingMode,
    OCIStreamChunk,
    OCITextContentPart,
    OCIToolCall,
    OCIToolDefinition,
    OCIVendors,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    Delta,
    LlmProviders,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.utils import (
    ChatCompletionMessageToolCall,
    CustomStreamWrapper,
    ModelResponse,
    Usage,
)

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
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError as e:
        raise ImportError(
            "cryptography package is required for OCI authentication. "
            "Please install it with: pip install cryptography"
        ) from e

    key = serialization.load_pem_private_key(
        key_str.encode("utf-8"),
        password=None,
    )
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(
            "The provided private key is not an RSA key, which is required for OCI signing."
        )
    return key


def load_private_key_from_file(file_path: str):
    """Loads a private key from a file path"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            key_str = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Private key file not found: {file_path}")
    except OSError as e:
        raise OSError(f"Failed to read private key file '{file_path}': {e}") from e

    if not key_str:
        raise ValueError(f"Private key file is empty: {file_path}")

    return load_private_key_from_str(key_str)


def get_vendor_from_model(model: str) -> OCIVendors:
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

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = []
        vendor = get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            raise ValueError(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
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
        vendor = get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            raise ValueError(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
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
        oci_key_file = optional_params.get("oci_key_file")

        if (
            not oci_user
            or not oci_fingerprint
            or not oci_tenancy
            or not (oci_key or oci_key_file)
        ):
            raise Exception(
                "Missing required parameters: oci_user, oci_fingerprint, oci_tenancy, "
                "and at least one of oci_key or oci_key_file."
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

        signed_headers = [
            "date",
            "(request-target)",
            "host",
            "content-length",
            "content-type",
            "x-content-sha256",
        ]
        signing_string = build_signature_string(
            method, path, headers_to_sign, signed_headers
        )

        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError as e:
            raise ImportError(
                "cryptography package is required for OCI authentication. "
                "Please install it with: pip install cryptography"
            ) from e

        private_key = (
            load_private_key_from_str(oci_key)
            if oci_key
            else load_private_key_from_file(oci_key_file) if oci_key_file else None
        )

        if private_key is None:
            raise Exception(
                "Private key is required for OCI authentication. Please provide either oci_key or oci_key_file."
            )

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

        headers.update(
            {
                "authorization": authorization,
                "date": date,
                "host": host,
                "content-type": content_type,
                "content-length": content_length,
                "x-content-sha256": x_content_sha256,
            }
        )

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
        oci_key_file = optional_params.get("oci_key_file")
        oci_compartment_id = optional_params.get("oci_compartment_id")

        if (
            not oci_user
            or not oci_fingerprint
            or not oci_tenancy
            or not (oci_key or oci_key_file)
            or not oci_compartment_id
        ):
            raise Exception(
                "Missing required parameters: oci_user, oci_fingerprint, oci_tenancy, oci_compartment_id "
                "and at least one of oci_key or oci_key_file."
            )

        if not api_base:
            raise Exception(
                "Either `api_base` must be provided or `litellm.api_base` must be set. Alternatively, you can set the `oci_region` optional parameter to use the default OCI region."
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
        oci_region = optional_params.get("oci_region", "us-ashburn-1")
        return f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com/20231130/actions/chat"

    def _get_optional_params(self, vendor: OCIVendors, optional_params: dict) -> Dict:
        selected_params = {}
        if vendor == OCIVendors.COHERE:
            raise ValueError(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        for value in open_ai_to_oci_param_map.values():
            if value in optional_params:
                selected_params[value] = optional_params[value]
        if "tools" in selected_params:
            selected_params["tools"] = adapt_tool_definition_to_oci_standard(
                selected_params["tools"], vendor
            )
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
            raise Exception("kwarg `oci_compartment_id` is required for OCI requests")

        vendor = get_vendor_from_model(model)

        if vendor == OCIVendors.COHERE:
            raise Exception(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
        else:
            data = OCICompletionPayload(
                compartmentId=oci_compartment_id,
                servingMode=OCIServingMode(
                    servingType="ON_DEMAND",
                    modelId=model,
                ),
                chatRequest=OCIChatRequestPayload(
                    apiFormat=vendor.value,
                    messages=adapt_messages_to_generic_oci_standard(messages),
                    **self._get_optional_params(vendor, optional_params),
                ),
            )

        return data.model_dump(exclude_none=True)

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

        if not isinstance(json, dict):
            raise OCIError(
                message="Invalid response format from OCI",
                status_code=raw_response.status_code,
            )

        try:
            completion_response = OCICompletionResponse(**json)
        except TypeError as e:
            raise OCIError(
                message=f"Response cannot be casted to OCICompletionResponse: {str(e)}",
                status_code=raw_response.status_code,
            )

        vendor = get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            raise ValueError(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
        else:
            iso_str = completion_response.chatResponse.timeCreated
            dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            model_response.created = int(dt.timestamp())

        model_response.model = completion_response.modelId

        message = model_response.choices[0].message  # type: ignore
        if vendor == OCIVendors.COHERE:
            raise ValueError(
                "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
            )
        else:
            response_message = completion_response.chatResponse.choices[0].message
            if response_message.content and response_message.content[0].type == "TEXT":
                message.content = response_message.content[0].text
            if response_message.toolCalls:
                message.tool_calls = adapt_tools_to_openai_standard(
                    response_message.toolCalls
                )

        usage = Usage(
            prompt_tokens=completion_response.chatResponse.usage.promptTokens,
            completion_tokens=completion_response.chatResponse.usage.completionTokens,
            total_tokens=completion_response.chatResponse.usage.totalTokens,
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
    ) -> "OCIStreamWrapper":
        if "stream" in data:
            del data["stream"]
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
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        completion_stream = response.iter_text()

        streaming_response = OCIStreamWrapper(
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
    ) -> "OCIStreamWrapper":
        if "stream" in data:
            del data["stream"]

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
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        completion_stream = response.aiter_text()

        async def split_chunks(completion_stream: AsyncIterator[str]):
            async for item in completion_stream:
                for chunk in item.split("\n\n"):
                    if not chunk:
                        continue
                    yield chunk.strip()

        streaming_response = OCIStreamWrapper(
            completion_stream=split_chunks(completion_stream),
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streaming_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OCIError(status_code=status_code, message=error_message)


open_ai_to_generic_oci_role_map: Dict[str, OCIRoles] = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
}


def adapt_messages_to_generic_oci_standard_content_message(
    role: str, content: Union[str, list]
) -> OCIMessage:
    new_content: List[OCIContentPartUnion] = []
    if isinstance(content, str):
        return OCIMessage(
            role=open_ai_to_generic_oci_role_map[role],
            content=[OCITextContentPart(text=content)],
            toolCalls=None,
            toolCallId=None,
        )

    # content is a list of content items:
    # [
    #     {"type": "text", "text": "Hello"},
    #     {"type": "image_url", "image_url": "https://example.com/image.png"}
    # ]
    for content_item in content:
        if not isinstance(content_item, dict):
            raise Exception("Each content item must be a dictionary")

        type = content_item.get("type")
        if not isinstance(type, str):
            raise Exception("Prop `type` is not a string")

        if type not in ["text", "image_url"]:
            raise Exception(f"Prop `{type}` is not supported")

        if type == "text":
            text = content_item.get("text")
            if not isinstance(text, str):
                raise Exception("Prop `text` is not a string")
            new_content.append(OCITextContentPart(text=text))

        elif type == "image_url":
            image_url = content_item.get("image_url")
            if not isinstance(image_url, str):
                raise Exception("Prop `image_url` is not a string")
            new_content.append(OCIImageContentPart(imageUrl=image_url))

    return OCIMessage(
        role=open_ai_to_generic_oci_role_map[role],
        content=new_content,
        toolCalls=None,
        toolCallId=None,
    )


def adapt_messages_to_generic_oci_standard_tool_call(
    role: str, tool_calls: list
) -> OCIMessage:
    tool_calls_formated = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise Exception("Each tool call must be a dictionary")

        if tool_call.get("type") != "function":
            raise Exception("OCI only supports function tools")

        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise Exception("Prop `id` is not a string")

        tool_function = tool_call.get("function")
        if not isinstance(tool_function, dict):
            raise Exception("Prop `function` is not a dictionary")

        function_name = tool_function.get("name")
        if not isinstance(function_name, str):
            raise Exception("Prop `name` is not a string")

        arguments = tool_call["function"].get("arguments", "{}")
        if not isinstance(arguments, str):
            raise Exception("Prop `arguments` is not a string")

        # tool_calls_formated.append(OCIToolCall(
        #     id=tool_call_id,
        #     type="FUNCTION",
        #     function=OCIFunction(
        #         name=function_name,
        #         arguments=arguments
        #     )
        # ))

        tool_calls_formated.append(
            OCIToolCall(
                id=tool_call_id,
                type="FUNCTION",
                name=function_name,
                arguments=arguments,
            )
        )

    return OCIMessage(
        role=open_ai_to_generic_oci_role_map[role],
        content=None,
        toolCalls=tool_calls_formated,
        toolCallId=None,
    )


def adapt_messages_to_generic_oci_standard_tool_response(
    role: str, tool_call_id: str, content: str
) -> OCIMessage:
    return OCIMessage(
        role=open_ai_to_generic_oci_role_map[role],
        content=[OCITextContentPart(text=content)],
        toolCalls=None,
        toolCallId=tool_call_id,
    )


def adapt_messages_to_generic_oci_standard(
    messages: List[AllMessageValues],
) -> List[OCIMessage]:
    new_messages = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")

        if role == "assistant" and tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise Exception("Prop `tool_calls` must be a list of tool calls")
            new_messages.append(
                adapt_messages_to_generic_oci_standard_tool_call(role, tool_calls)
            )

        elif role in ["system", "user", "assistant"] and content is not None:
            if not isinstance(content, (str, list)):
                raise Exception(
                    "Prop `content` must be a string or a list of content items"
                )
            new_messages.append(
                adapt_messages_to_generic_oci_standard_content_message(role, content)
            )

        elif role == "tool":
            if not isinstance(tool_call_id, str):
                raise Exception("Prop `tool_call_id` is required and must be a string")
            if not isinstance(content, str):
                raise Exception("Prop `content` is not a string")
            new_messages.append(
                adapt_messages_to_generic_oci_standard_tool_response(
                    role, tool_call_id, content
                )
            )

    return new_messages


def adapt_tool_definition_to_oci_standard(tools: List[Dict], vendor: OCIVendors):
    new_tools = []
    if vendor == OCIVendors.COHERE:
        raise ValueError(
            "Cohere models are not yet supported in the litellm OCI chat completion endpoint. Use the Cohere API directly."
        )
    else:
        for tool in tools:
            if tool["type"] != "function":
                raise Exception("OCI only supports function tools")

            tool_function = tool.get("function")
            if not isinstance(tool_function, dict):
                raise Exception("Prop `function` is not a dictionary")

            new_tool = OCIToolDefinition(
                type="FUNCTION",
                name=tool_function.get("name"),
                description=tool_function.get("description", ""),
                parameters=tool_function.get("parameters", {}),
            )
            new_tools.append(new_tool)

    return new_tools


def adapt_tools_to_openai_standard(
    tools: List[OCIToolCall],
) -> List[ChatCompletionMessageToolCall]:
    new_tools = []
    for tool in tools:
        new_tool = ChatCompletionMessageToolCall(
            id=tool.id,
            type="function",
            function={
                "name": tool.name,
                "arguments": tool.arguments,
            },
        )
        new_tools.append(new_tool)
    return new_tools


class OCIStreamWrapper(CustomStreamWrapper):
    """
    Custom stream wrapper for OCI responses.
    This class is used to handle streaming responses from OCI's API.
    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

    def chunk_creator(self, chunk: Any):
        if not isinstance(chunk, str):
            raise ValueError(f"Chunk is not a string: {chunk}")
        if not chunk.startswith("data:"):
            raise ValueError(f"Chunk does not start with 'data:': {chunk}")
        dict_chunk = json.loads(chunk[5:])  # Remove 'data: ' prefix and parse JSON
        try:
            typed_chunk = OCIStreamChunk(**dict_chunk)
        except TypeError as e:
            raise ValueError(f"Chunk cannot be casted to OCIStreamChunk: {str(e)}")

        if typed_chunk.index is None:
            typed_chunk.index = 0

        text = ""
        if typed_chunk.message and typed_chunk.message.content:
            for item in typed_chunk.message.content:
                if isinstance(item, OCITextContentPart):
                    text += item.text
                elif isinstance(item, OCIImageContentPart):
                    raise ValueError(
                        "OCI does not support image content in streaming responses"
                    )
                else:
                    raise ValueError(
                        f"Unsupported content type in OCI response: {item.type}"
                    )

        tool_calls = None
        if typed_chunk.message and typed_chunk.message.toolCalls:
            tool_calls = adapt_tools_to_openai_standard(typed_chunk.message.toolCalls)

        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=typed_chunk.index if typed_chunk.index else 0,
                    delta=Delta(
                        content=text,
                        tool_calls=(
                            [tool.model_dump() for tool in tool_calls]
                            if tool_calls
                            else None
                        ),
                        provider_specific_fields=None,  # OCI does not have provider specific fields in the response
                        thinking_blocks=None,  # OCI does not have thinking blocks in the response
                        reasoning_content=None,  # OCI does not have reasoning content in the response
                    ),
                    finish_reason=typed_chunk.finishReason,
                )
            ]
        )
