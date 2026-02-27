import base64
import datetime
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple, Union
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
    CohereChatRequest,
    CohereMessage,
    CohereChatResult,
    CohereParameterDefinition,
    CohereStreamChunk,
    CohereTool,
    CohereToolCall,
    OCIChatRequestPayload,
    OCICompletionPayload,
    OCICompletionResponse,
    OCIContentPartUnion,
    OCIImageContentPart,
    OCIImageUrl,
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
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.utils import (
    ChatCompletionMessageToolCall,
    CustomStreamWrapper,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OCISignerProtocol(Protocol):
    """
    Protocol for OCI request signers (e.g., oci.signer.Signer).

    This protocol defines the interface expected for OCI SDK signer objects.
    Compatible with the OCI Python SDK's Signer class.

    See: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html
    """

    def do_request_sign(self, request: Any, *, enforce_content_headers: bool = False) -> None:
        """
        Sign an HTTP request by adding authentication headers.

        Args:
            request: Request object with method, url, headers, body, and path_url attributes
            enforce_content_headers: Whether to enforce content-type and content-length headers
        """
        ...


@dataclass
class OCIRequestWrapper:
    """
    Wrapper for HTTP requests compatible with OCI signer interface.

    This class wraps request data in a format compatible with OCI SDK signers,
    which expect objects with method, url, headers, body, and path_url attributes.
    """
    method: str
    url: str
    headers: dict
    body: bytes

    @property
    def path_url(self) -> str:
        """Returns the path + query string for OCI signing."""
        parsed_url = urlparse(self.url)
        return parsed_url.path + ("?" + parsed_url.query if parsed_url.query else "")


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
            "response_format": "responseFormat",
        }

        # Cohere and Gemini use the same parameter mapping as GENERIC
        self.openai_to_oci_cohere_param_map = self.openai_to_oci_generic_param_map.copy()

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = []
        vendor = get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            open_ai_to_oci_param_map = self.openai_to_oci_cohere_param_map
            open_ai_to_oci_param_map.pop("tool_choice")
            open_ai_to_oci_param_map.pop("max_retries")
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
            open_ai_to_oci_param_map = self.openai_to_oci_cohere_param_map
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        all_params = {**non_default_params, **optional_params}

        for key, value in all_params.items():
            alias = open_ai_to_oci_param_map.get(key)

            if alias is False:
                # Workaround for mypy issue
                if drop_params or litellm.drop_params:
                    continue
                raise Exception(f"param `{key}` is not supported on OCI")

            if alias is None:
                adapted_params[key] = value
                continue

            adapted_params[alias] = value

            if alias == "responseFormat":
                adapted_params["response_format"] = value

        return adapted_params

    def _sign_with_oci_signer(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
    ) -> Tuple[dict, bytes]:
        """
        Sign request using OCI SDK Signer object.

        Args:
            headers: Request headers to be signed
            optional_params: Optional parameters including oci_signer
            request_data: The request body dict to be sent in HTTP request
            api_base: The complete URL for the HTTP request

        Returns:
            Tuple of (signed_headers, encoded_body)

        Raises:
            OCIError: If signing fails
            ValueError: If HTTP method is unsupported
        """
        oci_signer = optional_params.get("oci_signer")
        body = json.dumps(request_data).encode("utf-8")
        method = str(optional_params.get("method", "POST")).upper()

        if method not in ["POST", "GET", "PUT", "DELETE", "PATCH"]:
            raise ValueError(f"Unsupported HTTP method: {method}")

        prepared_headers = headers.copy()
        prepared_headers.setdefault("content-type", "application/json")
        prepared_headers.setdefault("content-length", str(len(body)))

        request_wrapper = OCIRequestWrapper(
            method=method,
            url=api_base,
            headers=prepared_headers,
            body=body
        )

        if oci_signer is None:
            raise ValueError("oci_signer cannot be None when calling _sign_with_oci_signer")

        try:
            oci_signer.do_request_sign(request_wrapper, enforce_content_headers=True)
        except Exception as e:
            raise OCIError(
                status_code=500,
                message=(
                    f"Failed to sign request with provided oci_signer: {str(e)}. "
                    "The signer must implement the OCI SDK Signer interface with a "
                    "do_request_sign(request, enforce_content_headers=True) method. "
                    "See: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html"
                )
            ) from e

        headers.update(request_wrapper.headers)
        return headers, body

    def _sign_with_manual_credentials(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
    ) -> Tuple[dict, None]:
        """
        Sign request using manual OCI credentials.

        Args:
            headers: Request headers to be signed
            optional_params: Optional parameters including OCI credentials
            request_data: The request body dict to be sent in HTTP request
            api_base: The complete URL for the HTTP request

        Returns:
            Tuple of (signed_headers, None)

        Raises:
            Exception: If required credentials are missing
            ImportError: If cryptography package is not installed
        """
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

        # Handle oci_key - it should be a string (PEM content)
        oci_key_content = None
        if oci_key:
            if isinstance(oci_key, str):
                oci_key_content = oci_key
                # Fix common issues with PEM content
                # Replace escaped newlines with actual newlines
                oci_key_content = oci_key_content.replace("\\n", "\n")
                # Ensure proper line endings
                if "\r\n" in oci_key_content:
                    oci_key_content = oci_key_content.replace("\r\n", "\n")
            else:
                raise OCIError(
                    status_code=400,
                    message=f"oci_key must be a string containing the PEM private key content. "
                    f"Got type: {type(oci_key).__name__}",
                )

        private_key = (
            load_private_key_from_str(oci_key_content)
            if oci_key_content
            else load_private_key_from_file(oci_key_file) if oci_key_file else None
        )

        if private_key is None:
            raise OCIError(
                status_code=400,
                message="Private key is required for OCI authentication. Please provide either oci_key or oci_key_file.",
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
        Sign the OCI request by adding authentication headers.

        Supports two signing modes:
        1. OCI SDK Signer: Use an oci_signer object to sign the request
        2. Manual Signing: Use OCI credentials to manually sign the request

        Args:
            headers: Request headers to be signed
            optional_params: Optional parameters including auth credentials or oci_signer
            request_data: The request body dict to be sent in HTTP request
            api_base: The complete URL for the HTTP request
            api_key: Optional API key (not used for OCI)
            model: Optional model name
            stream: Optional streaming flag
            fake_stream: Optional fake streaming flag

        Returns:
            Tuple of (signed_headers, encoded_body):
            - If oci_signer is provided: Returns (headers, body) where body is the encoded JSON
            - If manual credentials are provided: Returns (headers, None) as body is not returned
              for the manual signing path

        Raises:
            OCIError: If signing fails with oci_signer
            Exception: If required credentials are missing
            ImportError: If cryptography package is not installed (manual signing only)

        Example:
            >>> from oci.signer import Signer
            >>> signer = Signer(
            ...     tenancy="ocid1.tenancy.oc1..",
            ...     user="ocid1.user.oc1..",
            ...     fingerprint="xx:xx:xx",
            ...     private_key_file_location="~/.oci/key.pem"
            ... )
            >>> headers, body = config.sign_request(
            ...     headers={},
            ...     optional_params={"oci_signer": signer},
            ...     request_data={"message": "Hello"},
            ...     api_base="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/..."
            ... )
        """
        oci_signer = optional_params.get("oci_signer")

        # If a signer is provided, use it for request signing
        if oci_signer is not None:
            return self._sign_with_oci_signer(headers, optional_params, request_data, api_base)

        # Standard manual credential signing
        return self._sign_with_manual_credentials(headers, optional_params, request_data, api_base)

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
        """
        Validate the OCI environment and credentials.

        Supports two authentication modes:
        1. OCI SDK Signer: Pass an oci_signer object (e.g., oci.signer.Signer)
        2. Manual Credentials: Pass oci_user, oci_fingerprint, oci_tenancy, and oci_key/oci_key_file

        Args:
            headers: Request headers to populate
            model: Model name
            messages: List of chat messages
            optional_params: Optional parameters including authentication credentials
            litellm_params: LiteLLM parameters
            api_key: Optional API key (not used for OCI)
            api_base: Optional API base URL

        Returns:
            Updated headers dict

        Raises:
            Exception: If required parameters are missing or invalid
        """
        oci_signer = optional_params.get("oci_signer")
        oci_region = optional_params.get("oci_region", "us-ashburn-1")

        # Determine api_base
        api_base = (
            api_base
            or litellm.api_base
            or f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com"
        )

        if not api_base:
            raise Exception(
                "Either `api_base` must be provided or `litellm.api_base` must be set. "
                "Alternatively, you can set the `oci_region` optional parameter to use the default OCI region."
            )

        # Validate credentials only if signer is not provided
        if oci_signer is None:
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
                    "and at least one of oci_key or oci_key_file. "
                    "Alternatively, provide an oci_signer object from the OCI SDK."
                )

        # Common header setup
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
            open_ai_to_oci_param_map = self.openai_to_oci_cohere_param_map
            # remove tool_choice from the map
            open_ai_to_oci_param_map.pop("tool_choice")
            # Add default values for Cohere API
            selected_params = {
                "maxTokens": 600,
                "temperature": 1,
                "topK": 0,
                "topP": 0.75,
                "frequencyPenalty": 0
            }
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        # Map OpenAI params to OCI params
        for openai_key, oci_key in open_ai_to_oci_param_map.items():
            if oci_key and openai_key in optional_params:
                selected_params[oci_key] = optional_params[openai_key]  # type: ignore[index]

        # Also check for already-mapped OCI params (for backward compatibility)
        for oci_value in open_ai_to_oci_param_map.values():
            if oci_value and oci_value in optional_params and oci_value not in selected_params:
                selected_params[oci_value] = optional_params[oci_value]  # type: ignore[index]

        if "tools" in selected_params:
            if vendor == OCIVendors.COHERE:
                selected_params["tools"] = self.adapt_tool_definitions_to_cohere_standard(  # type: ignore[assignment]
                    selected_params["tools"]  # type: ignore[arg-type]
                )
            else:
                selected_params["tools"] = adapt_tool_definition_to_oci_standard(  # type: ignore[assignment]
                    selected_params["tools"], vendor  # type: ignore[arg-type]
                )

        # Transform response_format type to OCI uppercase format
        if "responseFormat" in selected_params:
            rf = selected_params["responseFormat"]
            if isinstance(rf, dict) and "type" in rf:
                rf_payload = dict(rf)
                selected_params["responseFormat"] = rf_payload

                response_type = rf_payload["type"]
                schema_payload: Optional[Any] = None

                if "json_schema" in rf_payload:
                    raw_schema_payload = rf_payload.pop("json_schema")
                    if isinstance(raw_schema_payload, dict):
                        schema_payload = dict(raw_schema_payload)
                    else:
                        schema_payload = raw_schema_payload

                if schema_payload is not None:
                    rf_payload["jsonSchema"] = schema_payload

                if vendor == OCIVendors.COHERE:
                    # Cohere expects lower-case type values
                    rf_payload["type"] = response_type
                else:
                    format_type = response_type.upper()
                    if format_type == "JSON":
                        format_type = "JSON_OBJECT"
                    rf_payload["type"] = format_type

        return selected_params

    def adapt_messages_to_cohere_standard(self, messages: List[AllMessageValues]) -> List[CohereMessage]:
        """Build chat history for Cohere models."""
        chat_history = []
        for msg in messages[:-1]:  # All messages except the last one
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, list):
                # Extract text from content array
                text_content = ""
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_content += content_item.get("text", "")
                content = text_content

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            # Handle tool calls
            tool_calls: Optional[List[CohereToolCall]] = None
            if role == "assistant" and "tool_calls" in msg and msg.get("tool_calls"):  # type: ignore[union-attr,typeddict-item]
                tool_calls = []
                for tool_call in msg["tool_calls"]:  # type: ignore[union-attr,typeddict-item]
                    # Parse arguments if they're a JSON string
                    raw_arguments: Any = tool_call.get("function", {}).get("arguments", {})
                    if isinstance(raw_arguments, str):
                        try:
                            arguments: Dict[str, Any] = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = raw_arguments

                    tool_calls.append(CohereToolCall(
                        name=str(tool_call.get("function", {}).get("name", "")),
                        parameters=arguments
                    ))

            if role == "user":
                chat_history.append(CohereMessage(role="USER", message=content))
            elif role == "assistant":
                chat_history.append(CohereMessage(role="CHATBOT", message=content, toolCalls=tool_calls))
            elif role == "tool":
                # Tool messages need special handling
                chat_history.append(CohereMessage(
                    role="TOOL",
                    message=content,
                    toolCalls=None  # Tool messages don't have tool calls
                ))

        return chat_history

    def adapt_tool_definitions_to_cohere_standard(self, tools: List[Dict[str, Any]]) -> List[CohereTool]:
        """Adapt tool definitions to Cohere format."""
        cohere_tools = []
        for tool in tools:
            function_def = tool.get("function", {})
            parameters = function_def.get("parameters", {}).get("properties", {})
            required = function_def.get("parameters", {}).get("required", [])

            parameter_definitions = {}
            for param_name, param_schema in parameters.items():
                parameter_definitions[param_name] = CohereParameterDefinition(
                    description=param_schema.get("description", ""),
                    type=param_schema.get("type", "string"),
                    isRequired=param_name in required
                )

            cohere_tools.append(CohereTool(
                name=function_def.get("name", ""),
                description=function_def.get("description", ""),
                parameterDefinitions=parameter_definitions
            ))

        return cohere_tools

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from message content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_content = ""
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_content += content_item.get("text", "")
            return text_content
        return str(content)

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

        oci_serving_mode = optional_params.get("oci_serving_mode", "ON_DEMAND")
        if oci_serving_mode not in ["ON_DEMAND", "DEDICATED"]:
            raise Exception(
                "kwarg `oci_serving_mode` must be either 'ON_DEMAND' or 'DEDICATED'"
            )

        if oci_serving_mode == "DEDICATED":
            oci_endpoint_id = optional_params.get("oci_endpoint_id", model)
            servingMode = OCIServingMode(
                servingType="DEDICATED",
                endpointId=oci_endpoint_id,
            )
        else:
            servingMode = OCIServingMode(
                servingType="ON_DEMAND",
                modelId=model,
            )

        # Build request based on vendor type
        if vendor == OCIVendors.COHERE:
            # For Cohere, we need to use the specific Cohere format
            # Extract the last user message as the main message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                raise Exception("No user message found for Cohere model")

            # Extract system messages into preambleOverride
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            preamble_override = None
            if system_messages:
                preamble = "\n".join(
                    self._extract_text_content(msg["content"]) for msg in system_messages
                )
                if preamble:
                    preamble_override = preamble

            # Create Cohere-specific chat request
            optional_cohere_params = self._get_optional_params(OCIVendors.COHERE, optional_params)
            chat_request = CohereChatRequest(
                apiFormat="COHERE",
                message=self._extract_text_content(user_messages[-1]["content"]),
                chatHistory=self.adapt_messages_to_cohere_standard(messages),
                preambleOverride=preamble_override,
                **optional_cohere_params
            )

            data = OCICompletionPayload(
                compartmentId=oci_compartment_id,
                servingMode=servingMode,
                chatRequest=chat_request
            )
        else:
            # Use generic format for other vendors
            data = OCICompletionPayload(
                compartmentId=oci_compartment_id,
                servingMode=servingMode,
                chatRequest=OCIChatRequestPayload(
                    apiFormat=vendor.value,
                    messages=adapt_messages_to_generic_oci_standard(messages),
                    **self._get_optional_params(vendor, optional_params),
                ),
            )

        return data.model_dump(exclude_none=True)

    def _handle_cohere_response(
        self,
        json_response: dict,
        model: str,
        model_response: ModelResponse
    ) -> ModelResponse:
        """Handle Cohere-specific response format."""
        cohere_response = CohereChatResult(**json_response)
        # Cohere response format (uses camelCase)
        model_id = model

        # Set basic response info
        model_response.model = model_id
        model_response.created = int(datetime.datetime.now().timestamp())

        # Extract the response text
        response_text = cohere_response.chatResponse.text
        oci_finish_reason = cohere_response.chatResponse.finishReason

        # Map finish reason
        if oci_finish_reason == "COMPLETE":
            finish_reason = "stop"
        elif oci_finish_reason == "MAX_TOKENS":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        # Handle tool calls
        tool_calls: Optional[List[Dict[str, Any]]] = None
        if cohere_response.chatResponse.toolCalls:
            tool_calls = []
            for tool_call in cohere_response.chatResponse.toolCalls:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",  # Generate a simple ID
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.parameters)
                    }
                })

        # Create choice
        from litellm.types.utils import Choices
        choice = Choices(
            index=0,
            message={
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls
            },
            finish_reason=finish_reason
        )
        model_response.choices = [choice]

        # Extract usage info
        usage_info = cohere_response.chatResponse.usage
        from litellm.types.utils import Usage
        model_response.usage = Usage(  # type: ignore[attr-defined]
            prompt_tokens=usage_info.promptTokens,  # type: ignore[union-attr]
            completion_tokens=usage_info.completionTokens,  # type: ignore[union-attr]
            total_tokens=usage_info.totalTokens  # type: ignore[union-attr]
        )

        return model_response

    def _handle_generic_response(
        self,
        json: dict,
        model: str,
        model_response: ModelResponse,
        raw_response: httpx.Response
    ) -> ModelResponse:
        """Handle generic OCI response format."""
        try:
            completion_response = OCICompletionResponse(**json)
        except TypeError as e:
            raise OCIError(
                message=f"Response cannot be casted to OCICompletionResponse: {str(e)}",
                status_code=raw_response.status_code,
            )

        iso_str = completion_response.chatResponse.timeCreated
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        model_response.created = int(dt.timestamp())

        model_response.model = completion_response.modelId

        message = model_response.choices[0].message  # type: ignore
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

        return model_response

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

        vendor = get_vendor_from_model(model)

        # Handle response based on vendor type
        if vendor == OCIVendors.COHERE:
            model_response = self._handle_cohere_response(json, model, model_response)
        else:
            model_response = self._handle_generic_response(json, model, model_response, raw_response)

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
            # Handle both OpenAI format (object with url) and string format
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str):
                raise Exception("Prop `image_url` must be a string or an object with a `url` property")
            new_content.append(OCIImageContentPart(imageUrl=OCIImageUrl(url=image_url)))

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

        # Check if this is a Cohere stream chunk
        if "apiFormat" in dict_chunk and dict_chunk.get("apiFormat") == "COHERE":
            return self._handle_cohere_stream_chunk(dict_chunk)
        else:
            return self._handle_generic_stream_chunk(dict_chunk)

    def _handle_cohere_stream_chunk(self, dict_chunk: dict):
        """Handle Cohere-specific streaming chunks."""
        try:
            typed_chunk = CohereStreamChunk(**dict_chunk)
        except TypeError as e:
            raise ValueError(f"Chunk cannot be casted to CohereStreamChunk: {str(e)}")

        if typed_chunk.index is None:
            typed_chunk.index = 0

        # Extract text content
        text = typed_chunk.text or ""

        # Map finish reason to standard format
        finish_reason = typed_chunk.finishReason
        if finish_reason == "COMPLETE":
            finish_reason = "stop"
        elif finish_reason == "MAX_TOKENS":
            finish_reason = "length"
        elif finish_reason is None:
            finish_reason = None
        else:
            finish_reason = "stop"

        # For Cohere, we don't have tool calls in the streaming format
        tool_calls = None

        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=typed_chunk.index if typed_chunk.index else 0,
                    delta=Delta(
                        content=text,
                        tool_calls=tool_calls,
                        provider_specific_fields=None,
                        thinking_blocks=None,
                        reasoning_content=None,
                    ),
                    finish_reason=finish_reason,
                )
            ]
        )

    def _handle_generic_stream_chunk(self, dict_chunk: dict):
        """Handle generic OCI streaming chunks."""
        # Fix missing required fields in tool calls before Pydantic validation
        # OCI streams tool calls progressively, so early chunks may be missing required fields
        if dict_chunk.get("message") and dict_chunk["message"].get("toolCalls"):
            for tool_call in dict_chunk["message"]["toolCalls"]:
                if "arguments" not in tool_call:
                    tool_call["arguments"] = ""
                if "id" not in tool_call:
                    tool_call["id"] = ""
                if "name" not in tool_call:
                    tool_call["name"] = ""

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
