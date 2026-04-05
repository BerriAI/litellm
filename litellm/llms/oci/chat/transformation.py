import datetime
import json
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

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
from litellm.llms.oci.common_utils import (
    OCIError,
    OCIRequestWrapper,  # re-exported for backwards compatibility
    get_oci_base_url,
    resolve_oci_credentials,
    sign_oci_request,
    validate_oci_environment,
)
from litellm.types.llms.oci import (
    CohereChatRequest,
    CohereChatResult,
    CohereMessage,
    CohereParameterDefinition,
    CohereStreamChunk,
    CohereTool,
    CohereToolCall,
    CohereToolMessage,
    CohereToolResult,
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
    Choices,
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


def get_vendor_from_model(model: str) -> OCIVendors:
    """
    Extracts the vendor from the model name.

    OCI GenAI API uses two apiFormat values:
    - "COHERE" for Cohere models (command-r, command-a, etc.)
    - "GENERIC" for all other models (Meta Llama, xAI Grok, Google Gemini, etc.)

    Args:
        model (str): The model name (e.g., "cohere.command-a-03-2025", "meta.llama-3.3-70b-instruct").
    Returns:
        OCIVendors: The vendor enum value.
    """
    vendor = model.split(".")[0].lower()
    if vendor == "cohere":
        return OCIVendors.COHERE
    return OCIVendors.GENERIC


# OCI GenAI REST API version — stable since service launch, unlikely to change
OCI_API_VERSION = "20231130"

# Streaming timeout — generous because OCI models may need to warm up on first request
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

        # Cohere uses the same parameter keys as GENERIC with three differences:
        # - tool_choice is unsupported
        # - stop sequences are named "stopSequences" not "stop"
        # - n (numGenerations) is a GenericChatRequest-only field; CohereChatRequest has no equivalent
        # Build a *separate* frozen reference map so callers never mutate the canonical dict.
        self._openai_to_oci_cohere_param_map = {
            k: ("stopSequences" if k == "stop" else v)
            for k, v in self.openai_to_oci_generic_param_map.items()
            if k not in ("tool_choice", "max_retries", "n")
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        param_map = (
            self._openai_to_oci_cohere_param_map
            if get_vendor_from_model(model) == OCIVendors.COHERE
            else self.openai_to_oci_generic_param_map
        )
        return [key for key, value in param_map.items() if value]

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
            open_ai_to_oci_param_map = self._openai_to_oci_cohere_param_map
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        all_params = {**non_default_params, **optional_params}

        for key, value in all_params.items():
            alias = open_ai_to_oci_param_map.get(key)

            if alias is False:
                # Workaround for mypy issue
                if drop_params or litellm.drop_params:
                    continue
                raise OCIError(
                    status_code=400,
                    message=f"param `{key}` is not supported on OCI",
                )

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
            method=method, url=api_base, headers=prepared_headers, body=body
        )

        if oci_signer is None:
            raise ValueError(
                "oci_signer cannot be None when calling _sign_with_oci_signer"
            )

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
                ),
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

        1. **OCI SDK Signer** — pass ``oci_signer`` (an ``oci.signer.Signer`` instance or any
           object implementing :class:`~litellm.llms.oci.common_utils.OCISignerProtocol`).
        2. **Manual RSA-SHA256** — pass ``oci_user``, ``oci_fingerprint``, ``oci_tenancy``, and
           ``oci_key`` (PEM string) or ``oci_key_file`` (path).  All of these can also be
           supplied via ``OCI_USER``, ``OCI_FINGERPRINT``, ``OCI_TENANCY``, and
           ``OCI_KEY_FILE`` environment variables.
        """
        return sign_oci_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

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
        if not messages:
            raise OCIError(
                status_code=400,
                message="kwarg `messages` must be an array of messages that follow the openai chat standard",
            )
        # Validate credentials early so the caller gets a clear error immediately
        # rather than a cryptic signing failure at request time.
        # Credentials may come from optional_params or OCI_* env vars.
        if optional_params.get("oci_signer") is None:
            creds = resolve_oci_credentials(optional_params)
            missing = [
                k
                for k in ("oci_user", "oci_fingerprint", "oci_tenancy", "oci_compartment_id")
                if not creds.get(k)
            ]
            if missing or not (creds.get("oci_key") or creds.get("oci_key_file")):
                raise OCIError(
                    status_code=401,
                    message=(
                        "Missing required parameters: oci_user, oci_fingerprint, oci_tenancy, oci_compartment_id "
                        "and at least one of oci_key or oci_key_file. "
                        "These can be supplied via optional_params or via OCI_USER, OCI_FINGERPRINT, "
                        "OCI_TENANCY, OCI_COMPARTMENT_ID, OCI_KEY_FILE environment variables. "
                        "Alternatively, provide an oci_signer object from the OCI SDK."
                    ),
                )
        return validate_oci_environment(headers, optional_params, api_key)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = get_oci_base_url(optional_params, api_base or litellm.api_base)
        return f"{base}/{OCI_API_VERSION}/actions/chat"

    def _get_optional_params(self, vendor: OCIVendors, optional_params: dict) -> Dict:
        selected_params: Dict = {}
        if vendor == OCIVendors.COHERE:
            open_ai_to_oci_param_map = self._openai_to_oci_cohere_param_map
        else:
            open_ai_to_oci_param_map = self.openai_to_oci_generic_param_map

        # Map OpenAI params to OCI params
        for openai_key, oci_key in open_ai_to_oci_param_map.items():
            if oci_key and openai_key in optional_params:
                selected_params[oci_key] = optional_params[openai_key]  # type: ignore[index]

        # Also check for already-mapped OCI params (for backward compatibility)
        for oci_value in open_ai_to_oci_param_map.values():
            if (
                oci_value
                and oci_value in optional_params
                and oci_value not in selected_params
            ):
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

    def adapt_messages_to_cohere_standard(
        self, messages: List[AllMessageValues]
    ) -> List[CohereMessage]:
        """Build chat history for Cohere models.

        Tool results are represented as OCI Cohere ``toolResults`` entries, where each
        entry carries the originating tool call (name + parameters resolved from the
        preceding assistant message) and the output text.
        """
        # First pass: build tool_call_id -> CohereToolCall lookup so tool-result
        # messages can reference the originating call by name and parameters.
        tool_call_lookup: Dict[str, CohereToolCall] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:  # type: ignore[union-attr]
                    tc_id = tc.get("id", "")
                    raw_args: Any = tc.get("function", {}).get("arguments", "{}")
                    try:
                        params: Dict[str, Any] = (
                            json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        )
                    except json.JSONDecodeError:
                        params = {}
                    tool_call_lookup[tc_id] = CohereToolCall(
                        name=str(tc.get("function", {}).get("name", "")),
                        parameters=params,
                    )

        chat_history: List[CohereMessage] = []
        for msg in messages[:-1]:  # All messages except the last one
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, list):
                # Extract text from content array
                text_content = ""
                for content_item in content:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "text"
                    ):
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
                    raw_arguments: Any = tool_call.get("function", {}).get(
                        "arguments", {}
                    )
                    if isinstance(raw_arguments, str):
                        try:
                            arguments: Dict[str, Any] = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = raw_arguments

                    tool_calls.append(
                        CohereToolCall(
                            name=str(tool_call.get("function", {}).get("name", "")),
                            parameters=arguments,
                        )
                    )

            if role == "user":
                chat_history.append(CohereMessage(role="USER", message=content))
            elif role == "assistant":
                chat_history.append(
                    CohereMessage(role="CHATBOT", message=content, toolCalls=tool_calls)
                )
            elif role == "tool":
                # Construct a proper OCI Cohere tool-result message.
                # The API expects toolResults with the originating call (name + params)
                # and a list of output objects — not a flat toolCallId string.
                tool_call_id = msg.get("tool_call_id", "")  # type: ignore[union-attr]
                cohere_call = tool_call_lookup.get(
                    tool_call_id,
                    CohereToolCall(name="", parameters={}),
                )
                chat_history.append(
                    CohereToolMessage(
                        toolResults=[
                            CohereToolResult(
                                call=cohere_call,
                                outputs=[{"result": content}],
                            )
                        ]
                    )
                )

        return chat_history

    def adapt_tool_definitions_to_cohere_standard(
        self, tools: List[Dict[str, Any]]
    ) -> List[CohereTool]:
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
                    isRequired=param_name in required,
                )

            cohere_tools.append(
                CohereTool(
                    name=function_def.get("name", ""),
                    description=function_def.get("description", ""),
                    parameterDefinitions=parameter_definitions,
                )
            )

        return cohere_tools

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from message content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_content = ""
            for content_item in content:
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "text"
                ):
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
        creds = resolve_oci_credentials(optional_params)
        oci_compartment_id = creds["oci_compartment_id"]
        if not oci_compartment_id:
            raise OCIError(
                status_code=400,
                message=(
                    "oci_compartment_id is required for OCI chat requests. "
                    "Pass it as optional_params or set the OCI_COMPARTMENT_ID env var."
                ),
            )

        vendor = get_vendor_from_model(model)

        oci_serving_mode = optional_params.get("oci_serving_mode", "ON_DEMAND")
        if oci_serving_mode not in ["ON_DEMAND", "DEDICATED"]:
            raise OCIError(
                status_code=400,
                message="kwarg `oci_serving_mode` must be either 'ON_DEMAND' or 'DEDICATED'",
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
                raise OCIError(
                    status_code=400,
                    message="No user message found — Cohere models require at least one user message",
                )

            # Extract system messages into preambleOverride
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            preamble_override = None
            if system_messages:
                preamble = "\n".join(
                    self._extract_text_content(msg["content"])
                    for msg in system_messages
                )
                if preamble:
                    preamble_override = preamble

            # Create Cohere-specific chat request
            optional_cohere_params = self._get_optional_params(
                OCIVendors.COHERE, optional_params
            )
            chat_request = CohereChatRequest(
                apiFormat="COHERE",
                message=self._extract_text_content(user_messages[-1]["content"]),
                chatHistory=self.adapt_messages_to_cohere_standard(messages),
                preambleOverride=preamble_override,
                **optional_cohere_params,
            )

            data = OCICompletionPayload(
                compartmentId=oci_compartment_id,
                servingMode=servingMode,
                chatRequest=chat_request,
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
        self, json_response: dict, model: str, model_response: ModelResponse
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

        # Map finish reason — pass through unknown reasons rather than silently mapping to "stop"
        if oci_finish_reason == "COMPLETE":
            finish_reason = "stop"
        elif oci_finish_reason == "MAX_TOKENS":
            finish_reason = "length"
        elif oci_finish_reason == "TOOL_CALL":
            finish_reason = "tool_calls"
        else:
            finish_reason = oci_finish_reason  # preserve unknown reasons as-is

        # Handle tool calls
        tool_calls: Optional[List[Dict[str, Any]]] = None
        if cohere_response.chatResponse.toolCalls:
            tool_calls = [
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.parameters),
                    },
                }
                for tool_call in cohere_response.chatResponse.toolCalls
            ]

        # Create choice
        choice = Choices(
            index=0,
            message={
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls,
            },
            finish_reason=finish_reason,
        )
        model_response.choices = [choice]

        # Extract usage info
        usage_info = cohere_response.chatResponse.usage
        model_response.usage = Usage(  # type: ignore[attr-defined]
            prompt_tokens=usage_info.promptTokens,  # type: ignore[union-attr]
            completion_tokens=usage_info.completionTokens,  # type: ignore[union-attr]
            total_tokens=usage_info.totalTokens,  # type: ignore[union-attr]
        )

        return model_response

    def _handle_generic_response(
        self,
        json: dict,
        model: str,
        model_response: ModelResponse,
        raw_response: httpx.Response,
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
        # message is None when a reasoning model spends all max_tokens on reasoning
        if response_message is not None:
            if response_message.content and len(response_message.content) > 0 and response_message.content[0].type == "TEXT":
                message.content = response_message.content[0].text
            if response_message.toolCalls:
                message.tool_calls = adapt_tools_to_openai_standard(
                    response_message.toolCalls
                )

        oci_usage = completion_response.chatResponse.usage
        usage = Usage(
            prompt_tokens=oci_usage.promptTokens,
            completion_tokens=oci_usage.completionTokens or 0,
            total_tokens=oci_usage.totalTokens,
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
            model_response = self._handle_generic_response(
                json, model, model_response, raw_response
            )

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
                data=signed_json_body if signed_json_body is not None else json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        def split_chunks(stream: Iterator[str]) -> Iterator[str]:
            """SSE events are separated by \\n\\n — yield one data line at a time."""
            for item in stream:
                for chunk in item.split("\n\n"):
                    stripped = chunk.strip()
                    if stripped:
                        yield stripped

        streaming_response = OCIStreamWrapper(
            completion_stream=split_chunks(response.iter_text()),
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
            client = get_async_httpx_client(llm_provider=LlmProviders.OCI, params={})

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=signed_json_body if signed_json_body is not None else json.dumps(data),
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
                    stripped = chunk.strip()
                    if stripped:
                        yield stripped

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
            raise OCIError(status_code=400, message="Each content item must be a dictionary")

        type = content_item.get("type")
        if not isinstance(type, str):
            raise OCIError(status_code=400, message="Each content item must have a string `type` field")

        if type not in ["text", "image_url"]:
            raise OCIError(status_code=400, message=f"Content type `{type}` is not supported by OCI")

        if type == "text":
            text = content_item.get("text")
            if not isinstance(text, str):
                raise OCIError(status_code=400, message="Content item of type `text` must have a string `text` field")
            new_content.append(OCITextContentPart(text=text))

        elif type == "image_url":
            image_url = content_item.get("image_url")
            # Handle both OpenAI format (object with url) and string format
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str):
                raise OCIError(
                    status_code=400,
                    message="Prop `image_url` must be a string or an object with a `url` property",
                )
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
            raise OCIError(status_code=400, message="Each tool call must be a dictionary")

        if tool_call.get("type") != "function":
            raise OCIError(status_code=400, message="OCI only supports function tool calls")

        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise OCIError(status_code=400, message="Tool call `id` must be a string")

        tool_function = tool_call.get("function")
        if not isinstance(tool_function, dict):
            raise OCIError(status_code=400, message="Tool call `function` must be a dictionary")

        function_name = tool_function.get("name")
        if not isinstance(function_name, str):
            raise OCIError(status_code=400, message="Tool call `function.name` must be a string")

        arguments = tool_call["function"].get("arguments", "{}")
        if not isinstance(arguments, str):
            raise OCIError(status_code=400, message="Tool call `function.arguments` must be a JSON string")

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
                raise OCIError(status_code=400, message="Message `tool_calls` must be a list")
            new_messages.append(
                adapt_messages_to_generic_oci_standard_tool_call(role, tool_calls)
            )

        elif role in ["system", "user", "assistant"] and content is not None:
            if not isinstance(content, (str, list)):
                raise OCIError(
                    status_code=400,
                    message="Message `content` must be a string or list of content parts",
                )
            new_messages.append(
                adapt_messages_to_generic_oci_standard_content_message(role, content)
            )

        elif role == "tool":
            if not isinstance(tool_call_id, str):
                raise OCIError(status_code=400, message="Tool result message must have a string `tool_call_id`")
            if not isinstance(content, str):
                raise OCIError(status_code=400, message="Tool result message `content` must be a string")
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
            raise OCIError(status_code=400, message="OCI only supports function tools")

        tool_function = tool.get("function")
        if not isinstance(tool_function, dict):
            raise OCIError(status_code=400, message="Tool `function` must be a dictionary")

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
            id=tool.id or f"call_{uuid.uuid4().hex[:24]}",
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
            raise OCIError(
                status_code=500,
                message=f"Chunk cannot be parsed as CohereStreamChunk: {str(e)}",
            )

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
        elif finish_reason == "TOOL_CALL":
            finish_reason = "tool_calls"
        # None → streaming in progress; unknown → pass through as-is

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
            raise OCIError(
                status_code=500,
                message=f"Chunk cannot be parsed as OCIStreamChunk: {str(e)}",
            )

        if typed_chunk.index is None:
            typed_chunk.index = 0

        text = ""
        if typed_chunk.message and typed_chunk.message.content:
            for item in typed_chunk.message.content:
                if isinstance(item, OCITextContentPart):
                    text += item.text
                elif isinstance(item, OCIImageContentPart):
                    raise OCIError(
                        status_code=500,
                        message="OCI returned image content in a streaming response — not supported",
                    )
                else:
                    raise OCIError(
                        status_code=500,
                        message=f"Unsupported content type in OCI streaming response: {item.type}",
                    )

        tool_calls = None
        if typed_chunk.message and typed_chunk.message.toolCalls:
            tool_calls = adapt_tools_to_openai_standard(typed_chunk.message.toolCalls)

        # Map OCI finish reasons to OpenAI convention (same as Cohere path)
        oci_finish_reason = typed_chunk.finishReason
        if oci_finish_reason == "COMPLETE":
            finish_reason: Optional[str] = "stop"
        elif oci_finish_reason == "MAX_TOKENS":
            finish_reason = "length"
        elif oci_finish_reason == "TOOL_CALLS":
            finish_reason = "tool_calls"
        else:
            finish_reason = oci_finish_reason  # None while streaming; unknown passed through

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
                    finish_reason=finish_reason,
                )
            ]
        )
