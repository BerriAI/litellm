"""
OCI Generative AI — chat transformation orchestrator.

This module wires together the Cohere-specific and Generic-model helpers to
implement the LiteLLM BaseConfig interface.  Heavy-lifting lives in:

  - :mod:`litellm.llms.oci.chat.cohere`  — Cohere message/tool/response logic
  - :mod:`litellm.llms.oci.chat.generic` — Generic message/tool/response logic
  - :mod:`litellm.llms.oci.common_utils` — auth, signing, schema utilities
"""

import json
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
from litellm.llms.oci.chat.cohere import (
    _extract_text_content,
    adapt_messages_to_cohere_standard,
    adapt_tool_definitions_to_cohere_standard,
    handle_cohere_response,
    handle_cohere_stream_chunk,
)
from litellm.llms.oci.chat.generic import (
    adapt_messages_to_generic_oci_standard,
    adapt_tool_definition_to_oci_standard,
    handle_generic_response,
    handle_generic_stream_chunk,
)
from litellm.llms.oci.common_utils import (
    OCI_API_VERSION,
    OCIError,
    OCIRequestWrapper,  # re-exported for backwards compatibility
    get_oci_base_url,
    resolve_oci_credentials,
    sign_oci_request,
    validate_oci_environment,
)
from litellm.types.llms.oci import (
    CohereChatRequest,
    OCIChatRequestPayload,
    OCICompletionPayload,
    OCIServingMode,
    OCIVendors,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    LlmProviders,
    ModelResponse,
    ModelResponseStream,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# Streaming timeout — generous because OCI models may need to warm up on first request
STREAMING_TIMEOUT = 60 * 5


def get_vendor_from_model(model: str) -> OCIVendors:
    """Return the OCI vendor enum for a model name.

    OCI GenAI uses two ``apiFormat`` values:

    - ``"COHERE"`` for Cohere models (``cohere.*``)
    - ``"GENERIC"`` for all others (Meta Llama, xAI Grok, Google Gemini, …)
    """
    vendor = model.split(".")[0].lower()
    if vendor == "cohere":
        return OCIVendors.COHERE
    return OCIVendors.GENERIC


class OCIChatConfig(BaseConfig):
    """LiteLLM BaseConfig implementation for OCI Generative AI chat."""

    def __init__(self) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
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

        # Cohere param map differs from GENERIC in three ways:
        # - tool_choice is unsupported
        # - stop sequences key is "stopSequences" not "stop"
        # - n (numGenerations) is GENERIC-only
        self.openai_to_oci_cohere_param_map = {
            k: ("stopSequences" if k == "stop" else v)
            for k, v in self.openai_to_oci_generic_param_map.items()
            if k not in ("tool_choice", "max_retries", "n")
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        param_map = (
            self.openai_to_oci_cohere_param_map
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
        param_map = (
            self.openai_to_oci_cohere_param_map
            if vendor == OCIVendors.COHERE
            else self.openai_to_oci_generic_param_map
        )

        for key, value in {**non_default_params, **optional_params}.items():
            alias = param_map.get(key)
            if alias is False:
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
    ) -> Tuple[dict, bytes]:
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
        if optional_params.get("oci_signer") is None:
            creds = resolve_oci_credentials(optional_params)
            missing = [
                k
                for k in (
                    "oci_user",
                    "oci_fingerprint",
                    "oci_tenancy",
                    "oci_compartment_id",
                )
                if not creds.get(k)
            ]
            if missing or not (creds.get("oci_key") or creds.get("oci_key_file")):
                raise OCIError(
                    status_code=401,
                    message=(
                        "Missing required parameters: oci_user, oci_fingerprint, oci_tenancy, "
                        "oci_compartment_id and at least one of oci_key or oci_key_file. "
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
        base = get_oci_base_url(optional_params, api_base)
        return f"{base}/{OCI_API_VERSION}/actions/chat"

    def _get_optional_params(self, vendor: OCIVendors, optional_params: dict) -> Dict:
        param_map = (
            self.openai_to_oci_cohere_param_map
            if vendor == OCIVendors.COHERE
            else self.openai_to_oci_generic_param_map
        )
        selected_params: Dict = {}

        for openai_key, oci_key in param_map.items():
            if oci_key and openai_key in optional_params:
                selected_params[oci_key] = optional_params[openai_key]  # type: ignore[index]

        for oci_value in param_map.values():
            if (
                oci_value
                and oci_value in optional_params
                and oci_value not in selected_params
            ):
                selected_params[oci_value] = optional_params[oci_value]  # type: ignore[index]

        if "tools" in selected_params:
            if vendor == OCIVendors.COHERE:
                selected_params["tools"] = adapt_tool_definitions_to_cohere_standard(  # type: ignore[assignment]
                    selected_params["tools"]  # type: ignore[arg-type]
                )
            else:
                selected_params["tools"] = adapt_tool_definition_to_oci_standard(  # type: ignore[assignment]
                    selected_params["tools"], vendor  # type: ignore[arg-type]
                )

        # Convert toolChoice from OpenAI string ("auto", "none", "required") to the
        # OCI dict form ({"type": "AUTO"} etc.) — the API rejects plain strings.
        if "toolChoice" in selected_params:
            tc = selected_params["toolChoice"]
            if isinstance(tc, str):
                tc_map = {
                    "auto": {"type": "AUTO"},
                    "none": {"type": "NONE"},
                    "required": {"type": "REQUIRED"},
                    "any": {"type": "REQUIRED"},
                }
                selected_params["toolChoice"] = tc_map.get(
                    tc.lower(), {"type": "FUNCTION", "name": tc}
                )

        if "responseFormat" in selected_params:
            rf = selected_params["responseFormat"]
            if isinstance(rf, dict) and "type" in rf:
                rf_payload = dict(rf)
                selected_params["responseFormat"] = rf_payload
                response_type = rf_payload["type"]
                schema_payload: Optional[Any] = None
                if "json_schema" in rf_payload:
                    raw_schema = rf_payload.pop("json_schema")
                    schema_payload = (
                        dict(raw_schema) if isinstance(raw_schema, dict) else raw_schema
                    )
                if schema_payload is not None:
                    rf_payload["jsonSchema"] = schema_payload
                if vendor == OCIVendors.COHERE:
                    rf_payload["type"] = response_type
                else:
                    fmt = response_type.upper()
                    rf_payload["type"] = "JSON_OBJECT" if fmt == "JSON" else fmt

        return selected_params

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
            serving_mode = OCIServingMode(
                servingType="DEDICATED",
                endpointId=optional_params.get("oci_endpoint_id", model),
            )
        else:
            serving_mode = OCIServingMode(servingType="ON_DEMAND", modelId=model)

        if vendor == OCIVendors.COHERE:
            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                raise OCIError(
                    status_code=400,
                    message="No user message found — Cohere models require at least one user message",
                )

            system_messages = [m for m in messages if m.get("role") == "system"]
            preamble_override = None
            if system_messages:
                preamble = "\n".join(
                    _extract_text_content(m["content"]) for m in system_messages
                )
                if preamble:
                    preamble_override = preamble

            chat_request = CohereChatRequest(
                apiFormat="COHERE",
                message=_extract_text_content(user_messages[-1]["content"]),
                chatHistory=adapt_messages_to_cohere_standard(messages),
                preambleOverride=preamble_override,
                **self._get_optional_params(OCIVendors.COHERE, optional_params),
            )
            data = OCICompletionPayload(
                compartmentId=oci_compartment_id,
                servingMode=serving_mode,
                chatRequest=chat_request,
            )
        else:
            data = OCICompletionPayload(
                compartmentId=oci_compartment_id,
                servingMode=serving_mode,
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
        response_json = raw_response.json()

        if response_json.get("error") is not None:
            raise OCIError(
                message=str(response_json["error"]),
                status_code=raw_response.status_code,
            )
        if not isinstance(response_json, dict):
            raise OCIError(
                message="Invalid response format from OCI",
                status_code=raw_response.status_code,
            )

        vendor = get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            model_response = handle_cohere_response(
                response_json, model, model_response
            )
        else:
            model_response = handle_generic_response(
                response_json, model, model_response, raw_response
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
        signed_json_body: bytes = b"",
    ) -> "OCIStreamWrapper":
        if "stream" in data:
            del data["stream"]
        if client is None or isinstance(client, AsyncHTTPHandler):
            client = _get_httpx_client(params={})

        try:
            response = client.post(
                api_base,
                headers=headers,
                data=signed_json_body or json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        def split_chunks(stream: Iterator[str]) -> Iterator[str]:
            for item in stream:
                for chunk in item.split("\n\n"):
                    stripped = chunk.strip()
                    if stripped:
                        yield stripped

        return OCIStreamWrapper(
            completion_stream=split_chunks(response.iter_text()),
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

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
        signed_json_body: bytes = b"",
    ) -> "OCIStreamWrapper":
        if "stream" in data:
            del data["stream"]
        if client is None or isinstance(client, HTTPHandler):
            client = get_async_httpx_client(llm_provider=LlmProviders.OCI, params={})

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=signed_json_body or json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        completion_stream = response.aiter_text()

        async def split_chunks(stream: AsyncIterator[str]) -> AsyncIterator[str]:
            async for item in stream:
                for chunk in item.split("\n\n"):
                    stripped = chunk.strip()
                    if stripped:
                        yield stripped

        return OCIStreamWrapper(
            completion_stream=split_chunks(completion_stream),
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OCIError(status_code=status_code, message=error_message)


class OCIStreamWrapper(CustomStreamWrapper):
    """Custom stream wrapper that dispatches OCI SSE chunks to the correct handler."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def chunk_creator(self, chunk: Any) -> ModelResponseStream:
        if not isinstance(chunk, str):
            raise ValueError(f"Chunk is not a string: {chunk}")
        if not chunk.startswith("data:"):
            raise ValueError(f"Chunk does not start with 'data:': {chunk}")
        dict_chunk = json.loads(chunk[5:])

        if dict_chunk.get("apiFormat") == "COHERE":
            return handle_cohere_stream_chunk(dict_chunk)
        return handle_generic_stream_chunk(dict_chunk)


__all__ = [
    "OCIChatConfig",
    "OCIStreamWrapper",
    "OCIRequestWrapper",
    "OCI_API_VERSION",
    "STREAMING_TIMEOUT",
    "get_vendor_from_model",
    "version",
]
