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
from litellm.constants import DEFAULT_OCI_CHAT_MAX_TOKENS
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
from litellm.utils import supports_reasoning
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# Streaming timeout — generous because OCI models may need to warm up on first request
STREAMING_TIMEOUT = 60 * 5


def _model_uses_max_completion_tokens(model: str) -> bool:
    """Return True for OCI-hosted models that require ``maxCompletionTokens``.

    OpenAI commercial models proxied through OCI (``openai.*``) reject
    ``maxTokens`` with HTTP 400 on the reasoning families (gpt-5.x, o-series)
    and accept ``maxCompletionTokens`` everywhere, so route the whole vendor
    prefix to it rather than chasing each new release in
    ``model_prices_and_context_window.json``. The ``openai.gpt-oss-*`` open
    weights are served by OCI's own stack and keep ``maxTokens``. Any other
    vendor falls back to the catalog's ``supports_reasoning`` flag.
    """
    if not model:
        return False
    name = model[4:] if model.lower().startswith("oci/") else model
    lowered = name.lower()
    if lowered.startswith("openai."):
        return not lowered.startswith("openai.gpt-oss")
    return supports_reasoning(model=name, custom_llm_provider="oci")


def _iter_sse_events(stream: Iterator[str]) -> Iterator[str]:
    """Yield one ``data:`` SSE line at a time from a sync text stream.

    The OCI streaming endpoint does not align SSE event boundaries with HTTP
    read boundaries. A single read may carry multiple events, a single event
    may straddle two reads, and some events arrive separated by only ``\\n``
    instead of ``\\n\\n``. This helper buffers across reads and yields each
    complete ``data:`` line so JSON parsing downstream never sees a partial
    payload.
    """
    buffer = ""
    for item in stream:
        buffer += item
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            stripped = line.strip()
            if stripped.startswith("data:"):
                yield stripped
    stripped = buffer.strip()
    if stripped.startswith("data:"):
        yield stripped


async def _aiter_sse_events(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Async twin of :func:`_iter_sse_events`."""
    buffer = ""
    async for item in stream:
        buffer += item
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            stripped = line.strip()
            if stripped.startswith("data:"):
                yield stripped
    stripped = buffer.strip()
    if stripped.startswith("data:"):
        yield stripped


def _normalize_tool_choice(selected_params: Dict) -> None:
    tc = selected_params.get("toolChoice")
    if tc is None:
        return
    if isinstance(tc, str):
        tc_map = {
            "auto": {"type": "AUTO"},
            "none": {"type": "NONE"},
            "required": {"type": "REQUIRED"},
            "any": {"type": "REQUIRED"},
        }
        selected_params["toolChoice"] = tc_map.get(tc.lower(), {"type": "FUNCTION", "name": tc})
        return
    if isinstance(tc, dict):
        raw_type = tc.get("type")
        if not isinstance(raw_type, str):
            raise OCIError(
                status_code=400,
                message=f"Invalid tool_choice for OCI: missing or non-string 'type' in {tc!r}",
            )
        upper = raw_type.upper()
        if upper == "FUNCTION":
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
            if not (isinstance(name, str) and name):
                raise OCIError(
                    status_code=400,
                    message="Invalid tool_choice for OCI: 'FUNCTION' type requires a non-empty function name",
                )
            selected_params["toolChoice"] = {"type": "FUNCTION", "name": name}
        elif upper in {"AUTO", "NONE", "REQUIRED"}:
            selected_params["toolChoice"] = {"type": upper}
        else:
            raise OCIError(
                status_code=400,
                message=(
                    f"Invalid tool_choice for OCI: unsupported type {raw_type!r}; "
                    "expected one of 'FUNCTION', 'AUTO', 'NONE', 'REQUIRED'"
                ),
            )
        return
    raise OCIError(
        status_code=400,
        message=(f"Invalid tool_choice for OCI: expected str or dict, got {type(tc).__name__}"),
    )


def _normalize_response_format(selected_params: Dict, vendor: OCIVendors) -> None:
    rf = selected_params.get("responseFormat")
    if not isinstance(rf, dict) or "type" not in rf:
        return

    rf_type = str(rf["type"]).lower()
    raw_schema = rf.get("json_schema")
    json_schema = raw_schema if isinstance(raw_schema, dict) else None

    if rf_type == "text":
        selected_params["responseFormat"] = {"type": "TEXT"}
        return

    if vendor == OCIVendors.COHERE:
        # OCI Cohere has no JSON_SCHEMA type; a schema rides on JSON_OBJECT.
        payload: Dict[str, Any] = {"type": "JSON_OBJECT"}
        if json_schema is not None and json_schema.get("schema") is not None:
            payload["schema"] = json_schema["schema"]
        selected_params["responseFormat"] = payload
        return

    if rf_type == "json_schema":
        if json_schema is None:
            raise OCIError(
                status_code=400,
                message="response_format type 'json_schema' requires a 'json_schema' object",
            )
        # OCI's ResponseJsonSchema accepts only name/description/schema/isStrict.
        # OpenAI sends `strict` instead of `isStrict`; forwarding it (or any
        # other extra key) makes OCI reject the whole request with HTTP 400.
        oci_schema: Dict[str, Any] = {"name": json_schema.get("name") or "response"}
        if json_schema.get("description") is not None:
            oci_schema["description"] = json_schema["description"]
        if json_schema.get("schema") is not None:
            oci_schema["schema"] = json_schema["schema"]
        if json_schema.get("strict") is not None:
            oci_schema["isStrict"] = json_schema["strict"]
        selected_params["responseFormat"] = {
            "type": "JSON_SCHEMA",
            "jsonSchema": oci_schema,
        }
        return

    fmt = rf_type.upper()
    selected_params["responseFormat"] = {"type": "JSON_OBJECT" if fmt == "JSON" else fmt}


def get_vendor_from_model(model: str) -> OCIVendors:
    """Return the OCI vendor enum for a model name.

    OCI GenAI uses two ``apiFormat`` values:

    - ``"COHERE"`` for Cohere models (``cohere.*``)
    - ``"GENERIC"`` for all others (Meta Llama, xAI Grok, Google Gemini, …)
    """
    name = model[4:] if model.lower().startswith("oci/") else model
    vendor = name.split(".")[0].lower()
    if vendor == "cohere":
        return OCIVendors.COHERE
    return OCIVendors.GENERIC


class OCIChatConfig(BaseConfig):
    """LiteLLM BaseConfig implementation for OCI Generative AI chat."""

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    def __init__(self) -> None:
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
            "reasoning_effort": "reasoningEffort",
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
        # The unsupported keys are kept in the map with value ``False`` so
        # ``map_openai_params`` either drops them (under drop_params) or raises
        # a clear error, rather than silently passing them through.
        self.openai_to_oci_cohere_param_map = {
            k: ("stopSequences" if k == "stop" else v) for k, v in self.openai_to_oci_generic_param_map.items()
        }
        self.openai_to_oci_cohere_param_map["tool_choice"] = False
        self.openai_to_oci_cohere_param_map["n"] = False
        # ``top_k`` is not a standard OpenAI param, but Cohere's chat request
        # accepts ``topK`` and LiteLLM commonly forwards ``top_k`` as a
        # passthrough param. Cohere-only — ``OCIChatRequestPayload`` (GENERIC)
        # has no ``topK`` field.
        self.openai_to_oci_cohere_param_map["top_k"] = "topK"
        # OCI Cohere models are not reasoning models; mark reasoning_effort
        # explicitly unsupported so callers either get a clear error or have
        # the param dropped under drop_params, rather than silently passing
        # through and tripping Pydantic validation on CohereChatRequest.
        self.openai_to_oci_cohere_param_map["reasoning_effort"] = False
        # CohereChatRequest has no logProbs/logitBias fields, so passing these
        # through would be silently dropped by Pydantic. Mark them unsupported
        # so get_supported_openai_params doesn't advertise them and callers
        # get a clear error (or drop_params behaviour) instead.
        self.openai_to_oci_cohere_param_map["logprobs"] = False
        self.openai_to_oci_cohere_param_map["logit_bias"] = False

    def get_supported_openai_params(self, model: str) -> List[str]:
        param_map = (
            self.openai_to_oci_cohere_param_map
            if get_vendor_from_model(model) == OCIVendors.COHERE
            else self.openai_to_oci_generic_param_map
        )
        # `n` is intentionally not advertised for Cohere even though n=1 is
        # tolerated: Cohere has no numGenerations field, so n>1 cannot be
        # honoured and advertising it would be misleading. Callers that gate on
        # this list strip n=1 (a no-op, matching what map_openai_params does);
        # callers that bypass it have n=1 dropped there. Both paths converge.
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
            self.openai_to_oci_cohere_param_map if vendor == OCIVendors.COHERE else self.openai_to_oci_generic_param_map
        )

        for key, value in {**non_default_params, **optional_params}.items():
            alias = param_map.get(key)
            if alias is False:
                # max_retries is a litellm-level control param (litellm applies
                # retries itself); it is never a generation param OCI accepts, so
                # drop it silently. The litellm proxy injects it on every request,
                # which otherwise 500s OCI calls unless drop_params is set.
                if key == "max_retries":
                    continue
                # n=1 (or None) is the OpenAI default: a single generation, which
                # every OCI model produces anyway. Drop it silently so standard
                # clients that always send n=1 (e.g. the MLflow gateway) are not
                # rejected; only n>1 is genuinely unsupported on Cohere, which
                # has no numGenerations field.
                if key == "n" and (value is None or value == 1):
                    continue
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
            # Preserve the original OpenAI ``response_format`` key alongside the
            # OCI-mapped ``responseFormat`` so downstream litellm framework code
            # (e.g. ``json_mode`` detection, logging) that inspects
            # ``optional_params["response_format"]`` continues to work.
            if alias == "responseFormat":
                adapted_params["response_format"] = value

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
        base = get_oci_base_url(optional_params, api_base or litellm.api_base)
        return f"{base}/{OCI_API_VERSION}/actions/chat"

    def _get_optional_params(self, vendor: OCIVendors, optional_params: dict, model: str = "") -> Dict:
        param_map = (
            self.openai_to_oci_cohere_param_map if vendor == OCIVendors.COHERE else self.openai_to_oci_generic_param_map
        )
        selected_params: Dict = {}

        # OpenAI reasoning models on OCI (e.g. GPT-5 family) reject "maxTokens"
        # and require "maxCompletionTokens" per OCI's /20231130/Chat schema.
        # Driven by the supports_reasoning flag in the model catalog. Cohere's
        # endpoint uses "maxTokens" regardless, so the override is GENERIC-only.
        max_tokens_key = (
            "maxCompletionTokens"
            if vendor != OCIVendors.COHERE and model and _model_uses_max_completion_tokens(model)
            else "maxTokens"
        )

        # ``map_openai_params`` runs before ``transform_request`` (and thus
        # before this helper), so by the time we see ``optional_params`` the
        # OpenAI keys have already been translated to their OCI aliases.
        # We still accept the original OpenAI key as a fallback for callers
        # that build ``optional_params`` directly, with OpenAI keys winning
        # over OCI aliases when both happen to be present. The first OpenAI
        # key reaching a given OCI target wins, so ``max_tokens`` /
        # ``max_completion_tokens`` (both → ``maxTokens``) don't double-write.
        for openai_key, oci_alias in param_map.items():
            if not oci_alias:
                continue
            target = max_tokens_key if oci_alias == "maxTokens" else oci_alias
            if target in selected_params:
                continue
            if openai_key in optional_params:
                selected_params[target] = optional_params[openai_key]  # type: ignore[index]
            elif oci_alias in optional_params:
                selected_params[target] = optional_params[oci_alias]  # type: ignore[index]

        # OCI's server-side default token cap is tiny (~20 tokens), so an
        # omitted max_tokens silently truncates the response mid-string. Most
        # callers never send a limit (MLflow judges among them), so inject a
        # sane default when one is absent, mirroring litellm's Anthropic config.
        if max_tokens_key not in selected_params:
            selected_params[max_tokens_key] = DEFAULT_OCI_CHAT_MAX_TOKENS

        # OCI expects uppercase reasoning levels (LOW/MEDIUM/HIGH/NONE); OpenAI
        # clients send lowercase. OpenAI's "disable" maps to OCI's "NONE".
        if "reasoningEffort" in selected_params:
            effort = selected_params["reasoningEffort"]
            if isinstance(effort, str):
                normalized = effort.upper()
                if normalized == "DISABLE":
                    normalized = "NONE"
                selected_params["reasoningEffort"] = normalized

        if "tools" in selected_params:
            if vendor == OCIVendors.COHERE:
                selected_params["tools"] = adapt_tool_definitions_to_cohere_standard(  # type: ignore[assignment]
                    selected_params["tools"]  # type: ignore[arg-type]
                )
            else:
                selected_params["tools"] = adapt_tool_definition_to_oci_standard(  # type: ignore[assignment]
                    selected_params["tools"],
                    vendor,  # type: ignore[arg-type]
                )

        # Normalise tool_choice to OCI's flat uppercase dict form
        # ({"type": "AUTO"|"NONE"|"REQUIRED"} or {"type": "FUNCTION", "name": "<fn>"}).
        # OCI rejects both the OpenAI string and the nested OpenAI dict shape.
        _normalize_tool_choice(selected_params)

        _normalize_response_format(selected_params, vendor)

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
                preamble = "\n".join(_extract_text_content(m["content"]) for m in system_messages)
                if preamble:
                    preamble_override = preamble

            chat_request = CohereChatRequest(
                apiFormat="COHERE",
                message=_extract_text_content(user_messages[-1]["content"]),
                chatHistory=adapt_messages_to_cohere_standard([m for m in messages if m.get("role") != "system"]),
                preambleOverride=preamble_override,
                **self._get_optional_params(OCIVendors.COHERE, optional_params, model),
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
                    **self._get_optional_params(vendor, optional_params, model),
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

        if not isinstance(response_json, dict):
            raise OCIError(
                message="Invalid response format from OCI",
                status_code=raw_response.status_code,
            )

        if response_json.get("error") is not None:
            raise OCIError(
                message=str(response_json["error"]),
                status_code=raw_response.status_code,
            )

        vendor = get_vendor_from_model(model)
        if vendor == OCIVendors.COHERE:
            model_response = handle_cohere_response(response_json, model, model_response, raw_response)
        else:
            model_response = handle_generic_response(response_json, model, model_response, raw_response)

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
        if client is None or isinstance(client, AsyncHTTPHandler):
            client = _get_httpx_client(params={})

        try:
            response = client.post(
                api_base,
                headers=headers,
                data=(signed_json_body if signed_json_body is not None else json.dumps(data)),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        return OCIStreamWrapper(
            completion_stream=_iter_sse_events(response.iter_text()),
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
        signed_json_body: Optional[bytes] = None,
    ) -> "OCIStreamWrapper":
        if client is None or isinstance(client, HTTPHandler):
            client = get_async_httpx_client(llm_provider=LlmProviders.OCI, params={})

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=(signed_json_body if signed_json_body is not None else json.dumps(data)),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise OCIError(status_code=e.response.status_code, message=e.response.text)

        if response.status_code != 200:
            raise OCIError(status_code=response.status_code, message=response.text)

        return OCIStreamWrapper(
            completion_stream=_aiter_sse_events(response.aiter_text()),
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
        # Tracks whether any prior Cohere chunk in this stream has emitted
        # tool calls. The Cohere handler uses this to decide whether the
        # terminal consolidation chunk's tool calls are duplicates (suppress)
        # or the only copy of the tool calls (pass through).
        self._cohere_tool_calls_emitted = False
        # Analogous flag for text content. Lets the Cohere handler distinguish
        # the common case (prior deltas already streamed the text, so the
        # terminal chunk's text is a duplicate to suppress) from the degenerate
        # single-event case (terminal chunk carries the only copy of the text).
        self._cohere_text_emitted = False

    def chunk_creator(self, chunk: Any) -> ModelResponseStream:
        if not isinstance(chunk, str):
            raise ValueError(f"Chunk is not a string: {chunk}")
        if not chunk.startswith("data:"):
            raise ValueError(f"Chunk does not start with 'data:': {chunk}")
        try:
            dict_chunk = json.loads(chunk[5:])
        except json.JSONDecodeError as e:
            raise OCIError(
                status_code=500,
                message=f"Chunk cannot be parsed as JSON: {str(e)}",
            )

        if dict_chunk.get("apiFormat") == "COHERE":
            result = handle_cohere_stream_chunk(
                dict_chunk,
                prior_tool_calls_emitted=self._cohere_tool_calls_emitted,
                prior_text_emitted=self._cohere_text_emitted,
            )
            if not self._cohere_tool_calls_emitted:
                for choice in result.choices:
                    if getattr(choice.delta, "tool_calls", None) is not None:
                        self._cohere_tool_calls_emitted = True
                        break
            if not self._cohere_text_emitted:
                for choice in result.choices:
                    if getattr(choice.delta, "content", None):
                        self._cohere_text_emitted = True
                        break
            return result
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
