"""The v1-side adapter for translation v2 (lives OUTSIDE litellm/translation).

This module is the one place that speaks both languages: it reads litellm
ambient state (model map helpers, drop_params/modify_params, the per-provider
allowlist flag) into the package's injected ``TranslationDeps`` value, adapts
v2's plain response bodies onto ``ModelResponse``, and owns the
``completion()`` fork. The translation package never imports the v1 stack;
this seam imports both, in the allowed direction.

Flag precedent (dossier section 7): ``litellm.translation_v2_providers`` is a
module global seeded from the ``LITELLM_TRANSLATION_V2_PROVIDERS``
comma-separated env var, and the proxy's ``litellm_settings`` generic setattr
fallback makes it yaml-configurable with zero extra plumbing. Off by default.
"""

from __future__ import annotations

import json
from typing import Any, Dict, FrozenSet, Optional, cast

import litellm
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CompletionTokensDetailsWrapper,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

from litellm.translation import TranslationDeps
from litellm.translation.ir import Body


def enabled_providers() -> FrozenSet[str]:
    """The per-provider opt-in allowlist, read at call time so proxy yaml and
    runtime changes apply without restart."""
    configured = getattr(litellm, "translation_v2_providers", None) or []
    return frozenset(str(entry).strip() for entry in configured if str(entry).strip())


def _max_tokens_for_model(model: str) -> Optional[int]:
    try:
        return litellm.utils.get_max_tokens(model)
    except Exception:
        return None


def _count_response_tokens(text: str) -> int:
    from litellm.utils import token_counter

    return token_counter(text=text, count_response_tokens=True)


def build_translation_deps(
    request_drop_params: Optional[bool] = None,
) -> TranslationDeps:
    drop_params_global = litellm.drop_params is True
    return TranslationDeps(
        max_tokens_for_model=_max_tokens_for_model,
        supports_capability=AnthropicModelInfo._supports_model_capability,
        capability_flag=AnthropicModelInfo._get_model_capability,
        count_response_tokens=_count_response_tokens,
        drop_params=drop_params_global or request_drop_params is True,
        drop_params_global=drop_params_global,
        modify_params=litellm.modify_params is True,
    )


UsageStyle = str  # "anthropic" | "bedrock_converse" | "openai" (v1 transform)


def to_model_response(
    body: Body,
    model_response: Optional[ModelResponse] = None,
    usage_style: UsageStyle = "anthropic",
) -> ModelResponse:
    """Adapt a v2 response body onto litellm's ModelResponse envelope.

    Mirrors v1's per-provider response assembly exactly (assign into the
    pre-allocated response's first choice, stamp created/model, setattr a
    real ``Usage`` built with that transform's exact kwarg set — extra-field
    serialization only dumps explicitly set fields). The envelope (chatcmpl
    id, created timestamp) stays litellm-ambient, except on the ``openai``
    style where v1's ``convert_to_model_response_object`` keeps the wire
    id/created/system_fingerprint and setattrs unknown top-level keys.
    """
    import time

    from litellm.types.utils import Message

    if usage_style == "openai":
        return _to_model_response_openai(body, model_response)

    response = model_response if model_response is not None else ModelResponse()
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    message_payload = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message_payload, dict):
        response.choices[0].message = Message(**cast(Dict[str, Any], message_payload))
    finish = first.get("finish_reason") if isinstance(first, dict) else None
    response.choices[0].finish_reason = cast(
        Any, finish if isinstance(finish, str) else "stop"
    )
    usage_payload = body.get("usage")
    if not isinstance(usage_payload, dict):
        usage = Usage()
    elif usage_style == "bedrock_converse":
        usage = _build_usage_converse(usage_payload)
    else:
        usage = _build_usage(usage_payload)
    setattr(response, "usage", usage)
    response.created = int(time.time())
    model = body.get("model")
    if isinstance(model, str):
        response.model = model
    return response


_OPENAI_BODY_ENVELOPE_FIELDS = (
    "object",
    "choices",
    "usage",
    "id",
    "created",
    "model",
    "system_fingerprint",
)


def _to_model_response_openai(
    body: Body, model_response: Optional[ModelResponse]
) -> ModelResponse:
    """Mirror ``convert_to_model_response_object``'s completion-branch
    assembly over the normalized body the openai_compat parser built: choices
    are REBUILT (v1 replaces the list with ``Choices(...)`` carrying the
    exact kwarg set), usage is the verbatim ``Usage(**raw_usage)``
    passthrough, the wire id/created/system_fingerprint survive, and unknown
    top-level keys are setattr'd."""
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        _safe_convert_created_field,
    )
    from litellm.types.utils import Choices, Message

    response = model_response if model_response is not None else ModelResponse()
    raw_choices = body.get("choices")
    rebuilt = []
    if isinstance(raw_choices, list):
        for idx, choice in enumerate(raw_choices):
            if not isinstance(choice, dict):
                continue
            message_payload = choice.get("message")
            message = (
                Message(**cast(Dict[str, Any], message_payload))
                if isinstance(message_payload, dict)
                else Message()
            )
            rebuilt.append(
                Choices(
                    finish_reason=choice.get("finish_reason"),
                    index=idx,
                    message=message,
                    logprobs=choice.get("logprobs"),
                    enhancements=choice.get("enhancements"),
                    provider_specific_fields=choice.get("provider_specific_fields"),
                )
            )
    response.choices = cast(Any, rebuilt)
    usage_payload = body.get("usage")
    if isinstance(usage_payload, dict):
        setattr(response, "usage", Usage(**cast(Dict[str, Any], usage_payload)))
    if "created" in body:
        response.created = _safe_convert_created_field(cast(Any, body["created"]))
    body_id = body.get("id")
    if isinstance(body_id, str) and body_id:
        response.id = body_id
    if "system_fingerprint" in body:
        response.system_fingerprint = cast(Any, body["system_fingerprint"])
    if "model" in body:
        wire_model = body["model"]
        if response.model is None:
            response.model = cast(Any, wire_model)
        elif "/" in response.model and wire_model is not None:
            # v1's openai handler pre-sets model_response.model to
            # "{custom_llm_provider}/{model}" for every non-"openai" compat
            # consumer (llms/openai/openai.py), and the completion branch then
            # rewrites it to "{provider}/{wire model}"
            # (convert_dict_to_response.py:699-711). Mirror it verbatim.
            compat_provider = response.model.split("/")[0]
            response.model = f"{compat_provider}/{wire_model}"
    for key, value in body.items():
        if key not in _OPENAI_BODY_ENVELOPE_FIELDS:
            setattr(response, key, value)
    return response


def _build_usage_converse(payload: dict) -> Usage:
    """Construct Usage with the exact kwarg set v1's converse
    ``_transform_usage`` passes: no ephemeral cache-creation detail and no
    server_tool_use/inference_geo/speed extras."""
    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=prompt_details.get("cached_tokens"),
            cache_creation_tokens=prompt_details.get("cache_creation_tokens"),
            text_tokens=prompt_details.get("text_tokens"),
        ),
        cache_creation_input_tokens=payload.get("cache_creation_input_tokens"),
        cache_read_input_tokens=payload.get("cache_read_input_tokens"),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            text_tokens=completion_details.get("text_tokens"),
        ),
    )


def _build_usage(payload: dict) -> Usage:
    """Construct Usage with the exact kwarg set v1's calculate_usage passes,
    because nested (extra-attribute) serialization only includes explicitly
    set fields; a kwargs-splat would drop the always-set None fields v1 emits
    (server_tool_use, inference_geo, speed, wrapper audio/image/video)."""
    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    creation_details = prompt_details.get("cache_creation_token_details")
    wrapper_details = (
        CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=creation_details.get("ephemeral_5m_input_tokens"),
            ephemeral_1h_input_tokens=creation_details.get("ephemeral_1h_input_tokens"),
        )
        if isinstance(creation_details, dict)
        else None
    )
    return Usage(
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=prompt_details.get("cached_tokens"),
            cache_creation_tokens=prompt_details.get("cache_creation_tokens"),
            cache_creation_token_details=wrapper_details,
            text_tokens=prompt_details.get("text_tokens"),
        ),
        cache_creation_input_tokens=payload.get("cache_creation_input_tokens"),
        cache_read_input_tokens=payload.get("cache_read_input_tokens"),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            text_tokens=completion_details.get("text_tokens"),
        ),
        server_tool_use=None,
        inference_geo=None,
        speed=None,
    )


def to_model_response_stream(body: Body, stream_id: str):
    """Adapt one v2 chunk body onto ModelResponseStream; the stream id is
    minted once per stream by the caller (v1 reuses one id for every chunk)
    unless the body pinned the wire id (openai dialect, mirroring v1's
    ``set_model_id``). system_fingerprint/citations are set explicitly
    because v1's wrapper always sets them and extra-field serialization only
    dumps set fields; a body-carried ``usage`` becomes the verbatim
    ``Usage(**dict)`` passthrough (v1 streaming_handler.py:1511-1516)."""
    from litellm.types.utils import ModelResponseStream

    payload = dict(body)
    body_id = payload.pop("id", None)
    usage_payload = payload.pop("usage", None)
    chunk = ModelResponseStream(
        id=body_id if isinstance(body_id, str) and body_id else stream_id,
        **cast(Dict[str, Any], {"system_fingerprint": None, **payload}),
    )
    if isinstance(usage_payload, dict):
        setattr(chunk, "usage", Usage(**cast(Dict[str, Any], usage_payload)))
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    if isinstance(first, dict) and first.get("finish_reason") is None:
        # v1 sets citations only on content chunks, never the finish chunk
        # nor the choices=[] usage chunk.
        setattr(chunk, "citations", None)
    return chunk


class HttpxJsonPort:
    """HttpPort over a fresh httpx.AsyncClient per request.

    A fresh client avoids binding litellm's cached async clients to the
    short-lived event loop the sync wrapper creates. Pooled-client reuse on
    the async path is a follow-up optimization; the flag-gated M3 traffic
    does not need it.
    """

    async def post_json(self, endpoint, body):  # noqa: ANN001, ANN201
        import httpx

        from litellm.translation.engine.http import HttpResponse

        async with httpx.AsyncClient(timeout=endpoint.timeout_seconds) as client:
            raw = await client.post(
                endpoint.url, headers=dict(endpoint.headers), json=dict(body)
            )
        try:
            payload = raw.json()
        except ValueError:
            payload = None
        return HttpResponse(
            status_code=raw.status_code,
            body=payload,
            text=raw.text,
            headers=dict(raw.headers),
        )


def try_completion_v2(
    *,
    model: str,
    messages: list,
    optional_param_args: dict,
    non_default_params: dict,
    api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[float],
    stream: Optional[bool],
    acompletion: Optional[bool],
    logging_obj,
    model_response: ModelResponse,
    request_drop_params: Optional[bool],
):
    """The completion() fork for translation v2 (anthropic, non-streaming).

    Returns None to stay on v1: flag off, streaming (the v2 stream seam is a
    follow-up), modify_params behaviors, or any request shape outside v2's
    proven surface (the fail-closed translate). Once the request is sent,
    failures raise the provider error contract exactly like v1; there is no
    silent re-send.
    """
    from litellm.translation import route
    from litellm.translation.engine.pipeline import prepare_chat_request

    if "anthropic" not in enabled_providers():
        return None
    if stream is True or litellm.modify_params is True:
        return None
    decision = route(
        schema="openai_chat",
        provider="anthropic",
        enabled_providers=frozenset({"anthropic"}),  # checked above
        body_touching=False,
    )
    if decision.tag != "v2":
        return None
    raw_body = _raw_openai_body(
        model, messages, optional_param_args, non_default_params
    )
    deps = build_translation_deps(request_drop_params=request_drop_params)
    prepared_result = prepare_chat_request(raw_body, "anthropic", deps)
    if prepared_result.is_error():
        litellm.verbose_logger.debug(
            "translation v2 fallback to v1: %s", prepared_result.error.summary
        )
        return None
    prepared = prepared_result.ok
    coroutine = _send_v2(
        prepared=prepared,
        deps=deps,
        model=model,
        messages=messages,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        logging_obj=logging_obj,
        model_response=model_response,
    )
    if acompletion is True:
        return coroutine
    import asyncio

    return asyncio.run(coroutine)


_BEDROCK_ROUTE_TO_PROVIDER = {
    "converse": "bedrock_converse",
    "invoke": "bedrock_invoke",
}

_AWS_PARAM_KEYS_PASSED_THROUGH = ("model_id", "guardrailConfig")


def try_completion_v2_bedrock(
    *,
    bedrock_route: str,
    model: str,
    messages: list,
    optional_param_args: dict,
    non_default_params: dict,
    api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[float],
    stream: Optional[bool],
    acompletion: Optional[bool],
    logging_obj,
    model_response: ModelResponse,
    request_drop_params: Optional[bool],
):
    """The completion() fork for translation v2 on the bedrock branch.

    Returns None to stay on v1: flag off, streaming, modify_params,
    provisioned-throughput model_id, non-Claude models, or any request shape
    outside v2's proven surface (the fail-closed translate). AWS auth params
    are split out of the body BEFORE parsing (they are envelope, not payload)
    and feed credential resolution; SigV4 signs AFTER the wire body is final
    (dossier section 8: sign-after-body-final, or bedrock 403s).
    """
    from litellm.translation.engine.pipeline import prepare_chat_request

    provider_key = _BEDROCK_ROUTE_TO_PROVIDER.get(bedrock_route)
    if provider_key is None or provider_key not in enabled_providers():
        return None
    if stream is True or litellm.modify_params is True:
        return None
    aws_params = {
        key: value
        for key, value in non_default_params.items()
        if key.startswith("aws_")
    }
    if any(key in non_default_params for key in _AWS_PARAM_KEYS_PASSED_THROUGH):
        return None  # provisioned throughput / guardrails stay on v1
    body_params = {
        key: value
        for key, value in non_default_params.items()
        if not key.startswith("aws_")
    }
    raw_body = _raw_openai_body(model, messages, optional_param_args, body_params)
    deps = build_translation_deps(request_drop_params=request_drop_params)
    prepared_result = prepare_chat_request(raw_body, provider_key, deps)  # type: ignore[arg-type]
    if prepared_result.is_error():
        litellm.verbose_logger.debug(
            "translation v2 fallback to v1: %s", prepared_result.error.summary
        )
        return None
    coroutine = _send_v2_bedrock(
        prepared=prepared_result.ok,
        provider_key=provider_key,
        deps=deps,
        model=model,
        messages=messages,
        aws_params=aws_params,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        logging_obj=logging_obj,
        model_response=model_response,
    )
    if acompletion is True:
        return coroutine
    import asyncio

    return asyncio.run(coroutine)


def _bedrock_endpoint_url(
    config,
    provider_key: str,
    model: str,
    aws_params: dict,
    aws_region_name: str,
    api_base: Optional[str],
) -> str:
    if provider_key == "bedrock_invoke":
        return config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params=dict(aws_params),
            litellm_params={},
            stream=False,
        )
    from litellm.llms.bedrock.common_utils import strip_bedrock_routing_prefix

    _, proxy_endpoint_url = config.get_runtime_endpoint(
        api_base=api_base,
        aws_bedrock_runtime_endpoint=aws_params.get("aws_bedrock_runtime_endpoint"),
        aws_region_name=aws_region_name,
    )
    model_id = config.encode_model_id(model_id=strip_bedrock_routing_prefix(model))
    return f"{proxy_endpoint_url}/model/{model_id}/converse"


async def _send_v2_bedrock(
    *,
    prepared,
    provider_key: str,
    deps: TranslationDeps,
    model: str,
    messages: list,
    aws_params: dict,
    api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[float],
    logging_obj,
    model_response: ModelResponse,
) -> ModelResponse:
    import httpx

    from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
        AmazonInvokeConfig,
    )
    from litellm.llms.bedrock.common_utils import BedrockError

    from litellm.translation.engine.pipeline import (
        response_dialect,
        translate_chat_response,
        wire_body,
    )

    config = AmazonInvokeConfig()
    region_params = dict(aws_params)
    aws_region_name = config._get_aws_region_name(
        optional_params=region_params, model=model
    )
    credentials = config.get_credentials(
        aws_access_key_id=aws_params.get("aws_access_key_id"),
        aws_secret_access_key=aws_params.get("aws_secret_access_key"),
        aws_session_token=aws_params.get("aws_session_token"),
        aws_region_name=aws_region_name,
        aws_session_name=aws_params.get("aws_session_name"),
        aws_profile_name=aws_params.get("aws_profile_name"),
        aws_role_name=aws_params.get("aws_role_name"),
        aws_web_identity_token=aws_params.get("aws_web_identity_token"),
        aws_sts_endpoint=aws_params.get("aws_sts_endpoint"),
        aws_external_id=aws_params.get("aws_external_id"),
    )
    url = _bedrock_endpoint_url(
        config, provider_key, model, aws_params, aws_region_name, api_base
    )
    # The wire body is FINAL here; SigV4 covers these exact bytes.
    data = json.dumps(wire_body(prepared, provider_key))  # type: ignore[arg-type]
    prepped = config.get_request_headers(
        credentials=credentials,
        aws_region_name=aws_region_name,
        extra_headers=None,
        endpoint_url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        api_key=api_key,
    )
    logging_obj.pre_call(
        input=messages,
        api_key="",
        additional_args={
            "complete_input_dict": data,
            "api_base": url,
            "headers": dict(prepped.headers),
        },
    )
    async with httpx.AsyncClient(
        timeout=float(timeout) if timeout else 600.0
    ) as client:
        raw = await client.post(url, headers=dict(prepped.headers), content=data)
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=raw.text,
        additional_args={"complete_input_dict": data},
    )
    if raw.status_code < 200 or raw.status_code >= 300:
        raise BedrockError(status_code=raw.status_code, message=raw.text)
    try:
        payload = raw.json()
    except ValueError as parse_error:
        raise BedrockError(
            status_code=422, message=f"non-JSON bedrock response: {raw.text[:200]}"
        ) from parse_error
    result = translate_chat_response(
        payload, prepared.request, provider_key, deps  # type: ignore[arg-type]
    )
    if result.is_error():
        raise BedrockError(status_code=500, message=result.error.summary)
    return to_model_response(
        result.ok, model_response, usage_style=response_dialect(provider_key)  # type: ignore[arg-type]
    )


_BODY_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "stream",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "user",
    "reasoning_effort",
    "thinking",
)


def _raw_openai_body(
    model: str, messages: list, optional_param_args: dict, non_default_params: dict
) -> dict:
    """Rebuild the caller's OpenAI-shape body from completion()'s pre-mapping
    locals. Provider-specific extras (non_default_params, e.g. top_k) ride
    along verbatim: anything v2 does not account for becomes a typed
    unsupported error and falls back to v1 -- the fail-closed allowlist."""
    named = {
        key: optional_param_args.get(key)
        for key in _BODY_FIELDS
        if optional_param_args.get(key) is not None
    }
    return {"model": model, "messages": messages, **named, **non_default_params}


async def _send_v2(
    *,
    prepared,
    deps: TranslationDeps,
    model: str,
    messages: list,
    api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[float],
    logging_obj,
    model_response: ModelResponse,
) -> ModelResponse:
    from litellm.llms.anthropic.common_utils import (
        AnthropicError,
        process_anthropic_headers,
    )
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    from litellm.translation.engine.http import Endpoint
    from litellm.translation.engine.pipeline import send_prepared, wire_body

    config = AnthropicConfig()
    headers = config.validate_environment(
        headers={},
        model=model,
        messages=messages,
        optional_params=dict(prepared.body),
        litellm_params={},
        api_key=api_key,
        api_base=api_base,
    )
    endpoint = Endpoint(
        url=api_base or "https://api.anthropic.com/v1/messages",
        headers=headers,
        timeout_seconds=float(timeout) if timeout else 600.0,
    )
    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={
            "complete_input_dict": wire_body(prepared),
            "api_base": endpoint.url,
            "headers": dict(headers),
        },
    )
    result = await send_prepared(prepared, "anthropic", deps, HttpxJsonPort(), endpoint)
    if result.is_error():
        error = result.error
        if error.tag == "provider_http":
            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=error.provider_http.text,
            )
            raise AnthropicError(
                status_code=error.provider_http.status_code,
                message=error.provider_http.text,
            )
        raise AnthropicError(status_code=500, message=error.summary)
    body = result.ok
    logging_obj.post_call(
        input=messages,
        api_key=api_key,
        original_response=json.dumps(body, default=str),
    )
    response = to_model_response(body, model_response)
    response._hidden_params["additional_headers"] = process_anthropic_headers({})
    return response
