from collections.abc import Coroutine, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Optional,
    cast,  # noqa: TID251  # casts moved verbatim from litellm/main.py
)

import httpx
import openai
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import LiteLLMUnknownProvider
from litellm.litellm_core_utils.completion_timeout import CompletionTimeout, StreamKind
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.get_litellm_params import (
    AWS_CREDENTIAL_KWARGS_KEYS,
    OAUTH_TOKEN_EXCHANGE_KWARGS_KEYS,
    OPTIONAL_KWARGS_KEYS,
    PROVIDER_AFFINITY_HEADER_KWARG_KEY,
    with_control_options,
)
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.hidden_params import HIDDEN_PARAMS_ATTR
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.common_utils import update_messages_with_model_file_ids
from litellm.litellm_core_utils.prompt_templates.factory import function_call_prompt, map_system_message_pt
from litellm.litellm_core_utils.provider_affinity import add_provider_affinity_header
from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout
from litellm.llms.base_llm import BaseConfig
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute, get_vertex_ai_model_route
from litellm.native.chat_mock import MOCK_RESPONSE_TYPE, mock_completion
from litellm.native.chat_providers.agents import (
    complete_a2a,
    complete_langflow,
    complete_langgraph,
    complete_microsoft_365_copilot,
    complete_ragflow,
)
from litellm.native.chat_providers.anthropic import complete_anthropic, complete_anthropic_text
from litellm.native.chat_providers.aws import (
    complete_amazon_nova,
    complete_bedrock,
    complete_bedrock_mantle,
    complete_sagemaker,
    complete_sagemaker_chat,
)
from litellm.native.chat_providers.azure import complete_azure, complete_azure_ai, complete_azure_text
from litellm.native.chat_providers.custom import complete_custom, complete_custom_providers
from litellm.native.chat_providers.gateways import (
    complete_cometapi,
    complete_edenai,
    complete_fal_ai,
    complete_nadir,
    complete_openrouter,
    complete_vercel_ai_gateway,
)
from litellm.native.chat_providers.google import complete_gdc, complete_vertex_ai, complete_vertex_ai_beta
from litellm.native.chat_providers.model_labs import (
    complete_aleph_alpha,
    complete_bytez,
    complete_cohere_chat,
    complete_compactifai,
    complete_deepseek,
    complete_fireworks_ai,
    complete_groq,
    complete_huggingface,
    complete_maritalk,
    complete_minimax,
    complete_mistral,
    complete_nlp_cloud,
    complete_predibase,
    complete_replicate,
    complete_scaledown,
    complete_text_completion_codestral,
    complete_together_ai,
    complete_xai,
)
from litellm.native.chat_providers.openai import (
    complete_aiohttp_openai,
    complete_custom_openai,
    complete_text_completion_inception,
    complete_text_completion_openai,
)
from litellm.native.chat_providers.platforms import (
    complete_cloudflare,
    complete_databricks,
    complete_databricks_agent,
    complete_datarobot,
    complete_gigachat,
    complete_gradient_ai,
    complete_heroku,
    complete_oci,
    complete_ovhcloud,
    complete_sap,
    complete_snowflake,
    complete_watsonx,
    complete_watsonx_text,
)
from litellm.native.chat_providers.self_hosted import (
    complete_hosted_vllm,
    complete_lemonade,
    complete_ollama,
    complete_ollama_chat,
    complete_oobabooga,
    complete_petals,
    complete_triton,
    complete_vllm,
)
from litellm.native.chat_providers.support import get_encoding
from litellm.native.request_support import (
    admission_input_tokens,
    proxy_auth_headers,
    register_custom_pricing_for_request,
)
from litellm.native.route_plan import NativeResponses, plan_chat_route
from litellm.responses.bridge_check import bridge_reasoning_effort, responses_api_bridge_check, str_keyed_mapping
from litellm.types.completion import CompletionDispatchContext
from litellm.types.litellm_params import ControlOptions
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.openai import (
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
    OpenAIWebSearchOptions,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders, ProviderSpecificHeader
from litellm.types.workload_identity import ANTHROPIC_WIF_KWARGS_KEYS, OPENAI_WIF_KWARGS_KEYS
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    ProviderConfigManager,
    exception_type,
    get_litellm_params,
    get_llm_provider,
    get_optional_params,
    peek_reasoning_summary_aliases,
    pre_process_non_default_params,
    reasoning_effort_with_summary_alias,
    strip_reasoning_summary_aliases_from_optional_params,
    supports_httpx_timeout,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession


async def resolve_dispatched_chat_response(
    pending: Coroutine[object, object, ModelResponse | CustomStreamWrapper],
) -> ModelResponse | CustomStreamWrapper:
    return await pending


_ANTHROPIC_ONLY_TOOL_KEYS: Final = frozenset({"input_examples", "eager_input_streaming"})


def _is_claude_tool_target(custom_llm_provider: str | None, model: str) -> bool:
    if custom_llm_provider == "anthropic":
        return True
    model_lower: Final = model.lower()
    if custom_llm_provider == "bedrock":
        return "claude" in model_lower or ("arn:" in model_lower and ":bedrock:" in model_lower)
    if custom_llm_provider == "azure_ai" or custom_llm_provider == "vertex_ai":
        return "claude" in model_lower
    return False


def _history_without_unanswered_tool_search_calls(
    messages: Sequence[object], custom_llm_provider: str | None, provider_config: BaseConfig
) -> tuple[object, ...] | None:
    from litellm.responses.litellm_completion_transformation.tool_search import (
        chat_messages_without_tool_search_calls,
        chat_route_rejects_unanswered_tool_calls,
        unanswered_tool_search_call_ids,
    )

    if not chat_route_rejects_unanswered_tool_calls(custom_llm_provider, provider_config):
        return None
    unanswered_search_call_ids: Final = unanswered_tool_search_call_ids(messages)
    if not unanswered_search_call_ids:
        return None
    return chat_messages_without_tool_search_calls(messages, unanswered_search_call_ids)


def _without_anthropic_only_tool_keys(tool: dict[str, object]) -> dict[str, object]:
    kept: Final = {key: value for key, value in tool.items() if key not in _ANTHROPIC_ONLY_TOOL_KEYS}
    function: Final = str_keyed_mapping(tool.get("function"))
    if function is None:
        return kept
    return {
        **kept,
        "function": {key: value for key, value in function.items() if key not in _ANTHROPIC_ONLY_TOOL_KEYS},
    }


def _drop_anthropic_only_tool_keys(tools: list[dict[str, object]] | None) -> list[dict[str, object]] | None:
    if tools is None:
        return None
    return [_without_anthropic_only_tool_keys(tool) if isinstance(tool, dict) else tool for tool in tools]


OPTIONAL_PARAMS_ADAPTER: Final = TypeAdapter(dict[str, object])


def _bridges_to_responses_api(
    responses_api_model_info: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    model_info: object,
    api_base: str | None,
) -> bool:
    if responses_api_model_info.get("mode") != "responses":
        return False
    from litellm.responses.main import resolve_responses_api_provider_config

    return isinstance(
        plan_chat_route(resolve_responses_api_provider_config(model, custom_llm_provider, model_info, api_base)),
        NativeResponses,
    )


_REQUEST_KWARGS_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_MODEL_INFO_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _azure_completion_fakes_stream(model: str, base_model: str | None) -> bool:
    if not litellm.AzureOpenAIO1Config().is_o_series_model(model=base_model or model):
        return False
    o_series_handler_config: Final = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders.AZURE
    )
    return o_series_handler_config is not None and o_series_handler_config.should_fake_stream(
        model=model, stream=True, custom_llm_provider=LlmProviders.AZURE.value
    )


def _completion_fakes_stream(
    model: str,
    custom_llm_provider: str | None,
    provider_config: BaseConfig | None,
    optional_params: Mapping[str, object],
    *,
    base_model: str | None,
    bridges_to_responses_api: bool,
) -> bool:
    if optional_params.get("fake_stream") is True:
        return True
    if custom_llm_provider is None:
        return False
    if bridges_to_responses_api:
        responses_config: Final = ProviderConfigManager.get_provider_responses_api_config(
            provider=custom_llm_provider, model=model
        )
        return responses_config is not None and responses_config.should_fake_stream(
            model=model, stream=True, custom_llm_provider=custom_llm_provider
        )
    if (
        custom_llm_provider == LlmProviders.VERTEX_AI.value
        and get_vertex_ai_model_route(model=model, litellm_params={"base_model": base_model})
        == VertexAIModelRoute.GEMMA
    ):
        return True
    if isinstance(provider_config, litellm.AmazonAI21Config):
        return True
    if custom_llm_provider == LlmProviders.AZURE.value:
        return _azure_completion_fakes_stream(model, base_model)
    return provider_config is not None and provider_config.should_fake_stream(
        model=model, stream=True, custom_llm_provider=custom_llm_provider
    )


def execute_chat(
    *,
    model: str,
    messages: list,  # mutable-ok: provider steps rebuild the message list, as in completion()
    kwargs: dict,  # mutable-ok: provider steps pop request-scoped keys, as in completion()
    non_default_params: dict[str, object],  # mutable-ok: seeded in place before the provider call
    control_options: ControlOptions,
    prompt_id: str | None,
    prompt_variables: dict | None,  # mutable-ok: forwarded to get_litellm_params as is
    timeout: float | str | httpx.Timeout | openai.Timeout | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stream_options: Mapping[str, object] | None = None,
    stop: str | Sequence[str] | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    modalities: Sequence[ChatCompletionModality] | None = None,
    prediction: ChatCompletionPredictionContentParam | None = None,
    audio: ChatCompletionAudioParam | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: Mapping[str, float] | None = None,
    user: str | None = None,
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    response_format: Mapping[str, object] | type[BaseModel] | None = None,
    seed: int | None = None,
    tools: list[dict[str, object]] | None = None,  # mutable-ok: provider steps rebuild tools, as in completion()
    tool_choice: str | Mapping[str, object] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    parallel_tool_calls: bool | None = None,
    web_search_options: OpenAIWebSearchOptions | None = None,
    include_server_side_tool_invocations: bool | None = None,
    deployment_id: str | None = None,
    extra_headers: dict[str, str] | None = None,  # mutable-ok: forwarded to CompletionDispatchContext as is
    safety_identifier: str | None = None,
    service_tier: str | None = None,
    store: bool | None = None,
    prompt_cache_key: str | None = None,
    functions: Sequence[Mapping[str, object]] | None = None,
    function_call: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    thinking: AnthropicThinkingParam | None = None,
    shared_session: Optional["ClientSession"] = None,
) -> ModelResponse | CustomStreamWrapper:
    api_base = kwargs.get("api_base", None)
    mock_response: Final[MOCK_RESPONSE_TYPE | None] = kwargs.get("mock_response", None)
    mock_tool_calls: Final = kwargs.get("mock_tool_calls", None)
    mock_timeout: Final = cast(bool | None, kwargs.get("mock_timeout", None))
    force_timeout: Final = kwargs.get("force_timeout", 600)  ## deprecated
    logger_fn: Final = kwargs.get("logger_fn", None)
    verbose: Final = kwargs.get("verbose", False)
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    litellm_logging_obj: Final = kwargs.get("litellm_logging_obj", None)
    id: Final = kwargs.get("id", None)
    metadata: Final = kwargs.get("metadata", None)
    model_info: Final = kwargs.get("model_info", None)
    proxy_server_request: Final = kwargs.get("proxy_server_request", None)
    provider_specific_header: Final = cast(
        ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        kwargs.get("provider_specific_header", None),
    )
    headers = kwargs.get("headers", None) or extra_headers

    headers = {} if headers is None else dict(headers)
    if extra_headers is not None:
        headers.update(extra_headers)
    # Inject proxy auth headers if configured
    if litellm.proxy_auth is not None:
        try:
            proxy_headers: Final = proxy_auth_headers(litellm.proxy_auth)
            headers.update(proxy_headers)
        except Exception as e:
            verbose_logger.warning("Failed to get proxy auth headers: %s", e)
    num_retries: Final = kwargs.get(
        "num_retries", None
    )  ## alt. param for 'max_retries'. Use this to pass retries w/ instructor.
    max_retries = kwargs.get("max_retries", None)
    cooldown_time: Final = kwargs.get("cooldown_time", None)
    organization: Final = kwargs.get("organization", None)
    ### VERIFY SSL ###
    ssl_verify: Final = kwargs.get("ssl_verify", None)
    ### CUSTOM MODEL COST ###
    input_cost_per_token: Final = kwargs.get("input_cost_per_token", None)
    output_cost_per_token: Final = kwargs.get("output_cost_per_token", None)
    cost_per_second: Final = kwargs.get("cost_per_second", None)
    input_cost_per_second: Final = kwargs.get("input_cost_per_second", None)
    output_cost_per_second: Final = kwargs.get("output_cost_per_second", None)
    ### CUSTOM PROMPT TEMPLATE ###
    initial_prompt_value: Final = kwargs.get("initial_prompt_value", None)
    roles: Final = kwargs.get("roles", None)
    final_prompt_value: Final = kwargs.get("final_prompt_value", None)
    bos_token: Final = kwargs.get("bos_token", None)
    eos_token: Final = kwargs.get("eos_token", None)
    preset_cache_key: Final = kwargs.get("preset_cache_key", None)
    hf_model_name: Final = kwargs.get("hf_model_name", None)
    supports_system_message: Final = kwargs.get("supports_system_message", None)
    base_model: Final = kwargs.get("base_model", None) or (
        model_info.get("base_model") if isinstance(model_info, dict) else None
    )
    ### DISABLE FLAGS ###
    disable_add_transform_inline_image_block: Final = kwargs.get("disable_add_transform_inline_image_block", None)
    ### TEXT COMPLETION CALLS ###
    text_completion: Final = kwargs.get("text_completion", False)
    ### ASYNC CALLS ###
    acompletion: Final = kwargs.get("acompletion", False)
    client: Final = kwargs.get("client", None)
    ### Admin Controls ###
    no_log: Final = kwargs.get("no-log", False)

    try:
        if base_url is not None:
            api_base = base_url
        is_router_call: Final = any("model_group" in (kwargs.get(k) or ()) for k in ("metadata", "litellm_metadata"))
        if is_router_call:
            max_retries = 0
        elif num_retries is not None:
            max_retries = num_retries
        logging: Final[LiteLLMLoggingObj] = cast(LiteLLMLoggingObj, litellm_logging_obj)
        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in
        model_response: Final = ModelResponse()
        setattr(model_response, "usage", litellm.Usage())
        if (
            kwargs.get("azure", False) is True
        ):  # don't remove flag check, to remain backwards compatible for repos like Codium
            custom_llm_provider = "azure"
        if deployment_id is not None:  # azure llms
            model = deployment_id
            custom_llm_provider = "azure"
        _supplemental_provider_params: Final = {k: kwargs[k] for k in OPTIONAL_KWARGS_KEYS if k in kwargs}
        model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
            litellm_params=(
                GenericLiteLLMParams.model_validate(_supplemental_provider_params)
                if _supplemental_provider_params
                else None
            ),
        )

        ## RESPONSES API BRIDGE LOGIC ## - check early and normalize model name
        responses_api_model_info, model = responses_api_bridge_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            web_search_options=web_search_options,
            api_base=api_base,
            deployment_model_info=str_keyed_mapping(
                cast(object, model_info)  # cast-ok: model_info is the untyped dict kwarg
            ),
        )

        if not _is_claude_tool_target(custom_llm_provider=custom_llm_provider, model=model):
            tools = _drop_anthropic_only_tool_keys(tools=tools)

        if provider_specific_header is not None:
            headers.update(
                ProviderSpecificHeaderUtils.get_provider_specific_headers(
                    provider_specific_header=provider_specific_header,
                    custom_llm_provider=custom_llm_provider,
                )
            )

        if model_response is not None and hasattr(model_response, HIDDEN_PARAMS_ATTR):
            model_response_hidden_params: Final = cast(  # cast-ok: preserve dynamic mapping behavior
                dict[str, object], getattr(model_response, HIDDEN_PARAMS_ATTR)
            )
            model_response_hidden_params["custom_llm_provider"] = custom_llm_provider
            model_response_hidden_params["region_name"] = kwargs.get(
                "aws_region_name", None
            )  # support region-based pricing for bedrock

        ### REGISTER CUSTOM MODEL PRICING -- IF GIVEN ###
        if (
            (input_cost_per_token is not None and output_cost_per_token is not None)
            or input_cost_per_second is not None
            or output_cost_per_second is not None
            or cost_per_second is not None
        ):
            register_custom_pricing_for_request(
                model=model,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
                model_info=model_info,
            )
        ### BUILD CUSTOM PROMPT TEMPLATE -- IF GIVEN ###
        custom_prompt_dict = {}
        if initial_prompt_value or roles or final_prompt_value or bos_token or eos_token:
            custom_prompt_dict = {model: {}}
            if initial_prompt_value:
                custom_prompt_dict[model]["initial_prompt_value"] = initial_prompt_value
            if roles:
                custom_prompt_dict[model]["roles"] = roles
            if final_prompt_value:
                custom_prompt_dict[model]["final_prompt_value"] = final_prompt_value
            if bos_token:
                custom_prompt_dict[model]["bos_token"] = bos_token
            if eos_token:
                custom_prompt_dict[model]["eos_token"] = eos_token

        messages = update_messages_with_model_file_ids(
            messages=messages,
            model_id=(kwargs.get("model_info") or {}).get("id", None),
            model_file_id_mapping=cast(
                dict[str, dict[str, str]],
                kwargs.get("model_file_id_mapping") or {},
            ),
        )

        provider_config: BaseConfig | None = None
        if custom_llm_provider is not None and custom_llm_provider in [provider.value for provider in LlmProviders]:
            provider_config = ProviderConfigManager.get_provider_chat_config(
                model=model,
                provider=LlmProviders(custom_llm_provider),
                base_model=base_model,
            )

        if provider_config is not None:
            translated_messages: Final = provider_config.translate_developer_role_to_system_role(messages=messages)
            history_for_route: Final = _history_without_unanswered_tool_search_calls(
                translated_messages, custom_llm_provider, provider_config
            )
            messages = (  # rebind-ok: completion() rewrites messages step by step for the provider
                translated_messages if history_for_route is None else list(history_for_route)
            )

        if (
            supports_system_message is not None
            and isinstance(supports_system_message, bool)
            and supports_system_message is False
        ):
            messages = map_system_message_pt(messages=messages)

        if dynamic_api_key is not None:
            api_key = dynamic_api_key
        # check if user passed in any of the OpenAI optional params
        deployment_model_info: Final = cast(object, kwargs.get("model_info"))  # cast-ok: untyped request kwargs
        bridges_to_responses_api: Final = _bridges_to_responses_api(
            responses_api_model_info, model, custom_llm_provider, deployment_model_info, api_base
        )
        allowed_openai_params: Final[list[str] | None] = (
            [*(kwargs.get("allowed_openai_params") or []), "reasoning_effort"]
            if bridges_to_responses_api
            else kwargs.get("allowed_openai_params")
        )
        optional_param_args: Final = {
            "functions": functions,
            "function_call": function_call,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stream_options": stream_options,
            "stop": stop,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "modalities": modalities,
            "prediction": prediction,
            "audio": audio,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            # params to identify the model
            "model": model,
            "custom_llm_provider": custom_llm_provider,
            "response_format": response_format,
            "seed": seed,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_retries": max_retries,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "api_version": api_version,
            "parallel_tool_calls": parallel_tool_calls,
            "messages": messages,
            "reasoning_effort": reasoning_effort,
            "verbosity": verbosity,
            "thinking": thinking,
            "web_search_options": web_search_options,
            "include_server_side_tool_invocations": (
                include_server_side_tool_invocations
                if include_server_side_tool_invocations is not None
                else kwargs.get("include_server_side_tool_invocations")
            ),
            "safety_identifier": safety_identifier,
            "service_tier": service_tier,
            "store": store,
            "prompt_cache_key": prompt_cache_key,
            "allowed_openai_params": allowed_openai_params,
            "base_model": base_model,
        }
        provider_reasoning_effort: Final = reasoning_effort_with_summary_alias(
            reasoning_effort,
            non_default_params,
            accepts_summary=provider_config is not None
            and provider_config.accepts_reasoning_summary_in_reasoning_effort,
        )

        mapping_args: Final = MappingProxyType({**optional_param_args, "reasoning_effort": provider_reasoning_effort})
        mapping_special_params: Final = MappingProxyType(dict(non_default_params))

        def optional_params_for_model_info(deployment_info: Mapping[str, object] | None) -> dict[str, object]:
            return get_optional_params(**mapping_args, model_info=deployment_info, **mapping_special_params)

        optional_params = optional_params_for_model_info(
            _MODEL_INFO_ADAPTER.validate_python(model_info) if isinstance(model_info, dict) else None
        )
        fakes_stream: Final = partial(
            _completion_fakes_stream,
            model,
            custom_llm_provider,
            provider_config,
            base_model=base_model if isinstance(base_model, str) else None,
            bridges_to_responses_api=bridges_to_responses_api is True,
        )
        resolve_timeout: Final = partial(
            CompletionTimeout.resolve,
            timeout,
            kwargs,
            custom_llm_provider,
            global_timeout=get_configured_request_timeout(),
            supports_httpx_timeout=supports_httpx_timeout,
        )
        streams: Final = stream is True

        def timeout_for_optional_params(attempt_params: Mapping[str, object]) -> float | httpx.Timeout:
            if not streams:
                return resolve_timeout(stream_kind=StreamKind.NOT_STREAMED)
            return resolve_timeout(stream_kind=StreamKind.FAKED if fakes_stream(attempt_params) else StreamKind.LIVE)

        ### TIMEOUT LOGIC ###
        timeout = timeout_for_optional_params(optional_params)
        logging.record_applied_timeout(timeout)
        processed_non_default_params: Final = pre_process_non_default_params(
            model=model,
            passed_params=optional_param_args,
            special_params=non_default_params,
            custom_llm_provider=custom_llm_provider,
            additional_drop_params=kwargs.get("additional_drop_params"),
            remove_sensitive_keys=True,
            add_provider_specific_params=True,
            provider_config=provider_config,
        )

        if litellm.add_function_to_prompt and optional_params.get(
            "functions_unsupported_model", None
        ):  # if user opts to add it to prompt, when API doesn't support function calling
            functions_unsupported_model: Final[list[object]] = TypeAdapter(list[object]).validate_python(
                optional_params.pop("functions_unsupported_model")
            )
            request_kwargs: Final = _REQUEST_KWARGS_ADAPTER.validate_python(kwargs)
            messages = function_call_prompt(
                messages=messages,
                functions=functions_unsupported_model,
                model=model,
                llm_provider=custom_llm_provider,
                drop_params=litellm.drop_params is True
                or normalize_drop_params(request_kwargs.get("drop_params")) is True,
            )

        # For logging - save the values of the litellm-specific params passed in
        requested_litellm_params: Final = get_litellm_params(
            acompletion=acompletion,
            api_key=api_key,
            force_timeout=force_timeout,
            logger_fn=logger_fn,
            verbose=verbose,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            litellm_call_id=kwargs.get("litellm_call_id", None),
            model_alias_map=litellm.model_alias_map,
            completion_call_id=id,
            metadata=metadata,
            model_info=model_info,
            proxy_server_request=proxy_server_request,
            preset_cache_key=preset_cache_key,
            no_log=no_log,
            cost_per_second=cost_per_second,
            input_cost_per_second=input_cost_per_second,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_second=output_cost_per_second,
            output_cost_per_token=output_cost_per_token,
            cooldown_time=cooldown_time,
            text_completion=kwargs.get("text_completion"),
            azure_ad_token_provider=kwargs.get("azure_ad_token_provider"),
            user_continue_message=kwargs.get("user_continue_message"),
            base_model=base_model,
            litellm_trace_id=kwargs.get("litellm_trace_id"),
            litellm_session_id=kwargs.get("litellm_session_id"),
            hf_model_name=hf_model_name,
            custom_prompt_dict=custom_prompt_dict,
            litellm_metadata=kwargs.get("litellm_metadata"),
            disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
            drop_params=kwargs.get("drop_params"),
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            ssl_verify=ssl_verify,
            merge_reasoning_content_in_choices=kwargs.get("merge_reasoning_content_in_choices", None),
            use_litellm_proxy=kwargs.get("use_litellm_proxy", False),
            api_version=api_version,
            azure_ad_token=kwargs.get("azure_ad_token"),
            tenant_id=kwargs.get("tenant_id"),
            client_id=kwargs.get("client_id"),
            client_secret=kwargs.get("client_secret"),
            azure_username=kwargs.get("azure_username"),
            azure_password=kwargs.get("azure_password"),
            azure_scope=kwargs.get("azure_scope"),
            max_retries=max_retries,
            timeout=timeout,
            litellm_request_debug=kwargs.get("litellm_request_debug", False),
            tpm=kwargs.get("tpm"),
            rpm=kwargs.get("rpm"),
            use_xai_oauth=kwargs.get("use_xai_oauth", False),
            gigachat_scope=kwargs.get("gigachat_scope"),
            gigachat_auth_url=kwargs.get("gigachat_auth_url"),
            gigachat_access_token=kwargs.get("gigachat_access_token"),
            **{
                key: kwargs[key]
                for key in (
                    *AWS_CREDENTIAL_KWARGS_KEYS,
                    *ANTHROPIC_WIF_KWARGS_KEYS,
                    *OPENAI_WIF_KWARGS_KEYS,
                    *OAUTH_TOKEN_EXCHANGE_KWARGS_KEYS,
                    "github_copilot_auth_type",
                    "github_copilot_user_session",
                    PROVIDER_AFFINITY_HEADER_KWARG_KEY,
                    "fireworks_forward_user_id",
                    "responses_api_version",
                )
                if key in kwargs
            },
        )
        litellm_params: Final = with_control_options(requested_litellm_params, control_options)
        if litellm_params.get("provider_affinity_header") is not None:
            try:
                headers = add_provider_affinity_header(
                    headers=headers or litellm.headers or MappingProxyType({}),
                    litellm_params=MappingProxyType(
                        {
                            "provider_affinity_header": litellm_params["provider_affinity_header"],
                            "litellm_session_id": kwargs.get("litellm_session_id"),
                            "session_id": kwargs.get("session_id"),
                            "metadata": metadata,
                            "litellm_metadata": kwargs.get("litellm_metadata"),
                        }
                    ),
                )
            except ValueError as affinity_error:
                raise litellm.BadRequestError(
                    message=str(affinity_error), model=model, llm_provider=custom_llm_provider
                ) from affinity_error
        logging.update_environment_variables(
            model=model,
            user=user,
            optional_params=processed_non_default_params,  # [IMPORTANT] - using processed_non_default_params ensures consistent params logged to langfuse for finetuning / eval datasets.
            litellm_params=litellm_params,
            custom_llm_provider=custom_llm_provider,
        )
        if mock_response or mock_tool_calls or mock_timeout:
            kwargs.pop("mock_timeout", None)  # remove for any fallbacks triggered
            return mock_completion(
                model,
                messages,
                stream=stream,
                n=n,
                mock_response=mock_response,
                mock_tool_calls=mock_tool_calls,
                logging=logging,
                acompletion=acompletion,
                mock_delay=kwargs.get("mock_delay", None),
                custom_llm_provider=custom_llm_provider,
                mock_timeout=mock_timeout,
                timeout=timeout,
                prompt_tokens=admission_input_tokens(
                    cast(Mapping[str, object], kwargs)  # cast-ok: completion's **kwargs is untyped
                ),
            )

        ## RESPONSES API BRIDGE LOGIC ## - check if model has 'mode: responses' in litellm.model_cost map
        # Only run the second bridge check if the first one didn't already
        # detect responses mode (e.g. via the "responses/" prefix).  The second
        # check handles cases like gpt-5.4+ with tools+reasoning_effort or
        # reasoningSummary/reasoning_summary without tools (AI SDK) that the first
        # (early) check doesn't cover.
        _reasoning_summary_for_bridge: Final = peek_reasoning_summary_aliases(optional_params)
        if responses_api_model_info.get("mode") != "responses":
            responses_api_model_info, model = responses_api_bridge_check(
                model=model,
                custom_llm_provider=custom_llm_provider,
                web_search_options=web_search_options,
                tools=tools,
                reasoning_effort=bridge_reasoning_effort(reasoning_effort, optional_params.get("extra_body")),
                reasoning_summary=_reasoning_summary_for_bridge,
                api_base=api_base,
                deployment_model_info=str_keyed_mapping(
                    cast(object, model_info)  # cast-ok: model_info is the untyped dict kwarg
                ),
            )

        # Use base_model (the true underlying model) for Azure model-type
        # detection when the deployment name differs from the model name.
        _azure_detection_model: Final = base_model or model

        if _bridges_to_responses_api(
            responses_api_model_info, model, custom_llm_provider, deployment_model_info, api_base
        ):
            from litellm.completion_extras import responses_api_bridge

            optional_params, rs_val = strip_reasoning_summary_aliases_from_optional_params(optional_params)

            if isinstance(reasoning_effort, dict) and "summary" in reasoning_effort:
                optional_params["reasoning_effort"] = reasoning_effort
            elif rs_val is not None:
                eff: Final = optional_params.get("reasoning_effort", reasoning_effort)
                if isinstance(eff, dict):
                    optional_params["reasoning_effort"] = {**eff, "summary": rs_val}
                elif eff is not None:
                    optional_params["reasoning_effort"] = {
                        "effort": eff,
                        "summary": rs_val,
                    }
                else:
                    optional_params["reasoning_effort"] = {"summary": rs_val}

            return responses_api_bridge.completion(  # pyright: ignore[reportReturnType]  # bridge returns a coroutine on the acompletion path; awaited by the async caller
                model=model,
                messages=messages,
                headers=headers,
                model_response=model_response,
                api_key=api_key,
                api_base=api_base,
                acompletion=acompletion,
                logging_obj=logging,
                optional_params=optional_params,
                litellm_params=litellm_params,
                timeout=timeout,
                client=client,  # pass AsyncOpenAI, OpenAI client
                custom_llm_provider=custom_llm_provider,
                encoding=get_encoding(),
                stream=stream,
                cache=kwargs.get("cache"),
            )
        elif (custom_llm_provider == "openai" and OpenAIGPT5Config.is_model_gpt_5_model(model)) or (
            custom_llm_provider == "azure"
            and litellm.AzureOpenAIGPT5Config.is_model_gpt_5_model(_azure_detection_model)
        ):
            optional_params, _ = strip_reasoning_summary_aliases_from_optional_params(optional_params)

        _dispatch_ctx: Final = CompletionDispatchContext(
            _azure_detection_model=_azure_detection_model,
            acompletion=acompletion,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            client=client,
            custom_llm_provider=custom_llm_provider,
            custom_prompt_dict=custom_prompt_dict,
            extra_headers=extra_headers,
            headers=headers,
            hf_model_name=hf_model_name,
            kwargs=kwargs,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            logging=logging,
            max_retries=max_retries,
            max_tokens=max_tokens,
            messages=messages,
            metadata=metadata,
            model=model,
            model_response=model_response,
            optional_params=optional_params,
            optional_params_for_model_info=optional_params_for_model_info,
            organization=organization,
            provider_config=provider_config,
            request_params=MappingProxyType(
                OPTIONAL_PARAMS_ADAPTER.validate_python({**optional_param_args, **non_default_params})
            ),
            shared_session=shared_session,
            stream=stream,
            temperature=temperature,
            text_completion=text_completion,
            timeout=timeout,
            timeout_for_optional_params=timeout_for_optional_params,
            top_p=top_p,
        )
        if custom_llm_provider == "azure":
            # azure configs
            ## check dynamic params ##
            response = complete_azure(_dispatch_ctx)
        elif custom_llm_provider == "azure_text":
            # azure configs
            response = complete_azure_text(_dispatch_ctx)
        elif custom_llm_provider == "deepseek":
            ## COMPLETION CALL

            response = complete_deepseek(_dispatch_ctx)

        elif custom_llm_provider == "azure_ai":
            response = complete_azure_ai(_dispatch_ctx)
        elif (
            custom_llm_provider == "text-completion-openai"
            or "ft:babbage-002" in model
            or "ft:davinci-002" in model  # support for finetuned completion models
            or custom_llm_provider in litellm.openai_text_completion_compatible_providers
            and kwargs.get("text_completion") is True
        ):
            response = complete_text_completion_openai(_dispatch_ctx)
        elif custom_llm_provider == "fireworks_ai":
            ## COMPLETION CALL
            response = complete_fireworks_ai(_dispatch_ctx)
        elif custom_llm_provider == "together_ai":
            response = complete_together_ai(_dispatch_ctx)
        elif custom_llm_provider == "heroku":
            response = complete_heroku(_dispatch_ctx)

        elif custom_llm_provider == "ragflow":
            ## COMPLETION CALL - RAGFlow uses HTTP handler to support custom URL paths
            response = complete_ragflow(_dispatch_ctx)
        elif custom_llm_provider == "scaledown":
            response = complete_scaledown(_dispatch_ctx)  # rebind-ok: shared provider return
        elif custom_llm_provider == "xai":
            ## COMPLETION CALL
            response = complete_xai(_dispatch_ctx)
        elif custom_llm_provider == "groq":
            response = complete_groq(_dispatch_ctx)
        elif custom_llm_provider == "bedrock_mantle":
            response = complete_bedrock_mantle(_dispatch_ctx)
        elif custom_llm_provider == "a2a":
            # A2A (Agent-to-Agent) Protocol
            # Resolve agent configuration from registry if model format is "a2a/<agent-name>"
            response = complete_a2a(_dispatch_ctx)
        elif custom_llm_provider == "gigachat":
            # GigaChat - Sber AI's LLM (Russia)
            response = complete_gigachat(_dispatch_ctx)

        elif custom_llm_provider == "sap":
            response = complete_sap(_dispatch_ctx)
        elif custom_llm_provider == "aiohttp_openai":
            # NEW aiohttp provider for 10-100x higher RPS
            response = complete_aiohttp_openai(_dispatch_ctx)
        elif custom_llm_provider == "cometapi":
            response = complete_cometapi(_dispatch_ctx)
        elif custom_llm_provider == "minimax":
            response = complete_minimax(_dispatch_ctx)
        elif custom_llm_provider == "hosted_vllm":
            response = complete_hosted_vllm(_dispatch_ctx)
        elif custom_llm_provider == "edenai":
            response = complete_edenai(_dispatch_ctx)  # rebind-ok: dispatch chain binds response per branch
        elif custom_llm_provider == "fal_ai":
            response = complete_fal_ai(_dispatch_ctx)  # rebind-ok: dispatch chain binds response per branch
        elif (
            # A known OpenAI model name only decides the route when nothing else
            # resolved a provider. get_llm_provider() already maps these names to
            # "openai", so a different value here was asked for explicitly (or came
            # from a register_model entry), and the provider config built for it
            # would be handed to the OpenAI handler.
            (model in litellm.open_ai_chat_completion_models and custom_llm_provider in (None, "openai"))
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "baseten"
            or custom_llm_provider == "sambanova"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "openai"
            or custom_llm_provider == "nebius"
            or custom_llm_provider == "wandb"
            or custom_llm_provider == "clarifai"
            or custom_llm_provider in litellm.openai_compatible_providers
            or JSONProviderRegistry.exists(custom_llm_provider)  # JSON-configured providers
            or "ft:gpt-3.5-turbo" in model  # finetune gpt-3.5-turbo
        ):  # allow user to make an openai call with a custom base
            # note: if a user sets a custom base - we should ensure this works
            # allow for the setting of dynamic and stateful api-bases
            response = complete_custom_openai(_dispatch_ctx)

        elif custom_llm_provider == "mistral":
            response = complete_mistral(_dispatch_ctx)
        elif "replicate" in model or custom_llm_provider == "replicate" or model in litellm.replicate_models:
            # Setting the relevant API KEY for replicate, replicate defaults to using os.environ.get("REPLICATE_API_TOKEN")
            response = complete_replicate(_dispatch_ctx)
        elif "clarifai" in model or custom_llm_provider == "clarifai" or model in litellm.clarifai_models:
            pass  # Deprecated - handled in the openai compatible provider section above
        elif custom_llm_provider == "anthropic_text":
            response = complete_anthropic_text(_dispatch_ctx)
        elif custom_llm_provider == "anthropic":
            response = complete_anthropic(_dispatch_ctx)
        elif custom_llm_provider == "nlp_cloud":
            response = complete_nlp_cloud(_dispatch_ctx)
        elif custom_llm_provider == "aleph_alpha":
            response = complete_aleph_alpha(_dispatch_ctx)
        elif custom_llm_provider == "cohere_chat" or custom_llm_provider == "cohere":
            response = complete_cohere_chat(_dispatch_ctx)
        elif custom_llm_provider == "maritalk":
            response = complete_maritalk(_dispatch_ctx)
        elif custom_llm_provider == "amazon_nova":
            response = complete_amazon_nova(_dispatch_ctx)
        elif custom_llm_provider == "huggingface":
            response = complete_huggingface(_dispatch_ctx)
        elif custom_llm_provider == "oci":
            response = complete_oci(_dispatch_ctx)
        elif custom_llm_provider == "compactifai":
            response = complete_compactifai(_dispatch_ctx)
        elif custom_llm_provider == "oobabooga":
            response = complete_oobabooga(_dispatch_ctx)
        elif custom_llm_provider == "databricks":
            response = complete_databricks(_dispatch_ctx)

        elif custom_llm_provider == "datarobot":
            response = complete_datarobot(_dispatch_ctx)
        elif custom_llm_provider == "openrouter":
            response = complete_openrouter(_dispatch_ctx)
        elif custom_llm_provider == "nadir":
            response = complete_nadir(_dispatch_ctx)  # rebind-ok: mirrors sibling provider branches
        elif custom_llm_provider == "vercel_ai_gateway":
            response = complete_vercel_ai_gateway(_dispatch_ctx)
        elif custom_llm_provider == "palm":
            raise ValueError(
                "Palm was decommisioned on October 2024. Please use the `gemini/` route for Gemini Google AI Studio Models. Announcement: https://ai.google.dev/palm_docs/palm?hl=en"
            )
        elif custom_llm_provider == "vertex_ai_beta" or custom_llm_provider == "gemini":
            response = complete_vertex_ai_beta(_dispatch_ctx)

        elif custom_llm_provider == "vertex_ai":
            response = complete_vertex_ai(_dispatch_ctx)
        elif custom_llm_provider == "predibase":
            response = complete_predibase(_dispatch_ctx)
        elif custom_llm_provider == "text-completion-codestral":
            response = complete_text_completion_codestral(_dispatch_ctx)
        elif custom_llm_provider == "text-completion-inception":
            response = complete_text_completion_inception(_dispatch_ctx)
        elif custom_llm_provider in ("sagemaker_chat", "sagemaker_nova"):
            # boto3 reads keys from .env
            # sagemaker_chat: HF Messages API endpoints
            # sagemaker_nova: Nova models on SageMaker (OpenAI-compatible)
            response = complete_sagemaker_chat(_dispatch_ctx)
        elif custom_llm_provider == "sagemaker":
            # boto3 reads keys from .env
            response = complete_sagemaker(_dispatch_ctx)
        elif custom_llm_provider == "bedrock":
            # boto3 reads keys from .env
            response = complete_bedrock(_dispatch_ctx)
        elif custom_llm_provider == "watsonx":
            response = complete_watsonx(_dispatch_ctx)
        elif custom_llm_provider == "watsonx_text":
            response = complete_watsonx_text(_dispatch_ctx)
        elif custom_llm_provider == "vllm":
            response = complete_vllm(_dispatch_ctx)
        elif custom_llm_provider == "ollama":
            response = complete_ollama(_dispatch_ctx)

        elif custom_llm_provider == "ollama_chat":
            response = complete_ollama_chat(_dispatch_ctx)

        elif custom_llm_provider == "triton":
            response = complete_triton(_dispatch_ctx)
        elif custom_llm_provider == "cloudflare":
            response = complete_cloudflare(_dispatch_ctx)

        elif custom_llm_provider == "petals" or model in litellm.petals_models:
            response = complete_petals(_dispatch_ctx)
        elif custom_llm_provider == "snowflake" or model in litellm.snowflake_models:
            response = complete_snowflake(_dispatch_ctx)
        elif custom_llm_provider == "gradient_ai":
            response = complete_gradient_ai(_dispatch_ctx)

        elif custom_llm_provider == "gdc":
            response = complete_gdc(_dispatch_ctx)
        elif custom_llm_provider == "bytez":
            response = complete_bytez(_dispatch_ctx)
        elif custom_llm_provider == "lemonade":
            response = complete_lemonade(_dispatch_ctx)

        elif custom_llm_provider == "ovhcloud" or model in litellm.ovhcloud_models:
            response = complete_ovhcloud(_dispatch_ctx)

        elif custom_llm_provider == "custom":
            response = complete_custom(_dispatch_ctx)

        elif custom_llm_provider in litellm._custom_providers:  # Assume custom LLM provider
            # Get the Custom Handler
            response = complete_custom_providers(_dispatch_ctx)

        elif custom_llm_provider == "langgraph":
            # LangGraph - Agent Runtime Provider
            response = complete_langgraph(_dispatch_ctx)

        elif custom_llm_provider == "microsoft_365_copilot":
            response = complete_microsoft_365_copilot(_dispatch_ctx)  # rebind-ok: shared provider return

        elif custom_llm_provider == "langflow":
            # LangFlow - Visual AI Agent Platform
            response = complete_langflow(_dispatch_ctx)

        elif custom_llm_provider == "databricks_agent":
            response = complete_databricks_agent(_dispatch_ctx)  # rebind-ok: dispatch chain binds response per branch

        else:
            raise LiteLLMUnknownProvider(model=model, custom_llm_provider=custom_llm_provider)
        return response
    except Exception as e:  # noqa: BLE001  # every provider failure maps to the litellm exception completion() raised
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={"messages": messages},
            extra_kwargs=kwargs,
        )
