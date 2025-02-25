from typing import Optional


def _get_base_model_from_litellm_call_metadata(
    metadata: Optional[dict],
) -> Optional[str]:
    if metadata is None:
        return None

    if metadata is not None:
        model_info = metadata.get("model_info", {})

        if model_info is not None:
            base_model = model_info.get("base_model", None)
            if base_model is not None:
                return base_model
    return None


def get_litellm_params(
    api_key: Optional[str] = None,
    force_timeout=600,
    azure=False,
    logger_fn=None,
    verbose=False,
    hugging_face=False,
    replicate=False,
    together_ai=False,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    litellm_call_id=None,
    model_alias_map=None,
    completion_call_id=None,
    metadata: Optional[dict] = None,
    model_info=None,
    proxy_server_request=None,
    acompletion=None,
    aembedding=None,
    preset_cache_key=None,
    no_log=None,
    input_cost_per_second=None,
    input_cost_per_token=None,
    output_cost_per_token=None,
    output_cost_per_second=None,
    cooldown_time=None,
    text_completion=None,
    azure_ad_token_provider=None,
    user_continue_message=None,
    base_model: Optional[str] = None,
    litellm_trace_id: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    custom_prompt_dict: Optional[dict] = None,
    litellm_metadata: Optional[dict] = None,
    disable_add_transform_inline_image_block: Optional[bool] = None,
    drop_params: Optional[bool] = None,
    prompt_id: Optional[str] = None,
    prompt_variables: Optional[dict] = None,
    async_call: Optional[bool] = None,
    ssl_verify: Optional[bool] = None,
    **kwargs,
) -> dict:
    litellm_params = {
        "acompletion": acompletion,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "api_base": api_base,
        "litellm_call_id": litellm_call_id,
        "model_alias_map": model_alias_map,
        "completion_call_id": completion_call_id,
        "aembedding": aembedding,
        "metadata": metadata,
        "model_info": model_info,
        "proxy_server_request": proxy_server_request,
        "preset_cache_key": preset_cache_key,
        "no-log": no_log,
        "stream_response": {},  # litellm_call_id: ModelResponse Dict
        "input_cost_per_token": input_cost_per_token,
        "input_cost_per_second": input_cost_per_second,
        "output_cost_per_token": output_cost_per_token,
        "output_cost_per_second": output_cost_per_second,
        "cooldown_time": cooldown_time,
        "text_completion": text_completion,
        "azure_ad_token_provider": azure_ad_token_provider,
        "user_continue_message": user_continue_message,
        "base_model": base_model
        or _get_base_model_from_litellm_call_metadata(metadata=metadata),
        "litellm_trace_id": litellm_trace_id,
        "hf_model_name": hf_model_name,
        "custom_prompt_dict": custom_prompt_dict,
        "litellm_metadata": litellm_metadata,
        "disable_add_transform_inline_image_block": disable_add_transform_inline_image_block,
        "drop_params": drop_params,
        "prompt_id": prompt_id,
        "prompt_variables": prompt_variables,
        "async_call": async_call,
        "ssl_verify": ssl_verify,
    }
    return litellm_params
