openai_to_bytez_param_map = {
    "stream": "stream",
    "max_tokens": "max_new_tokens",
    "max_completion_tokens": "max_new_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "n": "num_return_sequences",
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
    "tools": False,
    "tool_choice": False,
    "function_call": False,
    "functions": False,
    "max_retries": False,
    "extra_headers": False,
    "parallel_tool_calls": False,
    "audio": False,
    "web_search_options": False,
}


def map_openai_params_to_bytez_params(optional_params: dict, drop_params: bool) -> dict:
    new_optional_params = {}

    for key, value in optional_params.items():

        alias = openai_to_bytez_param_map.get(key)

        if alias is False:
            if drop_params:
                continue

            raise Exception(f"param `{key}` is not supported on Bytez")

        if alias is None:
            new_optional_params[key] = value
            continue

        new_optional_params[alias] = value

    return new_optional_params
