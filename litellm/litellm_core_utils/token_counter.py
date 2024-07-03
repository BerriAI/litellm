# What is this?
## Helper utilities for token counting
from typing import Optional

import litellm
from litellm import verbose_logger


def get_modified_max_tokens(
    model: str,
    base_model: str,
    messages: Optional[list],
    user_max_tokens: Optional[int],
    buffer_perc: Optional[float],
    buffer_num: Optional[float],
) -> Optional[int]:
    """
    Params:

    Returns the user's max output tokens, adjusted for:
    - the size of input - for models where input + output can't exceed X
    - model max output tokens - for models where there is a separate output token limit
    """
    try:
        if user_max_tokens is None:
            return None

        ## MODEL INFO
        _model_info = litellm.get_model_info(model=model)

        max_output_tokens = litellm.get_max_tokens(
            model=base_model
        )  # assume min context window is 4k tokens

        ## UNKNOWN MAX OUTPUT TOKENS - return user defined amount
        if max_output_tokens is None:
            return user_max_tokens

        input_tokens = litellm.token_counter(model=base_model, messages=messages)

        # token buffer
        if buffer_perc is None:
            buffer_perc = 0.1
        if buffer_num is None:
            buffer_num = 10
        token_buffer = max(
            buffer_perc * input_tokens, buffer_num
        )  # give at least a 10 token buffer. token counting can be imprecise.

        input_tokens += int(token_buffer)
        verbose_logger.debug(
            f"max_output_tokens: {max_output_tokens}, user_max_tokens: {user_max_tokens}"
        )
        ## CASE 1: model input + output can't exceed X - happens when max input = max output, e.g. gpt-3.5-turbo
        if _model_info["max_input_tokens"] == max_output_tokens:
            verbose_logger.debug(
                f"input_tokens: {input_tokens}, max_output_tokens: {max_output_tokens}"
            )
            if input_tokens > max_output_tokens:
                pass  # allow call to fail normally - don't set max_tokens to negative.
            elif (
                user_max_tokens + input_tokens > max_output_tokens
            ):  # we can still modify to keep it positive but below the limit
                verbose_logger.debug(
                    f"MODIFYING MAX TOKENS - user_max_tokens={user_max_tokens}, input_tokens={input_tokens}, max_output_tokens={max_output_tokens}"
                )
                user_max_tokens = int(max_output_tokens - input_tokens)
        ## CASE 2: user_max_tokens> model max output tokens
        elif user_max_tokens > max_output_tokens:
            user_max_tokens = max_output_tokens

        verbose_logger.debug(
            f"litellm.litellm_core_utils.token_counter.py::get_modified_max_tokens() - user_max_tokens: {user_max_tokens}"
        )

        return user_max_tokens
    except Exception as e:
        verbose_logger.error(
            "litellm.litellm_core_utils.token_counter.py::get_modified_max_tokens() - Error while checking max token limit: {}\nmodel={}, base_model={}".format(
                str(e), model, base_model
            )
        )
        return user_max_tokens
