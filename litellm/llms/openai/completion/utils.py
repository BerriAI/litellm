from typing import List, Union, cast

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    OpenAITextCompletionUserMessage,
)


def is_tokens_or_list_of_tokens(value: List):
    # Check if it's a list of integers (tokens)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return True
    # Check if it's a list of lists of integers (list of tokens)
    if isinstance(value, list) and all(
        isinstance(item, list) and all(isinstance(i, int) for i in item)
        for item in value
    ):
        return True
    return False


def _transform_prompt(
    messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
) -> AllPromptValues:
    if len(messages) == 1:  # base case
        message_content = messages[0].get("content")
        if (
            message_content
            and isinstance(message_content, list)
            and is_tokens_or_list_of_tokens(message_content)
        ):
            openai_prompt: AllPromptValues = cast(AllPromptValues, message_content)
        else:
            openai_prompt = ""
            content = convert_content_list_to_str(cast(AllMessageValues, messages[0]))
            openai_prompt += content
    else:
        prompt_str_list: List[str] = []
        for m in messages:
            try:  # expect list of int/list of list of int to be a 1 message array only.
                content = convert_content_list_to_str(cast(AllMessageValues, m))
                prompt_str_list.append(content)
            except Exception as e:
                raise e
        openai_prompt = prompt_str_list
    return openai_prompt
