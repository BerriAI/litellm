import json
from collections import defaultdict
from typing import Dict, List

from litellm._logging import verbose_logger
from litellm.types.utils import ChatCompletionMessageToolCall, Function


def _handle_invalid_parallel_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCall],
):
    """
    Handle hallucinated parallel tool call from openai - https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Code modified from: https://github.com/phdowling/openai_multi_tool_use_parallel_patch/blob/main/openai_multi_tool_use_parallel_patch.py
    """

    if tool_calls is None:
        return

    try:
        replacements: Dict[int, List[ChatCompletionMessageToolCall]] = defaultdict(list)
        for i, tool_call in enumerate(tool_calls):
            current_function = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            if current_function == "multi_tool_use.parallel":
                verbose_logger.debug(
                    "OpenAI did a weird pseudo-multi-tool-use call, fixing call structure.."
                )
                for _fake_i, _fake_tool_use in enumerate(function_args["tool_uses"]):
                    _function_args = _fake_tool_use["parameters"]
                    _current_function = _fake_tool_use["recipient_name"]
                    if _current_function.startswith("functions."):
                        _current_function = _current_function[len("functions.") :]

                    fixed_tc = ChatCompletionMessageToolCall(
                        id=f"{tool_call.id}_{_fake_i}",
                        type="function",
                        function=Function(
                            name=_current_function, arguments=json.dumps(_function_args)
                        ),
                    )
                    replacements[i].append(fixed_tc)

        shift = 0
        for i, replacement in replacements.items():
            tool_calls[:] = (
                tool_calls[: i + shift] + replacement + tool_calls[i + shift + 1 :]
            )
            shift += len(replacement)

        return tool_calls
    except Exception:
        return tool_calls
