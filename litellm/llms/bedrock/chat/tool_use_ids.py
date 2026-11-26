from collections.abc import Sequence
from typing import Final

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.mistral.common_utils import mistral_tool_call_id
from litellm.types.llms.bedrock import ContentBlock, MessageBlock


def with_mistral_tool_use_ids(messages: Sequence[MessageBlock], model: str) -> tuple[MessageBlock, ...]:
    if BaseAWSLLM.get_bedrock_invoke_provider(model) != "mistral":
        return tuple(messages)
    return tuple(_message_with_mistral_tool_use_ids(message) for message in messages)


def _message_with_mistral_tool_use_ids(message: MessageBlock) -> MessageBlock:
    return {**message, "content": [_block_with_mistral_tool_use_id(block) for block in message["content"]]}


def _block_with_mistral_tool_use_id(block: ContentBlock) -> ContentBlock:
    tool_use: Final = block.get("toolUse")
    if tool_use is not None:
        return {**block, "toolUse": {**tool_use, "toolUseId": mistral_tool_call_id(tool_use["toolUseId"])}}
    tool_result: Final = block.get("toolResult")
    if tool_result is not None:
        return {**block, "toolResult": {**tool_result, "toolUseId": mistral_tool_call_id(tool_result["toolUseId"])}}
    return block
