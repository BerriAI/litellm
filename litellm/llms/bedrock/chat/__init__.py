from .converse_handler import BedrockConverseLLM
from .invoke_handler import (
    AmazonAnthropicClaudeStreamDecoder,
    AmazonDeepSeekR1StreamDecoder,
    AWSEventStreamDecoder,
    BedrockLLM,
)


def get_bedrock_event_stream_decoder(
    invoke_provider: str, model: str, sync_stream: bool, json_mode: bool
):
    if invoke_provider == "anthropic":
        decoder: AWSEventStreamDecoder = AmazonAnthropicClaudeStreamDecoder(
            model=model,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        return decoder
    elif invoke_provider == "deepseek_r1":
        decoder = AmazonDeepSeekR1StreamDecoder(
            model=model,
            sync_stream=sync_stream,
        )
        return decoder
    else:
        decoder = AWSEventStreamDecoder(model=model)
        return decoder
