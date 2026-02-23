"""
Translates from OpenAI's `/v1/chat/completions` to the VLLM sdk `llm.generate`. 

NOT RECOMMENDED FOR PRODUCTION USE. Use `hosted_vllm/` instead.
"""

from ...hosted_vllm.chat.transformation import HostedVLLMChatConfig


class VLLMConfig(HostedVLLMChatConfig):
    """
    VLLM SDK supports the same OpenAI params as hosted_vllm.
    """

    pass
