"""
MiniMax OpenAI transformation config - extends OpenAI chat config for MiniMax's OpenAI-compatible API
"""

import re
from typing import Any, List, Optional, Tuple

import litellm
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIGPTConfig,
    OpenAIChatCompletionStreamingHandler,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import ModelResponseStream


class MinimaxChatConfig(OpenAIGPTConfig):
    """
    MiniMax OpenAI configuration that extends OpenAIGPTConfig.
    MiniMax provides an OpenAI-compatible API at:
    - International: https://api.minimax.io/v1
    - China: https://api.minimaxi.com/v1

    Supported models:
    - MiniMax-M2.1
    - MiniMax-M2.1-lightning
    - MiniMax-M2
    """

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> str:
        return (
            api_base
            or get_secret_str("MINIMAX_API_BASE")
            or "https://api.minimax.io/v1"
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base)
        if base_url.endswith("/chat/completions"):
            return base_url
        elif base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        elif base_url.endswith("/"):
            return f"{base_url}v1/chat/completions"
        else:
            return f"{base_url}/v1/chat/completions"

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model=model)
        additional_params = ["reasoning_split"]
        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider="minimax"):
                additional_params.append("thinking")
        except Exception:
            pass
        return base_params + additional_params

    def get_model_response_iterator(
        self,
        streaming_response: Any,
        sync_stream: Optional[bool] = None,
        json_mode: Optional[bool] = None,
    ):
        return MinimaxChatResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class MinimaxChatResponseIterator(OpenAIChatCompletionStreamingHandler):
    """
    MiniMax streaming response iterator that handles reasoning_content extraction.

    MiniMax streaming responses can contain:
    1. reasoning_details array - converts to reasoning_content
    2. <think>...</think> tags in content - extracts to reasoning_content
    3. "reasoning" field (alias for reasoning_content) - clears from content
    """

    started_reasoning_content: bool = False
    finished_reasoning_content: bool = False
    pending_reasoning: str = ""

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            choices = chunk.get("choices", [])

            for choice in choices:
                delta = choice.get("delta", {})

                # MiniMax uses "reasoning" (not "reasoning_content") in streaming chunks
                # The parent class maps "reasoning" -> "reasoning_content"
                # We clear content when either reasoning field is present
                has_reasoning = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                if has_reasoning:
                    delta["content"] = None
                    self.started_reasoning_content = True

                reasoning_details = delta.get("reasoning_details")
                if reasoning_details and isinstance(reasoning_details, list):
                    reasoning_text = ""
                    for detail in reasoning_details:
                        if isinstance(detail, dict) and detail.get("text"):
                            reasoning_text += detail.get("text", "")
                    if reasoning_text:
                        delta["reasoning_content"] = reasoning_text
                        delta["content"] = None
                        self.started_reasoning_content = True

                content = delta.get("content", "")
                if content and isinstance(content, str):
                    if "<think>" in content and "</think>" in content:
                        reasonings = re.findall(
                            r"<think>(.*?)</think>", content, re.DOTALL
                        )
                        if reasonings:
                            self.pending_reasoning += "".join(reasonings)
                        clean_content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        )
                        delta["content"] = clean_content if clean_content.strip() else None
                        if self.pending_reasoning:
                            delta["reasoning_content"] = self.pending_reasoning
                            self.started_reasoning_content = True
                            self.pending_reasoning = ""
                    elif "<think>" in content and "</think>" not in content:
                        clean_content = content.replace("<think>", "", 1)
                        delta["content"] = None
                        delta["reasoning_content"] = clean_content.strip()
                        self.started_reasoning_content = True
                    elif "</think>" in content and "<think>" not in content:
                        parts = content.split("</think>", 1)
                        before = parts[0].strip()
                        after = parts[1] if len(parts) > 1 else ""
                        if before:
                            delta["reasoning_content"] = before
                        delta["content"] = after if after.strip() else ("" if not after else None)
                        self.finished_reasoning_content = True
                    elif not self.finished_reasoning_content and content.strip():
                        if not self.started_reasoning_content:
                            self.started_reasoning_content = True
                        delta["reasoning_content"] = content.strip()
                        delta["content"] = None

            return super().chunk_parser(chunk)
        except Exception:
            return super().chunk_parser(chunk)
