"""
Production message filter callback - removes cache-busting content.
"""

from typing import Optional, Union
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm._logging import verbose_proxy_logger
from litellm.types.utils import CallTypesLiteral


class MessageFilterProd(CustomGuardrail):
    """
    Production message filter that removes dynamic cache-busting content.

    Removes billing headers with unique identifiers (like cch=xxxxx) that
    prevent KV cache utilization. Logs activity via verbose_proxy_logger only.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter_keywords = kwargs.get(
            "filter_keywords", ["x-anthropic-billing-header"]
        )

    def _should_remove_content_block(self, content_block: dict) -> bool:
        """Check if content block starts with filter keywords."""
        if not isinstance(content_block, dict):
            return False
        if content_block.get("type") != "text":
            return False
        text = content_block.get("text", "")
        if not isinstance(text, str):
            return False
        for keyword in self.filter_keywords:
            if text.startswith(keyword):
                verbose_proxy_logger.debug(
                    f"Removing content block starting with '{keyword}' (content redacted, length={len(text)})"
                )
                return True
        return False

    def _filter_content(self, content):
        """Filter content (string or list of blocks)."""
        if isinstance(content, str):
            for keyword in self.filter_keywords:
                if content.startswith(keyword):
                    verbose_proxy_logger.debug(
                        f"Removing string content starting with '{keyword}' (content redacted, length={len(content)})"
                    )
                    return None
            return content
        if isinstance(content, list):
            filtered = [
                block
                for block in content
                if not self._should_remove_content_block(block)
            ]
            if not filtered:
                return None
            return filtered
        return content

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        """Filter cache-busting content from messages and system fields."""
        filtered_count = 0

        # Filter messages array
        messages = data.get("messages")
        if isinstance(messages, list):
            new_messages = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if content is None:
                    new_messages.append(message)
                    continue
                filtered_content = self._filter_content(content)
                if filtered_content != content:
                    if filtered_content is None or filtered_content == "":
                        # Remove the entire message if content is empty/None
                        filtered_count += 1
                        verbose_proxy_logger.info(
                            f"Removed message with cache-busting content from {message.get('role', 'unknown')} message"
                        )
                        continue
                    message["content"] = filtered_content
                    filtered_count += 1
                    verbose_proxy_logger.info(
                        f"Filtered cache-busting content from {message.get('role', 'unknown')} message"
                    )
                new_messages.append(message)
            data["messages"] = new_messages

        # Filter system field
        system = data.get("system")
        if system is not None:
            filtered_system = self._filter_content(system)
            if filtered_system != system:
                if filtered_system is None or filtered_system == "":
                    data["system"] = None
                    filtered_count += 1
                    verbose_proxy_logger.info(
                        "Removed system field with cache-busting content"
                    )
                else:
                    data["system"] = filtered_system
                    filtered_count += 1
                    verbose_proxy_logger.info(
                        "Filtered cache-busting content from system field"
                    )

        if filtered_count > 0:
            verbose_proxy_logger.info(
                f"Message filter: removed cache-busting content from {filtered_count} location(s)"
            )

        return data


# Production instance
message_filter_prod = MessageFilterProd(filter_keywords=["x-anthropic-billing-header:"])
