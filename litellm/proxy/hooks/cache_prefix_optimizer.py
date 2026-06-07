"""
MiMo 缓存前缀优化器
拦截请求，将 System Prompt 中的动态日期移到 messages 末尾，
确保前缀一致性以最大化 Context C��命中率。

原理：
  Claude Code 每次启动会在 System Prompt 开头注入:
    "Today's date is 2026-05-27."
  这行每天变化，导致 MiMo 前缀缓存每天失效。

  本钩子将该日期从 system 消息中剥离，
  以最后一条 user 消息追加注释的形式重新注入，
  保证 system prompt 前缀完全静态。
"""

import re
import logging
import hashlib
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.cache_optimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[CACHE-OPT] %(message)s"))
    logger.addHandler(handler)

# 匹配 Claude Code 注入的日期行
# 格式1: "Today's date is 2026-05-27."
# 格式2: "# currentDate\nToday's date is 2026-05-27."
DATE_PATTERN = re.compile(
    r"(?:# currentDate\s*\n)?Today's date is \d{4}-\d{2}-\d{2}\.\s*\n?",
    re.MULTILINE,
)


def _extract_text_from_system_content(content):
    """从 system message 的 content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic 格式: [{"type": "text", "text": "..."}]
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content else ""


def _rebuild_system_content(original_content, cleaned_text):
    """用清理后的文本重建 system message 的 content 格式"""
    if isinstance(original_content, str):
        return cleaned_text
    if isinstance(original_content, list):
        # 重建为 Anthropic 格式
        new_blocks = []
        for block in original_content:
            if isinstance(block, dict) and block.get("type") == "text":
                new_block = dict(block)
                new_block["text"] = cleaned_text
                new_blocks.append(new_block)
                break  # 只替换第一个 text block
        return new_blocks if new_blocks else cleaned_text
    return cleaned_text


class CachePrefixOptimizer(CustomLogger):
    """
    预处理钩子：将 system prompt 中的动态日期移到末尾，
    确保 messages 前缀每天保持一致。
    """

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            self._optimize_messages(data)
        except Exception as e:
            logger.error(f"Error in pre_call_hook: {e}")
        return None  # 不阻断请求

    def _optimize_messages(self, data):
        messages = data.get("messages")
        if not messages or not isinstance(messages, list):
            return

        # 找到第一条 system 消息
        system_msg = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_msg = msg
                break

        if system_msg is None:
            return

        # 提取并检查是否包含日期
        original_content = system_msg.get("content", "")
        text = _extract_text_from_system_content(original_content)

        match = DATE_PATTERN.search(text)
        if not match:
            # No date found — still log hash for prefix tracking
            prompt_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
            msg_count = len(messages)
            last_role = messages[-1].get("role", "?") if messages else "?"
            logger.info(
                f"SYS_HASH={prompt_hash} "
                f"msgs={msg_count} "
                f"last_role={last_role} "
                f"prompt_len={len(text)} "
                f"(no date)"
            )
            return

        extracted_date = match.group(0).strip()
        # 清理：从 system prompt 中移除日期行
        cleaned_text = DATE_PATTERN.sub("", text, count=1).strip()

        # 重建 system message
        system_msg["content"] = _rebuild_system_content(original_content, cleaned_text)

        # 将日期注入到最后一条 user 消息末尾
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_content = msg.get("content", "")
                if isinstance(user_content, str):
                    msg["content"] = user_content + f"\n\n<!-- {extracted_date} -->"
                elif isinstance(user_content, list):
                    # Anthropic 格式：追加一个 text block
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"<!-- {extracted_date} -->",
                        }
                    )
                break

        logger.info(
            f"Date extracted from system prompt, moved to end: {extracted_date}"
        )

        # Debug: log system prompt hash for cache prefix tracking
        # Compare hashes across requests to detect prefix changes
        prompt_hash = hashlib.sha256(cleaned_text.encode()).hexdigest()[:12]
        msg_count = len(messages)
        last_role = messages[-1].get("role", "?") if messages else "?"
        logger.info(
            f"SYS_HASH={prompt_hash} "
            f"msgs={msg_count} "
            f"last_role={last_role} "
            f"prompt_len={len(cleaned_text)}"
        )


# 实例名，供 config 中引用
cache_optimizer = CachePrefixOptimizer()
