"""
Single chokepoint for reading prompt-cache token counts out of a SpendLog
``usage_object`` dict. Imported by the daily-spend DB writer and by
cost-savings read endpoints so both resolve cache tokens by the same rule.
"""


def extract_cache_read_tokens(usage_obj: dict) -> int:
    """
    Anthropic: top-level cache_read_input_tokens field.
    OpenAI-compatible (moonshotai, openai, deepseek, etc.): prompt_tokens_details.cached_tokens.
    """
    explicit = usage_obj.get("cache_read_input_tokens", 0) or 0
    if explicit:
        return int(explicit)
    details = usage_obj.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens", 0) or 0)


def extract_cache_creation_tokens(usage_obj: dict) -> int:
    """
    Anthropic: top-level cache_creation_input_tokens field.
    OpenAI-compatible (kimi-k2 etc.): prompt_tokens_details.cache_write_tokens
    or prompt_tokens_details.cache_creation_tokens.
    """
    explicit = usage_obj.get("cache_creation_input_tokens", 0) or 0
    if explicit:
        return int(explicit)
    details = usage_obj.get("prompt_tokens_details") or {}
    return int(details.get("cache_write_tokens", 0) or details.get("cache_creation_tokens", 0) or 0)
