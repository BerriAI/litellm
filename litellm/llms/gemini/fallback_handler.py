from typing import Any, Awaitable, Callable, Dict

from litellm._logging import verbose_logger
from litellm.llms.gemini.common_utils import should_fallback_to_google_code_assist
from litellm.llms.google_code_assist.chat import GoogleCodeAssistChat

_google_code_assist_chat = GoogleCodeAssistChat()


async def run_gemini_acompletion_with_code_assist_fallback(
    primary_call: Awaitable[Any],
    fallback_kwargs: Dict[str, Any],
    auto_fallback_to_google_code_assist: bool = False,
) -> Any:
    """
    Execute Gemini async completion and fallback to Google Code Assist when
    OAuth scope is insufficient.
    """
    try:
        return await primary_call
    except Exception as e:
        if not auto_fallback_to_google_code_assist:
            raise e

        if not should_fallback_to_google_code_assist(e):
            raise e

        verbose_logger.warning(
            "Gemini request failed with ACCESS_TOKEN_SCOPE_INSUFFICIENT. "
            "Falling back to google_code_assist."
        )
        return await _google_code_assist_chat.acompletion(**fallback_kwargs)


def run_gemini_completion_with_code_assist_fallback(
    primary_call: Callable[[], Any],
    fallback_kwargs: Dict[str, Any],
    auto_fallback_to_google_code_assist: bool = False,
) -> Any:
    """
    Execute Gemini sync completion and fallback to Google Code Assist when
    OAuth scope is insufficient.
    """
    try:
        return primary_call()
    except Exception as e:
        if not auto_fallback_to_google_code_assist:
            raise e

        if not should_fallback_to_google_code_assist(e):
            raise e

        verbose_logger.warning(
            "Gemini request failed with ACCESS_TOKEN_SCOPE_INSUFFICIENT. "
            "Falling back to google_code_assist."
        )
        return _google_code_assist_chat.completion(**fallback_kwargs)
