from unittest.mock import AsyncMock, patch

import pytest

from litellm.llms.gemini.fallback_handler import (
    run_gemini_acompletion_with_code_assist_fallback,
    run_gemini_completion_with_code_assist_fallback,
)


def test_run_gemini_completion_with_code_assist_fallback_disabled():
    def _raise_scope_error():
        raise Exception("ACCESS_TOKEN_SCOPE_INSUFFICIENT")

    with (
        patch(
            "litellm.llms.gemini.fallback_handler.should_fallback_to_google_code_assist",
            return_value=True,
        ),
        patch(
            "litellm.llms.gemini.fallback_handler._google_code_assist_chat.completion"
        ) as mock_completion,
    ):
        with pytest.raises(Exception, match="ACCESS_TOKEN_SCOPE_INSUFFICIENT"):
            run_gemini_completion_with_code_assist_fallback(
                primary_call=_raise_scope_error,
                fallback_kwargs={},
                auto_fallback_to_google_code_assist=False,
            )

        mock_completion.assert_not_called()


@pytest.mark.asyncio
async def test_run_gemini_acompletion_with_code_assist_fallback_enabled():
    async def _raise_scope_error():
        raise Exception("ACCESS_TOKEN_SCOPE_INSUFFICIENT")

    with (
        patch(
            "litellm.llms.gemini.fallback_handler.should_fallback_to_google_code_assist",
            return_value=True,
        ),
        patch(
            "litellm.llms.gemini.fallback_handler._google_code_assist_chat.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
    ):
        mock_acompletion.return_value = "fallback-ok"
        result = await run_gemini_acompletion_with_code_assist_fallback(
            primary_call=_raise_scope_error(),
            fallback_kwargs={},
            auto_fallback_to_google_code_assist=True,
        )

        assert result == "fallback-ok"
        mock_acompletion.assert_awaited_once()
