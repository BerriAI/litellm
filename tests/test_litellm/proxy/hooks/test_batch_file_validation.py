"""
VERIA-39 regression tests:

- The batch input-file token counter must measure embeddings (`input`)
  and text-completion (`prompt`) payloads, not only chat (`messages`).
- The batch rate-limiter pre-call hook must reject batch files that name
  models the caller is not authorized to use.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


# ---------------------------------------------------------------------------
# Token counter — covers all three batch payload shapes
# ---------------------------------------------------------------------------


def test_token_counter_counts_chat_messages():
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            }
        ]
    )
    assert usage.prompt_tokens > 0


def test_token_counter_counts_text_completion_prompt():
    """Pre-fix this returned 0 tokens (the function only inspected
    `messages`), letting `prompt`-style batches slip past TPM limits."""
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {"body": {"model": "gpt-3.5-turbo-instruct", "prompt": "hello world"}}
        ]
    )
    assert usage.prompt_tokens > 0


def test_token_counter_counts_embedding_input_string():
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {"body": {"model": "text-embedding-3-small", "input": "hello world"}}
        ]
    )
    assert usage.prompt_tokens > 0


def test_token_counter_counts_embedding_input_list():
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {
                "body": {
                    "model": "text-embedding-3-small",
                    "input": ["hello", "world"],
                }
            }
        ]
    )
    assert usage.prompt_tokens > 0


def test_token_counter_counts_text_completion_prompt_list():
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {
                "body": {
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": ["alpha", "beta"],
                }
            }
        ]
    )
    assert usage.prompt_tokens > 0


def test_token_counter_counts_pre_tokenized_prompt_int_list():
    """OpenAI's text-completion API accepts a single pre-tokenized prompt as
    a list of ints. Each int is one token; pre-fix this shape was silently
    counted as zero, leaving a TPM bypass."""
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {
                "body": {
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": [1, 2, 3, 4, 5],
                }
            }
        ]
    )
    assert usage.prompt_tokens == 5


def test_token_counter_counts_pre_tokenized_prompt_list_of_int_lists():
    """Multiple pre-tokenized prompts (`list[list[int]]`) — the most
    important bypass shape. A 1000-token batch must report 1000 tokens,
    not zero."""
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {
                "body": {
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": [[1] * 250, [2] * 250, [3] * 500],
                }
            }
        ]
    )
    assert usage.prompt_tokens == 1000


def test_token_counter_counts_pre_tokenized_input_for_embeddings():
    """Same shape applies to embeddings (`input`)."""
    from litellm.batches.batch_utils import _get_batch_job_input_file_usage

    usage = _get_batch_job_input_file_usage(
        file_content_dictionary=[
            {
                "body": {
                    "model": "text-embedding-3-small",
                    "input": [[1, 2, 3], [4, 5, 6]],
                }
            }
        ]
    )
    assert usage.prompt_tokens == 6


# ---------------------------------------------------------------------------
# Model extractor
# ---------------------------------------------------------------------------


def test_model_extractor_returns_distinct_models():
    from litellm.batches.batch_utils import _get_models_from_batch_input_file_content

    models = _get_models_from_batch_input_file_content(
        [
            {"body": {"model": "gpt-4o", "messages": []}},
            {"body": {"model": "gpt-4o", "messages": []}},  # duplicate
            {"body": {"model": "gpt-4o-mini", "messages": []}},
            {"body": {}},  # missing model
        ]
    )
    assert models == ["gpt-4o", "gpt-4o-mini"]


# ---------------------------------------------------------------------------
# Pre-call hook model validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_rejects_unauthorized_model_in_batch_file():
    """Pre-fix the hook only validated the outer `model` parameter and
    forwarded the file as-is. With this fix, a model named inside the
    JSONL that the caller cannot use must trigger a 403."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )

    # Simulated decoded batch file: caller is restricted to gpt-3.5
    # but the JSONL points at gpt-4o.
    file_dict = [
        {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}}
    ]

    user = UserAPIKeyAuth(
        api_key="sk-restricted",
        user_id="alice",
        models=["gpt-3.5-turbo"],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    # `can_key_call_model` raises a ProxyException for non-allowed models.
    async def _raise_unauthorized(**kwargs):
        raise Exception(
            f"Key not allowed to access model. This key only has access to models={kwargs['valid_token'].models}"
        )

    with (
        patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new=AsyncMock(side_effect=_raise_unauthorized),
        ),
        patch("litellm.proxy.proxy_server.llm_router", None),
    ):
        with pytest.raises(HTTPException) as exc:
            await rate_limiter._enforce_batch_file_model_access(
                user_api_key_dict=user,
                file_content_as_dict=file_dict,
            )

    assert exc.value.status_code == 403
    assert "gpt-4o" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_pre_call_allows_authorized_model_in_batch_file():
    """If every model in the JSONL is on the caller's allowlist, the hook
    must not raise."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )

    file_dict = [
        {
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]

    user = UserAPIKeyAuth(
        api_key="sk-ok",
        user_id="alice",
        models=["gpt-3.5-turbo"],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    with (
        patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new=AsyncMock(return_value=True),
        ),
        patch("litellm.proxy.proxy_server.llm_router", None),
    ):
        # Should not raise
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            file_content_as_dict=file_dict,
        )


@pytest.mark.asyncio
async def test_pre_call_skips_check_when_no_models_present():
    """Files without any `body.model` (corrupt or empty) must not 500;
    the rate limiter logs a warning elsewhere and proceeds."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    user = UserAPIKeyAuth(api_key="sk-ok", user_id="alice")

    # Should not raise even though `can_key_call_model` is the default
    # (would fail). The early-return on empty models keeps the call out
    # entirely.
    await rate_limiter._enforce_batch_file_model_access(
        user_api_key_dict=user,
        file_content_as_dict=[],
    )
    await rate_limiter._enforce_batch_file_model_access(
        user_api_key_dict=user,
        file_content_as_dict=[{"body": {}}],
    )
