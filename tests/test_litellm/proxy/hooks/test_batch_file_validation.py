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


def _models(file_content_as_dict):
    """Distinct body.model values, mirroring how the rate limiter collects the
    models from a streamed batch file before the access check."""
    return [
        entry["body"]["model"]
        for entry in file_content_as_dict
        if (entry.get("body") or {}).get("model")
    ]


# ---------------------------------------------------------------------------
# Token counter — covers all three batch payload shapes
# ---------------------------------------------------------------------------


def test_token_counter_counts_chat_messages():
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            }
        }
    )
    assert tokens > 0


def test_token_counter_counts_text_completion_prompt():
    """Pre-fix this returned 0 tokens (the counter only inspected
    `messages`), letting `prompt`-style batches slip past TPM limits."""
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {"body": {"model": "gpt-3.5-turbo-instruct", "prompt": "hello world"}}
    )
    assert tokens > 0


def test_token_counter_counts_embedding_input_string():
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {"body": {"model": "text-embedding-3-small", "input": "hello world"}}
    )
    assert tokens > 0


def test_token_counter_counts_embedding_input_list():
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {
            "body": {
                "model": "text-embedding-3-small",
                "input": ["hello", "world"],
            }
        }
    )
    assert tokens > 0


def test_token_counter_counts_text_completion_prompt_list():
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {
            "body": {
                "model": "gpt-3.5-turbo-instruct",
                "prompt": ["alpha", "beta"],
            }
        }
    )
    assert tokens > 0


def test_token_counter_counts_pre_tokenized_prompt_int_list():
    """OpenAI's text-completion API accepts a single pre-tokenized prompt as
    a list of ints. Each int is one token; pre-fix this shape was silently
    counted as zero, leaving a TPM bypass."""
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {
            "body": {
                "model": "gpt-3.5-turbo-instruct",
                "prompt": [1, 2, 3, 4, 5],
            }
        }
    )
    assert tokens == 5


def test_token_counter_counts_pre_tokenized_prompt_list_of_int_lists():
    """Multiple pre-tokenized prompts (`list[list[int]]`) — the most
    important bypass shape. A 1000-token batch must report 1000 tokens,
    not zero."""
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {
            "body": {
                "model": "gpt-3.5-turbo-instruct",
                "prompt": [[1] * 250, [2] * 250, [3] * 500],
            }
        }
    )
    assert tokens == 1000


def test_token_counter_counts_pre_tokenized_input_for_embeddings():
    """Same shape applies to embeddings (`input`)."""
    from litellm.batches.batch_utils import _count_entry_tokens

    tokens = _count_entry_tokens(
        {
            "body": {
                "model": "text-embedding-3-small",
                "input": [[1, 2, 3], [4, 5, 6]],
            }
        }
    )
    assert tokens == 6


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
                models=_models(file_dict),
            )

    assert exc.value.status_code == 403
    assert "gpt-4o" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_pre_call_allows_all_team_models_key_when_model_in_team_allowlist():
    """Keys with ``all-team-models`` must inherit the team allowlist when
    validating models embedded in batch JSONL."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    proxy_alias = "openai/openai/gpt-5.5-batch"
    file_dict = [
        {
            "body": {
                "model": proxy_alias,
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]
    user = UserAPIKeyAuth(
        api_key="sk-team",
        user_id="alice",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[proxy_alias],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    with patch("litellm.proxy.proxy_server.llm_router", None):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
        )


@pytest.mark.asyncio
async def test_pre_call_uses_current_team_allowlist_for_all_team_models_key():
    from litellm.proxy._types import LiteLLM_TeamTable, SpecialModelNames
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    stale_model = "stale-model"
    current_model = "current-model"
    file_dict = [
        {
            "body": {
                "model": stale_model,
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]
    user = UserAPIKeyAuth(
        api_key="sk-team",
        user_id="alice",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[stale_model],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-123",
        models=[current_model],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=AsyncMock(return_value=team_object),
        ) as mock_get_team_object,
        pytest.raises(HTTPException) as exc_info,
    ):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
        )

    assert exc_info.value.status_code == 403
    mock_get_team_object.assert_awaited_once()


@pytest.mark.asyncio
async def test_pre_call_allows_all_team_models_key_via_current_team_object():
    """Happy path for the team_object branch: with a DB client present, an
    ``all-team-models`` key whose batch model is on the *current* team
    allowlist must be authorized through the freshly-fetched team object,
    not the cached-``team_models`` fallback."""
    from litellm.proxy._types import LiteLLM_TeamTable, SpecialModelNames
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    current_model = "current-model"
    file_dict = [
        {
            "body": {
                "model": current_model,
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]
    user = UserAPIKeyAuth(
        api_key="sk-team",
        user_id="alice",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=["stale-model"],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-123",
        models=[current_model],
    )
    can_key_call_model = AsyncMock(return_value=True)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=AsyncMock(return_value=team_object),
        ) as mock_get_team_object,
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new=can_key_call_model,
        ),
    ):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
        )

    mock_get_team_object.assert_awaited_once()
    can_key_call_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_call_denies_all_team_models_key_via_member_scope():
    """The team_object branch must also apply the per-member model scope: a
    model on the team allowlist but outside the member's ``allowed_models``
    must be rejected with a 403."""
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_TeamMembership,
        LiteLLM_TeamTable,
        SpecialModelNames,
    )
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    team_model = "team-model"
    file_dict = [
        {
            "body": {
                "model": team_model,
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]
    user = UserAPIKeyAuth(
        api_key="sk-team",
        user_id="alice",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[team_model],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    team_object = LiteLLM_TeamTable(team_id="team-123", models=[team_model])
    membership = LiteLLM_TeamMembership(
        user_id="alice",
        team_id="team-123",
        litellm_budget_table=LiteLLM_BudgetTable(allowed_models=["other-model"]),
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=AsyncMock(return_value=team_object),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new=AsyncMock(return_value=membership),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
        )

    assert exc_info.value.status_code == 403
    assert team_model in str(exc_info.value.detail)


@pytest.mark.parametrize(
    ("team_fetch_error", "expected_status"),
    [
        (HTTPException(status_code=404, detail="team not found"), 404),
        (Exception("team fetch failed"), 403),
    ],
)
@pytest.mark.asyncio
async def test_pre_call_fails_closed_when_current_team_fetch_fails_for_all_team_models_key(
    team_fetch_error, expected_status
):
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    stale_model = "stale-model"
    file_dict = [
        {
            "body": {
                "model": stale_model,
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]
    user = UserAPIKeyAuth(
        api_key="sk-team",
        user_id="alice",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[stale_model],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=AsyncMock(side_effect=team_fetch_error),
        ) as mock_get_team_object,
        patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new=AsyncMock(return_value=True),
        ) as mock_can_key_call_model,
        pytest.raises(HTTPException) as exc_info,
    ):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
        )

    assert exc_info.value.status_code == expected_status
    mock_get_team_object.assert_awaited_once()
    mock_can_key_call_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_call_allows_teamless_all_team_models_key():
    """A teamless key with all-team-models must be allowed to submit batch jobs
    for any model (same as leaving models empty = unrestricted). Fails if
    someone re-introduces a teamless denial in _resolve_key_models_for_auth_check
    or adds a team_id guard that blocks the batch path."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    file_dict = [
        {
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "x"}],
            }
        }
    ]
    user = UserAPIKeyAuth(
        api_key="sk-orphan",
        user_id="alice",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    with patch("litellm.proxy.proxy_server.llm_router", None):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
        )


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
            models=_models(file_dict),
        )


@pytest.mark.asyncio
async def test_pre_call_skips_file_fetch_when_disabled_in_general_settings():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    user = UserAPIKeyAuth(api_key="sk-ok", user_id="alice", models=["*"])

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"disable_batch_input_file_rate_limiting": True},
    ):
        result = await rate_limiter.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={"input_file_id": "file-abc123"},
            call_type="acreate_batch",
        )

    assert result == {"input_file_id": "file-abc123"}
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_skips_file_fetch_for_configured_provider():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    user = UserAPIKeyAuth(api_key="sk-ok", user_id="alice", models=["*"])
    data = {"input_file_id": "file-abc123", "model": "my-vllm-model"}

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_providers": ["hosted_vllm"]},
        ),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            return_value={"custom_llm_provider": "hosted_vllm"},
        ),
        patch("litellm.afile_content", new=AsyncMock()) as mock_afile_content,
    ):
        result = await rate_limiter.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data=data,
            call_type="acreate_batch",
        )

    assert result == data
    # A real skip must short-circuit before any file download or rate-limit
    # work — assert the skip happened rather than the hook's error-recovery
    # path (which also returns data unchanged).
    mock_afile_content.assert_not_awaited()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_does_not_skip_for_spoofed_provider():
    """The provider skip is resolved from trusted deployment credentials, so a
    user-supplied ``custom_llm_provider`` that is not backed by the routing
    deployment must not trigger a skip: the input file must still be fetched
    and the rate-limit counters incremented."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    # An applicable rate limit keeps the no-limits shortcut from firing, so the
    # only thing that could prevent the fetch below is the provider skip. If the
    # spoofed ``custom_llm_provider`` were honored, afile_content would never be
    # awaited.
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 100}}
    ]
    rate_limiter.parallel_request_limiter.atomic_check_and_increment_by_n = AsyncMock(
        return_value={"overall_code": "OK", "statuses": []}
    )
    user = UserAPIKeyAuth(api_key="sk-ok", user_id="alice", models=["*"])

    mock_router = MagicMock()
    mock_router.model_list = []
    mock_router.resolve_model_name_from_model_id.return_value = "my-openai-model"

    mock_content = MagicMock()
    mock_content.content = (
        b'{"body": {"model": "my-openai-model", '
        b'"messages": [{"role": "user", "content": "hi"}]}}\n'
    )

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_providers": ["hosted_vllm"]},
        ),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            return_value={"custom_llm_provider": "openai"},
        ),
        patch(
            "litellm.afile_content", new=AsyncMock(return_value=mock_content)
        ) as mock_afile_content,
    ):
        await rate_limiter.async_pre_call_hook(
            user_api_key_dict=user,
            cache=MagicMock(),
            data={
                "input_file_id": "file-abc123",
                "model": "my-openai-model",
                "custom_llm_provider": "hosted_vllm",
            },
            call_type="acreate_batch",
        )

    # The spoofed provider did not short-circuit the skip decision: the file was
    # fetched and the counters were incremented.
    mock_afile_content.assert_awaited_once()
    rate_limiter.parallel_request_limiter.atomic_check_and_increment_by_n.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_input_file_usage_decodes_model_embedded_file_id():
    import base64

    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    original_file_id = "file-provider-xyz"
    encoded_payload = (
        base64.urlsafe_b64encode(
            f"litellm:{original_file_id};model,my-vllm-batch".encode()
        )
        .decode()
        .rstrip("=")
    )
    encoded_file_id = f"file-{encoded_payload}"

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )

    mock_content = MagicMock()
    mock_content.content = b'{"custom_id": "1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "my-vllm-batch", "messages": [{"role": "user", "content": "hi"}]}}\n'

    with (
        patch(
            "litellm.afile_content",
            new=AsyncMock(return_value=mock_content),
        ) as mock_afile_content,
        patch(
            "litellm.proxy.proxy_server.llm_router",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            return_value={
                "api_key": "test-key",
                "api_base": "http://vllm:8000/v1",
                "custom_llm_provider": "hosted_vllm",
            },
        ),
    ):
        await rate_limiter.count_input_file_usage(
            file_id=encoded_file_id,
            custom_llm_provider="openai",
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-ok", user_id="alice"),
            data={},
        )

    mock_afile_content.assert_awaited_once()
    assert mock_afile_content.await_args.kwargs["file_id"] == original_file_id
    assert mock_afile_content.await_args.kwargs["custom_llm_provider"] == "hosted_vllm"


@pytest.mark.asyncio
async def test_pre_call_allows_stripped_provider_model_when_key_has_proxy_alias():
    """After replace_model_in_jsonl, body.model is the provider id (e.g. gpt-5.5).
    Auth must check target_model_names from the unified file id, not reverse-map
    the stripped id."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    proxy_alias = "openai/openai/gpt-5.5-batch"
    file_dict = [
        {"body": {"model": "gpt-5.5", "messages": [{"role": "user", "content": "x"}]}}
    ]
    user = UserAPIKeyAuth(
        api_key="sk-ok",
        user_id="alice",
        models=[proxy_alias],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    mock_router = MagicMock()
    mock_router.model_list = []
    can_key_call_model = AsyncMock(return_value=True)

    with (
        patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new=can_key_call_model,
        ),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
            target_model_names=[proxy_alias],
        )

    can_key_call_model.assert_awaited_once()
    assert can_key_call_model.await_args.kwargs["model"] == proxy_alias
    mock_router.resolve_model_name_from_model_id.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_list_order",
    [
        [
            "openai/openai/gpt-5.5",
            "openai/openai/gpt-5.5-batch",
            "us/azure/openai/gpt-5.5",
        ],
        [
            "us/azure/openai/gpt-5.5",
            "openai/openai/gpt-5.5",
            "openai/openai/gpt-5.5-batch",
        ],
        [
            "openai/openai/gpt-5.5-batch",
            "us/azure/openai/gpt-5.5",
            "openai/openai/gpt-5.5",
        ],
    ],
)
async def test_pre_call_uses_target_model_names_not_stripped_reverse_lookup(
    model_list_order,
):
    """LIT-3593: three deployments strip to gpt-5.5; auth must use the upload
    target alias from target_model_names, not first-match reverse lookup."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    batch_alias = "openai/openai/gpt-5.5-batch"
    deployment_templates = {
        "openai/openai/gpt-5.5": {
            "model_name": "openai/openai/gpt-5.5",
            "litellm_params": {"model": "openai/gpt-5.5"},
            "model_info": {"id": "openai/openai/gpt-5.5", "mode": "chat"},
        },
        "openai/openai/gpt-5.5-batch": {
            "model_name": "openai/openai/gpt-5.5-batch",
            "litellm_params": {"model": "openai/gpt-5.5"},
            "model_info": {"id": "openai/openai/gpt-5.5-batch", "mode": "batch"},
        },
        "us/azure/openai/gpt-5.5": {
            "model_name": "us/azure/openai/gpt-5.5",
            "litellm_params": {"model": "azure/gpt-5.5"},
            "model_info": {"id": "openai/openai/gpt-5.5", "mode": "chat"},
        },
    }
    mock_router = MagicMock()
    mock_router.model_list = [deployment_templates[name] for name in model_list_order]

    def _resolve(model_id):
        for deployment in mock_router.model_list:
            actual_model = deployment.get("litellm_params", {}).get("model")
            if actual_model == model_id or (
                actual_model and actual_model.endswith(f"/{model_id}")
            ):
                return deployment.get("model_name")
        return None

    mock_router.resolve_model_name_from_model_id.side_effect = _resolve

    file_dict = [
        {"body": {"model": "gpt-5.5", "messages": [{"role": "user", "content": "x"}]}}
    ]
    user = UserAPIKeyAuth(
        api_key="sk-ok",
        user_id="alice",
        models=[batch_alias],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    can_key_call_model = AsyncMock(return_value=True)

    with (
        patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new=can_key_call_model,
        ),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await rate_limiter._enforce_batch_file_model_access(
            user_api_key_dict=user,
            models=_models(file_dict),
            target_model_names=[batch_alias],
        )

    can_key_call_model.assert_awaited_once()
    assert can_key_call_model.await_args.kwargs["model"] == batch_alias
    mock_router.resolve_model_name_from_model_id.assert_not_called()


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
        models=_models([]),
    )
    await rate_limiter._enforce_batch_file_model_access(
        user_api_key_dict=user,
        models=_models([{"body": {}}]),
    )


# ---------------------------------------------------------------------------
# Skip-path helpers
# ---------------------------------------------------------------------------


def _make_rate_limiter():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    return _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )


def test_get_batch_routing_model_uses_request_model_for_plain_file():
    rate_limiter = _make_rate_limiter()
    assert (
        rate_limiter._get_batch_routing_model({"model": "gpt-4o-mini"}) == "gpt-4o-mini"
    )


def test_get_batch_routing_model_prefers_file_bound_over_request_model():
    """``create_batch`` routes a model-embedded file id on its bound model and
    ignores the top-level ``model``. The skip decision must use the same
    precedence, otherwise a caller could point ``model`` at a skip-listed
    provider while the file routes a rate-limited one."""
    import base64

    rate_limiter = _make_rate_limiter()
    encoded = (
        base64.urlsafe_b64encode(b"litellm:file-xyz;model,vllm-batch")
        .decode()
        .rstrip("=")
    )
    assert (
        rate_limiter._get_batch_routing_model(
            {"input_file_id": f"file-{encoded}", "model": "gpt-4o-mini"}
        )
        == "vllm-batch"
    )


def test_get_batch_routing_model_returns_none_without_model_or_file():
    rate_limiter = _make_rate_limiter()
    assert rate_limiter._get_batch_routing_model({}) is None
    assert rate_limiter._get_batch_routing_model({"input_file_id": ""}) is None


def test_get_batch_routing_model_decodes_model_embedded_file_id():
    import base64

    rate_limiter = _make_rate_limiter()
    encoded = (
        base64.urlsafe_b64encode(b"litellm:file-xyz;model,vllm-batch")
        .decode()
        .rstrip("=")
    )
    assert (
        rate_limiter._get_batch_routing_model({"input_file_id": f"file-{encoded}"})
        == "vllm-batch"
    )


def test_get_batch_routing_model_uses_unified_file_id_target():
    rate_limiter = _make_rate_limiter()
    with (
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.decode_model_from_file_id",
            return_value=None,
        ),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils._is_base64_encoded_unified_file_id",
            return_value="unified-id",
        ),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_models_from_unified_file_id",
            return_value=["model-a", "model-b"],
        ),
    ):
        assert (
            rate_limiter._get_batch_routing_model({"input_file_id": "file-managed"})
            == "model-a"
        )


def test_key_requires_batch_model_access_check_branches():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    check = _PROXY_BatchRateLimiter._key_requires_batch_model_access_check
    assert check(UserAPIKeyAuth(api_key="sk", models=["*"])) is False
    assert check(UserAPIKeyAuth(api_key="sk", models=["all-proxy-models"])) is False
    assert (
        check(UserAPIKeyAuth(api_key="sk", models=[], access_group_ids=["grp"])) is True
    )
    assert check(UserAPIKeyAuth(api_key="sk", models=[])) is False
    assert check(UserAPIKeyAuth(api_key="sk", models=["gpt-4o-mini"])) is True
    # Wildcard / all-proxy-models grant access to every model, so
    # can_key_call_model passes any model regardless of access groups (which
    # only ever widen access). Such keys must not be forced to download and
    # validate the JSONL even when access_group_ids are also present.
    assert (
        check(UserAPIKeyAuth(api_key="sk", models=["*"], access_group_ids=["grp"]))
        is False
    )
    assert (
        check(
            UserAPIKeyAuth(
                api_key="sk", models=["all-proxy-models"], access_group_ids=["grp"]
            )
        )
        is False
    )
    # A concrete model allowlist is still a subset even with access groups.
    assert (
        check(
            UserAPIKeyAuth(
                api_key="sk", models=["gpt-4o-mini"], access_group_ids=["grp"]
            )
        )
        is True
    )


def test_has_applicable_batch_rate_limits():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    has_limits = _PROXY_BatchRateLimiter._has_applicable_batch_rate_limits
    assert has_limits([{"rate_limit": {"tokens_per_unit": 100}}]) is True
    assert has_limits([{"rate_limit": {"requests_per_unit": 5}}]) is True
    assert has_limits([{"rate_limit": {"max_parallel_requests": 2}}]) is True
    assert has_limits([{"rate_limit": {}}, {}]) is False


def test_should_skip_returns_false_when_key_needs_model_access_check():
    rate_limiter = _make_rate_limiter()
    user = UserAPIKeyAuth(api_key="sk", models=["gpt-4o-mini"])
    should_skip, descriptors = rate_limiter._should_skip_batch_input_file_processing(
        data={"input_file_id": "file-abc"}, user_api_key_dict=user
    )
    assert should_skip is False
    assert descriptors is None


def test_should_skip_ignores_client_supplied_metadata_flag():
    """A caller must not be able to bypass batch rate limits by setting
    ``litellm_metadata.skip_batch_input_file_rate_limiting`` in the request
    body. The skip decision is server-controlled only, so with applicable rate
    limits the JSONL is still processed despite the client flag."""
    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        should_skip, descriptors = (
            rate_limiter._should_skip_batch_input_file_processing(
                data={
                    "input_file_id": "file-abc",
                    "litellm_metadata": {"skip_batch_input_file_rate_limiting": True},
                },
                user_api_key_dict=user,
            )
        )
    assert should_skip is False


def test_should_not_skip_for_forged_model_embedded_file_id():
    """A ``file-<base64>`` id embeds an unsigned model name the caller fully
    controls, so a caller can re-encode any accessible provider file id with a
    skip-listed model while the JSONL still routes rate-limited ``body.model``
    entries. The per-model skip must therefore never fire: with applicable rate
    limits, a forged skip-listed file-bound model still falls through to file
    processing and counter enforcement."""
    import base64

    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    encoded = (
        base64.urlsafe_b64encode(b"litellm:file-xyz;model,gpt-4o-mini")
        .decode()
        .rstrip("=")
    )
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"skip_batch_input_file_rate_limiting_for_models": ["gpt-4o-mini"]},
    ):
        should_skip, descriptors = (
            rate_limiter._should_skip_batch_input_file_processing(
                data={"input_file_id": f"file-{encoded}"},
                user_api_key_dict=user,
            )
        )
    assert should_skip is False
    assert descriptors is not None


def test_should_not_skip_for_skip_listed_top_level_model():
    """A caller must not bypass batch rate limits by naming a skip-listed model
    in the top-level ``model`` while routing a different model through the JSONL
    ``body.model`` entries. No per-model skip exists, so a skip-listed model over
    a plain file still gets processed."""
    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"skip_batch_input_file_rate_limiting_for_models": ["gpt-4o-mini"]},
    ):
        should_skip, descriptors = (
            rate_limiter._should_skip_batch_input_file_processing(
                data={"model": "gpt-4o-mini", "input_file_id": "file-abc"},
                user_api_key_dict=user,
            )
        )
    assert should_skip is False


def test_should_not_skip_when_file_bound_provider_is_rate_limited():
    """A caller must not bypass batch rate limits by pointing the top-level
    ``model`` at a skip-listed provider while the model-embedded ``input_file_id``
    routes to a rate-limited provider. ``create_batch`` runs the batch on the
    file-bound model, so the skip decision must resolve the provider from that
    model and still process the file when its provider is not skip-listed."""
    import base64

    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    encoded = (
        base64.urlsafe_b64encode(b"litellm:file-orig;model,vllm-batch")
        .decode()
        .rstrip("=")
    )

    def _creds(model_id, **kwargs):
        provider = "hosted_vllm" if model_id == "vllm-batch" else "openai"
        return {"custom_llm_provider": provider}

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_providers": ["openai"]},
        ),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            side_effect=_creds,
        ),
    ):
        should_skip, descriptors = (
            rate_limiter._should_skip_batch_input_file_processing(
                data={"input_file_id": f"file-{encoded}", "model": "gpt-skip"},
                user_api_key_dict=user,
            )
        )
    assert should_skip is False
    assert descriptors is not None


def test_should_skip_when_file_bound_provider_is_skip_listed():
    """The provider skip must still fire when the model the batch actually runs
    on (the file-bound model) resolves to a skip-listed provider, even if the
    top-level ``model`` resolves to a different, non-skipped provider."""
    import base64

    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    encoded = (
        base64.urlsafe_b64encode(b"litellm:file-orig;model,vllm-batch")
        .decode()
        .rstrip("=")
    )

    def _creds(model_id, **kwargs):
        provider = "hosted_vllm" if model_id == "vllm-batch" else "openai"
        return {"custom_llm_provider": provider}

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_providers": ["hosted_vllm"]},
        ),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            side_effect=_creds,
        ),
    ):
        should_skip, descriptors = (
            rate_limiter._should_skip_batch_input_file_processing(
                data={"input_file_id": f"file-{encoded}", "model": "gpt-skip"},
                user_api_key_dict=user,
            )
        )
    assert should_skip is True


def test_warns_once_for_unsupported_model_skip_setting():
    """Operators who set the no-op per-model skip key get a single warning so a
    misconfigured deployment does not silently leave batch limits unenforced."""
    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_models": ["gpt-4o-mini"]},
        ),
        patch(
            "litellm.proxy.hooks.batch_rate_limiter.verbose_proxy_logger"
        ) as mock_logger,
    ):
        for _ in range(3):
            rate_limiter._should_skip_batch_input_file_processing(
                data={"model": "gpt-4o-mini", "input_file_id": "file-abc"},
                user_api_key_dict=user,
            )
    assert mock_logger.warning.call_count == 1
    assert (
        "skip_batch_input_file_rate_limiting_for_models"
        in mock_logger.warning.call_args[0][0]
    )


def test_no_warning_when_model_skip_setting_absent():
    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"requests_per_unit": 5}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_providers": ["openai"]},
        ),
        patch(
            "litellm.proxy.hooks.batch_rate_limiter.verbose_proxy_logger"
        ) as mock_logger,
    ):
        rate_limiter._should_skip_batch_input_file_processing(
            data={"model": "gpt-4o-mini", "input_file_id": "file-abc"},
            user_api_key_dict=user,
        )
    mock_logger.warning.assert_not_called()


def test_should_skip_when_no_rate_limits_configured():
    rate_limiter = _make_rate_limiter()
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {}}
    ]
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        should_skip, descriptors = (
            rate_limiter._should_skip_batch_input_file_processing(
                data={"model": "gpt-4o-mini", "input_file_id": "file-abc"},
                user_api_key_dict=user,
            )
        )
    assert should_skip is True
    assert descriptors is None


def test_should_not_skip_and_reuses_descriptors_when_limits_present():
    rate_limiter = _make_rate_limiter()
    descriptors = [{"rate_limit": {"tokens_per_unit": 100}}]
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = (
        descriptors
    )
    user = UserAPIKeyAuth(api_key="sk", models=["*"])
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        should_skip, returned = rate_limiter._should_skip_batch_input_file_processing(
            data={"model": "gpt-4o-mini", "input_file_id": "file-abc"},
            user_api_key_dict=user,
        )
    assert should_skip is False
    assert returned is descriptors


def test_resolve_fetch_params_uses_request_model_credentials():
    rate_limiter = _make_rate_limiter()
    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            return_value={
                "api_key": "k",
                "api_base": "http://vllm:8000/v1",
                "custom_llm_provider": "hosted_vllm",
            },
        ),
    ):
        provider_file_id, fetch_kwargs = (
            rate_limiter._resolve_batch_input_file_fetch_params(
                file_id="file-plain-openai",
                custom_llm_provider="openai",
                data={"model": "my-vllm-batch"},
            )
        )
    assert provider_file_id == "file-plain-openai"
    assert fetch_kwargs["model"] == "my-vllm-batch"
    assert fetch_kwargs["custom_llm_provider"] == "hosted_vllm"
    assert fetch_kwargs["api_base"] == "http://vllm:8000/v1"


def test_resolve_fetch_params_fails_open_on_credential_lookup_error():
    rate_limiter = _make_rate_limiter()
    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            side_effect=HTTPException(status_code=404, detail="no creds"),
        ),
    ):
        provider_file_id, fetch_kwargs = (
            rate_limiter._resolve_batch_input_file_fetch_params(
                file_id="file-plain-openai",
                custom_llm_provider="openai",
                data={"model": "my-vllm-batch"},
            )
        )
    assert provider_file_id == "file-plain-openai"
    assert fetch_kwargs == {"custom_llm_provider": "openai"}


def test_resolve_fetch_params_model_embedded_fails_open_on_credential_error():
    import base64

    rate_limiter = _make_rate_limiter()
    encoded = (
        base64.urlsafe_b64encode(b"litellm:file-orig;model,vllm-batch")
        .decode()
        .rstrip("=")
    )
    encoded_file_id = f"file-{encoded}"

    get_credentials = MagicMock(
        side_effect=HTTPException(status_code=404, detail="no creds")
    )
    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            get_credentials,
        ),
    ):
        provider_file_id, fetch_kwargs = (
            rate_limiter._resolve_batch_input_file_fetch_params(
                file_id=encoded_file_id,
                custom_llm_provider="openai",
                data={},
            )
        )
    get_credentials.assert_called_once()
    assert provider_file_id == "file-orig"
    assert fetch_kwargs == {"custom_llm_provider": "openai"}


@pytest.mark.asyncio
async def test_check_and_increment_computes_descriptors_when_not_passed():
    from litellm.proxy.hooks.batch_rate_limiter import (
        BatchFileUsage,
        _PROXY_BatchRateLimiter,
    )

    parallel_request_limiter = MagicMock()
    parallel_request_limiter._create_rate_limit_descriptors.return_value = [
        {"rate_limit": {"tokens_per_unit": 100}}
    ]
    parallel_request_limiter.atomic_check_and_increment_by_n = AsyncMock(
        return_value={"overall_code": "OK", "statuses": []}
    )
    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=parallel_request_limiter,
    )

    await rate_limiter._check_and_increment_batch_counters(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk", models=["*"]),
        data={"model": "gpt-4o-mini"},
        batch_usage=BatchFileUsage(total_tokens=10, request_count=1),
        descriptors=None,
    )

    parallel_request_limiter._create_rate_limit_descriptors.assert_called_once()


@pytest.mark.asyncio
async def test_count_input_file_usage_raises_on_non_bytes_content():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )

    bad_content = MagicMock()
    bad_content.content = "not-bytes"

    with patch("litellm.afile_content", new=AsyncMock(return_value=bad_content)):
        with pytest.raises(ValueError, match="Expected bytes content"):
            await rate_limiter.count_input_file_usage(
                file_id="file-plain",
                custom_llm_provider="openai",
                user_api_key_dict=UserAPIKeyAuth(api_key="sk", models=["*"]),
                data={},
            )


# Streaming input counting — peak memory must not scale with a full dict list
# ---------------------------------------------------------------------------


def _make_batch_input_bytes(n_rows: int, padding: int = 200) -> bytes:
    import json as _json

    pad = "x" * padding
    rows = []
    for i in range(n_rows):
        rows.append(
            _json.dumps(
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o" if i % 2 else "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": f"{pad} {i}"}],
                    },
                }
            )
        )
    return ("\n".join(rows)).encode("utf-8")


def test_iter_batch_input_entries_matches_dict_list():
    from litellm.batches.batch_utils import (
        _get_file_content_as_dictionary,
        _iter_batch_input_entries,
    )

    raw = _make_batch_input_bytes(50)
    streamed = list(_iter_batch_input_entries(raw))
    assert streamed == _get_file_content_as_dictionary(raw)
    assert streamed[0]["custom_id"] == "request-0"
    # tolerant of blank lines and a missing trailing newline
    assert list(_iter_batch_input_entries(raw + b"\n\n")) == streamed


def test_streaming_count_peak_below_dict_list():
    import gc
    import tracemalloc

    from litellm.batches.batch_utils import (
        _get_file_content_as_dictionary,
        _iter_batch_input_entries,
    )

    raw = _make_batch_input_bytes(8000)

    def _measure(fn):
        gc.collect()
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def _stream():
        count = 0
        models: set = set()
        for entry in _iter_batch_input_entries(raw):
            count += 1
            model = (entry.get("body") or {}).get("model")
            if model:
                models.add(model)
        return count

    def _build_list():
        return len(_get_file_content_as_dictionary(raw))

    stream_peak = _measure(_stream)
    list_peak = _measure(_build_list)
    assert stream_peak < list_peak * 0.5, (
        f"streaming count peak {stream_peak} is not a clear win over the dict "
        f"list {list_peak} (ratio {stream_peak / list_peak:.2f})"
    )


@pytest.mark.asyncio
async def test_count_input_file_usage_streams_without_building_list():
    """count_input_file_usage must count requests/tokens in one streaming pass.
    Mocks the download; asserts the count is correct and that the dict-list
    helper is never called (a revert to the list approach would call it)."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    raw = _make_batch_input_bytes(10)
    fake_content = MagicMock()
    fake_content.content = raw

    with (
        patch("litellm.afile_content", new=AsyncMock(return_value=fake_content)),
        patch(
            "litellm.batches.batch_utils._get_file_content_as_dictionary"
        ) as mock_dict_list,
    ):
        usage = await rate_limiter.count_input_file_usage(
            file_id="file-not-managed",
            custom_llm_provider="openai",
            user_api_key_dict=None,
        )

    assert usage.request_count == 10
    assert usage.total_tokens > 0
    mock_dict_list.assert_not_called()


def _one_row_batch_bytes(model: str) -> bytes:
    import json as _json

    return (
        _json.dumps(
            {
                "custom_id": "r0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": "x"}],
                },
            }
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_count_input_file_usage_enforces_models_when_token_counting_fails():
    """Security regression: a row whose content makes token counting raise must
    NOT skip the model allowlist check. async_pre_call_hook swallows non-HTTP
    exceptions and submits the batch, so a raised counting error would otherwise
    fail open. The access check must still run and deny the restricted model."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    fake_content = MagicMock()
    fake_content.content = _one_row_batch_bytes("restricted-model")
    user = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="bob",
        models=["only-allowed"],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    def _boom(*args, **kwargs):
        raise ValueError("unsupported content part: input_audio")

    deny = AsyncMock(side_effect=Exception("model not in allowlist"))

    with (
        patch("litellm.afile_content", new=AsyncMock(return_value=fake_content)),
        patch("litellm.proxy.hooks.batch_rate_limiter._count_entry_tokens", new=_boom),
        patch("litellm.proxy.auth.auth_checks.can_key_call_model", new=deny),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock(model_list=[])),
    ):
        with pytest.raises(HTTPException) as exc:
            await rate_limiter.count_input_file_usage(
                file_id="file-not-managed",
                custom_llm_provider="openai",
                user_api_key_dict=user,
            )

    # The access check ran despite token counting failing, and denied the model.
    deny.assert_awaited()
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_count_input_file_usage_estimates_tokens_when_counting_fails_for_allowed_model():
    """A token-counting failure for an allowed model must not hard-block the batch
    (the pre-streaming behavior let such batches through), but it also must not
    zero the token total, which would let a caller evade the TPM limit by sending
    rows the counter cannot measure. The row falls back to a conservative
    size-based estimate so the batch proceeds with a non-zero count."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    fake_content = MagicMock()
    fake_content.content = _one_row_batch_bytes("allowed-model")
    user = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="bob",
        models=["allowed-model"],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    def _boom(*args, **kwargs):
        raise ValueError("unsupported content part: file")

    allow = AsyncMock(return_value=True)

    with (
        patch("litellm.afile_content", new=AsyncMock(return_value=fake_content)),
        patch("litellm.proxy.hooks.batch_rate_limiter._count_entry_tokens", new=_boom),
        patch("litellm.proxy.auth.auth_checks.can_key_call_model", new=allow),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock(model_list=[])),
    ):
        usage = await rate_limiter.count_input_file_usage(
            file_id="file-not-managed",
            custom_llm_provider="openai",
            user_api_key_dict=user,
        )

    allow.assert_awaited()
    assert usage.request_count == 1
    # Estimated, not zeroed: a crafted uncountable row can't evade the TPM limit.
    assert usage.total_tokens > 0


@pytest.mark.asyncio
async def test_count_input_file_usage_collects_models_after_malformed_line():
    """A malformed JSONL line must not abort model collection. A restricted model
    named on a row AFTER a malformed line must still be collected and denied by the
    allowlist check, otherwise a caller could hide a restricted model behind a bad
    row."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    fake_content = MagicMock()
    fake_content.content = (
        _one_row_batch_bytes("only-allowed")
        + b"{ this is not valid json\n"
        + _one_row_batch_bytes("restricted-model")
    )
    user = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="bob",
        models=["only-allowed"],
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    async def _deny_restricted(model, **kwargs):
        if model == "restricted-model":
            raise Exception("model not in allowlist")
        return True

    deny = AsyncMock(side_effect=_deny_restricted)

    with (
        patch("litellm.afile_content", new=AsyncMock(return_value=fake_content)),
        patch("litellm.proxy.auth.auth_checks.can_key_call_model", new=deny),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock(model_list=[])),
    ):
        with pytest.raises(HTTPException) as exc:
            await rate_limiter.count_input_file_usage(
                file_id="file-not-managed",
                custom_llm_provider="openai",
                user_api_key_dict=user,
            )

    assert exc.value.status_code == 403
