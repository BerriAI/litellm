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
    deployment must not trigger a skip."""
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    rate_limiter = _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.return_value = (
        []
    )
    user = UserAPIKeyAuth(api_key="sk-ok", user_id="alice", models=["*"])

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"skip_batch_input_file_rate_limiting_for_providers": ["hosted_vllm"]},
        ),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
            return_value={"custom_llm_provider": "openai"},
        ),
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

    # Reaching descriptor evaluation proves the spoofed provider did not
    # short-circuit the skip decision via the provider allow-list.
    rate_limiter.parallel_request_limiter._create_rate_limit_descriptors.assert_called_once()


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
    Auth must check the proxy model_name the key was granted, not the stripped id."""
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
    mock_router.resolve_model_name_from_model_id.return_value = proxy_alias
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
            file_content_as_dict=file_dict,
        )

    can_key_call_model.assert_awaited_once()
    assert can_key_call_model.await_args.kwargs["model"] == proxy_alias


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


# ---------------------------------------------------------------------------
# Skip-path helpers
# ---------------------------------------------------------------------------


def _make_rate_limiter():
    from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

    return _PROXY_BatchRateLimiter(
        internal_usage_cache=MagicMock(),
        parallel_request_limiter=MagicMock(),
    )


def test_get_batch_routing_model_prefers_request_model():
    rate_limiter = _make_rate_limiter()
    assert (
        rate_limiter._get_batch_routing_model({"model": "gpt-4o-mini"}) == "gpt-4o-mini"
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


def test_matches_skip_list_handles_empty_and_entry_shapes():
    rate_limiter = _make_rate_limiter()
    assert rate_limiter._matches_skip_list("gpt-4o", []) is False
    assert rate_limiter._matches_skip_list("gpt-4o", ["gpt-4o"]) is True
    assert rate_limiter._matches_skip_list("vertex_ai/gemini", ["vertex_ai"]) is True
    assert rate_limiter._matches_skip_list("gpt-4o", [None, "", "claude"]) is False


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


def test_should_skip_honors_litellm_metadata_flag():
    rate_limiter = _make_rate_limiter()
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
    assert should_skip is True
    assert descriptors is None


def test_should_skip_honors_per_model_skip_list():
    rate_limiter = _make_rate_limiter()
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
    assert should_skip is True
    assert descriptors is None


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

    with patch(
        "litellm.proxy.openai_files_endpoints.common_utils.get_credentials_for_model",
        side_effect=HTTPException(status_code=404, detail="no creds"),
    ):
        provider_file_id, fetch_kwargs = (
            rate_limiter._resolve_batch_input_file_fetch_params(
                file_id=encoded_file_id,
                custom_llm_provider="openai",
                data={},
            )
        )
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
