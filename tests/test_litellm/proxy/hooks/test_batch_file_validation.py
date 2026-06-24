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
