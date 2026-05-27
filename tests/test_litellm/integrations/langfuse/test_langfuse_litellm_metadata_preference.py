"""
LIT-3293: For Anthropic-compatible requests sent through `/v1/messages`,
proxy auth metadata is stored in `litellm_metadata`, but the Langfuse callback
previously read `litellm_params["metadata"]`. As a result, LiteLLM spend logs
contained `user_api_key_alias` / `user_api_key_user_id`, while matching Langfuse
observations did not.

These tests pin the desired behavior of `log_event_on_langfuse` resolving
metadata via `get_litellm_metadata_from_kwargs(kwargs)`:
  - prefers `litellm_metadata` when present
  - falls back to `metadata` when only `metadata` is set
  - merges (`litellm_metadata` wins, but spend keys from `metadata` are added)
    when both are present
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def langfuse_logger():
    """Build a LangFuseLogger without actually instantiating the Langfuse SDK."""
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    with patch.object(LangFuseLogger, "__init__", lambda self, *a, **kw: None):
        logger = LangFuseLogger()
    logger.Langfuse = MagicMock()
    logger.langfuse_sdk_version = "2.60.0"
    logger.is_mock_mode = True
    logger.upstream_langfuse = None
    return logger


def _kwargs_with(
    *,
    litellm_metadata=None,
    metadata=None,
    proxy_request_headers=None,
):
    """Construct kwargs the way litellm_logging would for a /v1/messages call."""
    litellm_params = {}
    if metadata is not None:
        litellm_params["metadata"] = metadata
    if litellm_metadata is not None:
        litellm_params["litellm_metadata"] = litellm_metadata
    if proxy_request_headers is not None:
        litellm_params["proxy_server_request"] = {"headers": proxy_request_headers}

    return {
        "litellm_params": litellm_params,
        "litellm_call_id": "test-call-id-LIT3293",
        "optional_params": {},
        "messages": [{"role": "user", "content": "hi"}],
        "call_type": "anthropic_messages",
        "custom_llm_provider": "anthropic",
        "model": "claude-3-5-sonnet-20240620",
        "standard_logging_object": None,
        "response_cost": 0.0,
    }


def _captured_trace_kwargs(langfuse_logger, kwargs):
    """Run log_event_on_langfuse and return the kwargs passed to Langfuse.trace()."""
    response = MagicMock()
    response.id = "msg_test"
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response.system_fingerprint = None
    response.choices = []
    # Ensure isinstance ModelResponse check is False so we go through the
    # generic content path (avoids touching ModelResponse internals).
    from litellm.types.utils import ModelResponse

    response.__class__ = ModelResponse

    trace_obj = MagicMock()
    generation_obj = MagicMock()
    generation_obj.trace_id = kwargs["litellm_call_id"]
    trace_obj.generation.return_value = generation_obj
    langfuse_logger.Langfuse.trace.return_value = trace_obj

    langfuse_logger.log_event_on_langfuse(
        kwargs=kwargs,
        response_obj=response,
        start_time=None,
        end_time=None,
        user_id=None,
    )
    return langfuse_logger.Langfuse.trace.call_args.kwargs


def test_litellm_metadata_is_preferred_over_metadata(langfuse_logger):
    """The /v1/messages bug: only `litellm_metadata` carries proxy auth keys."""
    kwargs = _kwargs_with(
        litellm_metadata={
            "user_api_key_alias": "alias-LIT3293",
            "user_api_key_user_id": "user-LIT3293",
            "trace_name": "from-litellm-metadata",
        },
        metadata=None,
    )
    trace_kwargs = _captured_trace_kwargs(langfuse_logger, kwargs)

    assert trace_kwargs.get("name") == "from-litellm-metadata"


def test_metadata_fallback_preserved_when_litellm_metadata_absent(langfuse_logger):
    """No regression for callers that only set metadata."""
    kwargs = _kwargs_with(
        litellm_metadata=None,
        metadata={"trace_name": "from-metadata-only"},
    )
    trace_kwargs = _captured_trace_kwargs(langfuse_logger, kwargs)
    assert trace_kwargs.get("name") == "from-metadata-only"


def test_litellm_metadata_wins_when_both_set(langfuse_logger):
    """If both are set, litellm_metadata wins (spend keys merged in by helper)."""
    kwargs = _kwargs_with(
        litellm_metadata={"trace_name": "from-litellm-metadata"},
        metadata={"trace_name": "from-metadata"},
    )
    trace_kwargs = _captured_trace_kwargs(langfuse_logger, kwargs)
    assert trace_kwargs.get("name") == "from-litellm-metadata"


def test_header_metadata_still_applied_on_top_of_litellm_metadata(langfuse_logger):
    """add_metadata_from_header still runs after metadata resolution."""
    kwargs = _kwargs_with(
        litellm_metadata={"trace_name": "base"},
        metadata=None,
        proxy_request_headers={"langfuse_trace_name": "header-override"},
    )
    trace_kwargs = _captured_trace_kwargs(langfuse_logger, kwargs)
    # add_metadata_from_header warns + overwrites when trace_name is already present
    assert trace_kwargs.get("name") == "header-override"


def test_empty_kwargs_does_not_crash(langfuse_logger):
    """Defense-in-depth: missing litellm_params should not regress."""
    kwargs = {
        "litellm_params": {},
        "litellm_call_id": "test-call-id-empty",
        "optional_params": {},
        "messages": [],
        "call_type": "completion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "standard_logging_object": None,
        "response_cost": 0.0,
    }
    response = MagicMock()
    response.id = "x"
    response.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    response.system_fingerprint = None
    response.choices = []
    from litellm.types.utils import ModelResponse
    response.__class__ = ModelResponse
    trace_obj = MagicMock()
    generation_obj = MagicMock()
    generation_obj.trace_id = "test-call-id-empty"
    trace_obj.generation.return_value = generation_obj
    langfuse_logger.Langfuse.trace.return_value = trace_obj
    out = langfuse_logger.log_event_on_langfuse(
        kwargs=kwargs,
        response_obj=response,
        start_time=None,
        end_time=None,
        user_id=None,
    )
    assert out is not None
