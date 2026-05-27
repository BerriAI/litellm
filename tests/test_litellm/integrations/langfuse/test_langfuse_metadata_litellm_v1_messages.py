"""Regression tests for LIT-3293.

The Langfuse callback used to read proxy-auth metadata only from
``litellm_params["metadata"]``. New endpoints (``/v1/messages``,
``/v1/responses``, ``/v1/batches``, ``/v1/files``, ``/thread*``,
``/assistant*`` — see ``LITELLM_METADATA_ROUTES`` in
``litellm.proxy.litellm_pre_call_utils``) store proxy auth under
``litellm_params["litellm_metadata"]`` instead, so the auth fields
(``user_api_key_alias``, ``user_api_key_user_id``, …) were dropped and
the Langfuse generation name fell back to ``litellm-<call_type>``
instead of ``litellm:<key_alias>``.

The fix routes metadata extraction through
``litellm.litellm_core_utils.core_helpers.get_litellm_metadata_from_kwargs``
which prefers ``litellm_metadata`` and merges spend-tracking fields
from ``metadata`` when both are present.
"""
import os
from datetime import datetime
from unittest.mock import MagicMock

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-test")
os.environ.setdefault("LANGFUSE_MOCK", "true")

from litellm.integrations.langfuse.langfuse import LangFuseLogger  # noqa: E402
from litellm.types.utils import (  # noqa: E402
    Choices,
    Message,
    ModelResponse,
    Usage,
)


PROXY_AUTH_META = {
    "user_api_key": "hashed-key",
    "user_api_key_hash": "hashed-key",
    "user_api_key_alias": "engineering-prod-key",
    "user_api_key_user_id": "user@example.com",
    "user_api_key_team_id": "team-eng",
    "user_api_key_team_alias": "engineering",
    "user_api_key_org_id": None,
    "user_api_key_end_user_id": None,
    "user_api_key_request_route": "/v1/messages",
}


def _build_response() -> ModelResponse:
    return ModelResponse(
        id="msg_test_123",
        created=int(datetime.now().timestamp()),
        model="claude-3-5-sonnet-latest",
        choices=[Choices(index=0, message=Message(role="assistant", content="Hi!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def _build_kwargs(*, litellm_metadata, metadata, call_type="anthropic_messages",
                  model="claude-3-5-sonnet-latest", provider="anthropic"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "litellm_params": {
            "metadata": dict(metadata) if metadata else {},
            "litellm_metadata": dict(litellm_metadata) if litellm_metadata else {},
            "proxy_server_request": {},
            "api_base": "https://api.anthropic.com",
            "custom_llm_provider": provider,
        },
        "litellm_call_id": "call-abc-123",
        "call_type": call_type,
        "optional_params": {"stream": False},
        "standard_logging_object": {
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": dict(litellm_metadata or metadata or {}),
        },
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "custom_llm_provider": provider,
        "user": "user@example.com",
    }


def _run_logger(kwargs, response):
    logger = LangFuseLogger()
    captured = {"trace": None, "generation": None}
    fake_generation = MagicMock()
    fake_trace = MagicMock()

    def cap_gen(**g):
        captured["generation"] = g
        return fake_generation

    def cap_trace(**t):
        captured["trace"] = t
        return fake_trace

    fake_trace.generation = cap_gen
    logger.Langfuse = MagicMock()
    logger.Langfuse.trace = cap_trace
    logger.log_event_on_langfuse(
        kwargs=kwargs,
        response_obj=response,
        start_time=kwargs["start_time"],
        end_time=kwargs["end_time"],
        user_id=kwargs.get("user"),
        level="DEFAULT",
        status_message=None,
    )
    return captured


class TestLangfuseLitellmMetadata:
    """Regression: Langfuse must pick up proxy-auth fields from litellm_metadata."""

    def test_v1_messages_surfaces_user_api_key_alias_in_generation_name(self):
        kwargs = _build_kwargs(
            litellm_metadata=PROXY_AUTH_META,
            metadata={},
            call_type="anthropic_messages",
        )
        captured = _run_logger(kwargs, _build_response())
        assert captured["generation"] is not None
        assert captured["generation"]["name"] == "litellm:engineering-prod-key"

    def test_v1_messages_includes_user_api_key_fields_in_generation_metadata(self):
        kwargs = _build_kwargs(
            litellm_metadata=PROXY_AUTH_META,
            metadata={},
            call_type="anthropic_messages",
        )
        captured = _run_logger(kwargs, _build_response())
        gen_md = captured["generation"]["metadata"]
        assert gen_md["user_api_key_alias"] == "engineering-prod-key"
        assert gen_md["user_api_key_user_id"] == "user@example.com"
        assert gen_md["user_api_key_team_id"] == "team-eng"
        assert gen_md["user_api_key_team_alias"] == "engineering"

    def test_chat_completions_metadata_still_works(self):
        legacy_meta = {
            "user_api_key_alias": "my-chat-key",
            "user_api_key_user_id": "alice",
            "user_api_key_team_id": "t",
        }
        kwargs = _build_kwargs(
            litellm_metadata={},
            metadata=legacy_meta,
            call_type="completion",
            model="gpt-4o",
            provider="openai",
        )
        captured = _run_logger(kwargs, _build_response())
        assert captured["generation"]["name"] == "litellm:my-chat-key"
        gen_md = captured["generation"]["metadata"]
        assert gen_md["user_api_key_alias"] == "my-chat-key"
        assert gen_md["user_api_key_user_id"] == "alice"

    def test_both_present_picks_litellm_metadata(self):
        kwargs = _build_kwargs(
            litellm_metadata=PROXY_AUTH_META,
            metadata={"requester_metadata": {"client": "claude-code"}},
            call_type="anthropic_messages",
        )
        captured = _run_logger(kwargs, _build_response())
        assert captured["generation"]["name"] == "litellm:engineering-prod-key"

    def test_no_metadata_at_all_still_renders_safe_default(self):
        kwargs = _build_kwargs(
            litellm_metadata={},
            metadata={},
            call_type="anthropic_messages",
        )
        captured = _run_logger(kwargs, _build_response())
        assert captured["generation"]["name"] == "litellm-anthropic_messages"

    def test_caller_owned_litellm_metadata_not_mutated_empty_metadata_branch(self):
        """When metadata is empty, the helper returns litellm_metadata directly —
        the callback must not mutate it."""
        kwargs = _build_kwargs(
            litellm_metadata=dict(PROXY_AUTH_META),
            metadata={},
            call_type="anthropic_messages",
        )
        captured_lm = kwargs["litellm_params"]["litellm_metadata"]
        before_keys = set(captured_lm.keys())
        before_values = dict(captured_lm)
        _run_logger(kwargs, _build_response())
        assert set(captured_lm.keys()) == before_keys
        for k, v in before_values.items():
            assert captured_lm.get(k) == v

    def test_caller_owned_litellm_metadata_not_mutated_merge_branch(self):
        """When BOTH metadata and litellm_metadata are populated,
        ``get_litellm_metadata_from_kwargs`` invokes
        ``add_missing_spend_metadata_to_litellm_metadata``, which mutates
        its first argument in place. The Langfuse callback must guard
        against this so the caller-owned dict is unchanged.

        Setup: put proxy-auth fields in ``litellm_metadata`` and a *second*
        proxy-auth field only in ``metadata`` — without the guard,
        ``litellm_metadata`` would silently gain ``user_api_key_metadata``
        after the callback runs.
        """
        existing_litellm_md = dict(PROXY_AUTH_META)
        extra_in_metadata = {
            # this matches the "user_api_key" substring used by
            # add_missing_spend_metadata_to_litellm_metadata as a merge filter
            "user_api_key_metadata": {"src": "request-headers"},
            # non-matching key — should never leak into litellm_metadata
            "requester_metadata": {"client": "claude-code"},
        }
        kwargs = _build_kwargs(
            litellm_metadata=existing_litellm_md,
            metadata=extra_in_metadata,
            call_type="anthropic_messages",
        )
        captured_lm = kwargs["litellm_params"]["litellm_metadata"]
        captured_md = kwargs["litellm_params"]["metadata"]
        before_lm_keys = set(captured_lm.keys())
        before_lm_values = dict(captured_lm)
        before_md_keys = set(captured_md.keys())
        before_md_values = dict(captured_md)
        _run_logger(kwargs, _build_response())
        # litellm_metadata is the dict the merge helper mutates — assert no
        # extra keys leaked in.
        assert set(captured_lm.keys()) == before_lm_keys, (
            "Langfuse callback mutated litellm_params['litellm_metadata'] "
            f"(added keys: {set(captured_lm.keys()) - before_lm_keys})"
        )
        for k, v in before_lm_values.items():
            assert captured_lm.get(k) == v
        # metadata is read-only in the helper but assert it too for symmetry.
        assert set(captured_md.keys()) == before_md_keys
        for k, v in before_md_values.items():
            assert captured_md.get(k) == v
