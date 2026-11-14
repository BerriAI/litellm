import pytest

import litellm
from litellm.proxy._types import ModelAccessDeniedProxyException
from litellm.proxy.auth.auth_checks import can_object_call_model
from litellm.proxy.auth.user_api_key_auth import _get_model_names_for_budget_checks
from litellm.proxy.litellm_pre_call_utils import resolve_wildcard_prefixed_model


class _DummyRouter:
    def __init__(self, model_list):
        self.model_list = model_list
        self.model_group_alias = {}
        self.model_name_to_deployment_indices = {}

    def get_model_access_groups(self, model_name: str, team_id: str | None = None):
        return []


@pytest.fixture
def sample_router():
    return _DummyRouter(
        model_list=[
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {"model_name": "google/*", "litellm_params": {"model": "vertex_ai/*"}},
            {"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}},
            {
                "model_name": "codex-auto-review",
                "litellm_params": {"model": "openai/codex-auto-review"},
            },
            {
                "model_name": "DeepSeek-V4-Flash",
                "litellm_params": {
                    "model": "DeepSeek-V4-Flash",
                    "custom_llm_provider": "hosted_vllm",
                },
            },
        ]
    )


def test_prefix_enforcement_disabled_by_default(sample_router, monkeypatch):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", False)
    assert resolve_wildcard_prefixed_model("gpt-6.1-sol", sample_router) == "gpt-6.1-sol"
    assert resolve_wildcard_prefixed_model("vertex_ai/gemini-2.5-pro", sample_router) == "vertex_ai/gemini-2.5-pro"
    assert resolve_wildcard_prefixed_model("openai/gemini-3.8-flash", sample_router) == "openai/gemini-3.8-flash"


@pytest.mark.parametrize(
    "raw_model,expected_model",
    [
        ("gpt-6.1-sol", "openai/gpt-6.1-sol"),
        ("openai/gpt-6.1-sol", "openai/gpt-6.1-sol"),
        ("gemini-2.5-pro", "google/gemini-2.5-pro"),
        ("vertex_ai/gemini-2.5-pro", "google/gemini-2.5-pro"),
        ("google/gemini-2.5-pro", "google/gemini-2.5-pro"),
        ("gemini-3.8-flash", "google/gemini-3.8-flash"),
        ("vertex_ai/gemini-3.8-flash", "google/gemini-3.8-flash"),
        ("google/gemini-3.8-flash", "google/gemini-3.8-flash"),
        ("claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"),
        ("anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"),
        ("codex-auto-review", "codex-auto-review"),
        ("openai/codex-auto-review", "codex-auto-review"),
        ("DeepSeek-V4-Flash", "DeepSeek-V4-Flash"),
        ("hosted_vllm/DeepSeek-V4-Flash", "DeepSeek-V4-Flash"),
    ],
)
def test_prefix_enforcement_normalizes_valid_inputs(sample_router, monkeypatch, raw_model, expected_model):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    assert resolve_wildcard_prefixed_model(raw_model, sample_router) == expected_model


@pytest.mark.parametrize(
    "invalid_model",
    [
        "openai/gemini-3.8-flash",
        "vertex/gemini-3.8-flash",
        "vertext/gemini-3.8-flash",
        "test/gemini-3.8-flash",
        "google/gpt-6.1-sol",
        "openai/DeepSeek-V4-Flash",
        "google/codex-auto-review",
        "openai/openai/gpt-6.1-sol",
        "openai/chatgpt/gpt-4o",
    ],
)
def test_prefix_enforcement_rejects_mismatched_or_unknown_prefixes(sample_router, monkeypatch, invalid_model):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    assert resolve_wildcard_prefixed_model(invalid_model, sample_router) is None
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        can_object_call_model(
            model=invalid_model,
            llm_router=sample_router,
            models=["openai/*", "google/*", "anthropic/*", "codex-auto-review", "DeepSeek-V4-Flash"],
            object_type="team",
        )
    assert str(exc_info.value.code) == "403"


def test_can_object_call_model_allows_normalized_wildcard_access(sample_router, monkeypatch):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    for valid_google_call in ("gemini-2.5-pro", "vertex_ai/gemini-2.5-pro", "google/gemini-2.5-pro"):
        assert (
            can_object_call_model(
                model=valid_google_call,
                llm_router=sample_router,
                models=["google/*"],
                object_type="team",
            )
            is True
        )


def test_key_alias_chained_after_prefix_normalization_is_enforced(monkeypatch):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    router = _DummyRouter(
        model_list=[
            {"model_name": "allowed-model", "litellm_params": {"model": "openai/gpt-4o"}},
        ]
    )
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        can_object_call_model(
            model="openai/gpt-4o",
            llm_router=router,
            models=["allowed-model"],
            object_type="key",
            key_model_aliases={"allowed-model": "private-model"},
        )
    assert str(exc_info.value.code) == "403"
    assert "openai/gpt-4o" in exc_info.value.message
    assert "private-model" not in exc_info.value.message


def test_team_scoped_deployments_are_skipped_and_never_leaked(monkeypatch):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    router = _DummyRouter(
        model_list=[
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {
                "model_name": "model_name_team-1_secret-uuid",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"team_id": "team-1"},
            },
        ]
    )
    assert resolve_wildcard_prefixed_model("openai/gpt-4o", router) == "openai/gpt-4o"
    with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
        can_object_call_model(
            model="openai/gpt-4o",
            llm_router=router,
            models=["google/*"],
            object_type="team",
        )
    assert "model_name_team-1_secret-uuid" not in exc_info.value.message
    assert "openai/gpt-4o" in exc_info.value.message


def test_budget_check_model_names_include_normalized_model(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    router = _DummyRouter(
        model_list=[
            {"model_name": "allowed-model", "litellm_params": {"model": "openai/gpt-4o"}},
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    assert _get_model_names_for_budget_checks("openai/gpt-4o") == ["openai/gpt-4o", "allowed-model"]


def test_oversized_model_names_are_not_cached(sample_router, monkeypatch):
    from litellm.proxy import litellm_pre_call_utils

    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    oversized = "openai/gpt-4o-" + ("a" * 2000)
    resolved = resolve_wildcard_prefixed_model(oversized, sample_router)
    assert resolved == oversized
    assert oversized not in litellm_pre_call_utils._ROUTER_RESOLVED_CACHE


def test_custom_named_deployment_does_not_hijack_wildcard_model(monkeypatch):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    router = _DummyRouter(
        model_list=[
            {"model_name": "hr-assistant", "litellm_params": {"model": "openai/gpt-4o"}},
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {"model_name": "codex-auto-review", "litellm_params": {"model": "openai/codex-auto-review"}},
        ]
    )
    assert resolve_wildcard_prefixed_model("openai/gpt-4o", router) == "openai/gpt-4o"
    assert resolve_wildcard_prefixed_model("gpt-4o", router) == "openai/gpt-4o"
    assert resolve_wildcard_prefixed_model("hr-assistant", router) == "hr-assistant"
    assert resolve_wildcard_prefixed_model("openai/codex-auto-review", router) == "codex-auto-review"


def test_in_place_model_list_mutation_and_unprefixed_custom_deployment(monkeypatch):
    monkeypatch.setattr(litellm, "enable_model_prefix_enforcement", True)
    router = _DummyRouter(
        model_list=[
            {"model_name": "hr-bot", "litellm_params": {"model": "gpt-4o"}},
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
        ]
    )
    assert resolve_wildcard_prefixed_model("gpt-4o", router) == "openai/gpt-4o"
    assert resolve_wildcard_prefixed_model("vertex_ai/gemini-2.5-pro", router) is None

    # Mutate model_list in-place (simulating /model/new)
    router.model_list.append(
        {"model_name": "google/*", "litellm_params": {"model": "vertex_ai/*"}}
    )
    assert resolve_wildcard_prefixed_model("vertex_ai/gemini-2.5-pro", router) == "google/gemini-2.5-pro"
