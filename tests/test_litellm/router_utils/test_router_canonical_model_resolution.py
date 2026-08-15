"""Tests for same-provider canonical model-name resolution.

Invariants under test (see litellm/router_utils/canonical_model_resolution.py):
- I1: resolution never re-points a request that any existing route accepts
  (exact name, alias, wildcard/pattern, default_deployment, team routes).
- I2: identity only -- same provider, cost-map-attested; never a family/version
  hop, never cross-provider (Bedrock/Vertex/first-party are distinct).
- I3: auth is AND-on-target -- the resolved group must itself be allowed.
- Ambiguity fails closed.
"""

import pytest

from litellm import Router
from litellm.proxy.auth.auth_checks import _can_object_call_model
from litellm.proxy._types import ProxyException
from litellm.router_utils.canonical_model_resolution import (
    build_canonical_index,
    canonicalize,
    lookup,
)

ANTHROPIC_GROUP = "anthropic/claude-haiku-4-5"
DATED = "claude-haiku-4-5-20251001"
UNDATED = "claude-haiku-4-5"
BEDROCK_FORM = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture
def anthropic_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": ANTHROPIC_GROUP,
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-test"},
            }
        ]
    )


class TestCanonicalize:
    def test_strips_provider_route_prefix(self):
        assert canonicalize("anthropic/claude-haiku-4-5") == ("anthropic", UNDATED)

    def test_bare_dated_name_infers_provider(self):
        assert canonicalize(DATED) == ("anthropic", DATED)

    def test_bedrock_form_is_bedrock_not_anthropic(self):
        identity = canonicalize(BEDROCK_FORM)
        assert identity is not None
        assert identity[0] == "bedrock"

    def test_unknown_model_returns_none(self):
        assert canonicalize("totally-made-up-model-xyz") is None

    def test_empty_returns_none(self):
        assert canonicalize("") is None


class TestBuildIndexAndLookup:
    def test_dated_and_undated_spellings_resolve(self):
        index = build_canonical_index(
            [
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                }
            ]
        )
        assert lookup(index, DATED) == ANTHROPIC_GROUP
        assert lookup(index, UNDATED) == ANTHROPIC_GROUP

    def test_no_family_version_hop(self):
        """claude-sonnet-5 must never resolve to a haiku group (different model)."""
        index = build_canonical_index(
            [
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                }
            ]
        )
        assert lookup(index, "claude-sonnet-5") is None

    def test_cross_provider_never_matches(self):
        """An Anthropic-form request must not land on a Bedrock-only deployment,
        and a Bedrock-form request must not land on a first-party deployment."""
        bedrock_only = build_canonical_index(
            [
                {
                    "model_name": "claude-haiku-bedrock",
                    "litellm_params": {"model": f"bedrock/{BEDROCK_FORM}"},
                }
            ]
        )
        assert lookup(bedrock_only, DATED) is None
        assert lookup(bedrock_only, UNDATED) is None
        # Same-provider spelling still works for the Bedrock group.
        assert lookup(bedrock_only, BEDROCK_FORM) == "claude-haiku-bedrock"

        anthropic_only = build_canonical_index(
            [
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                }
            ]
        )
        assert lookup(anthropic_only, BEDROCK_FORM) is None

    def test_vertex_deployment_does_not_capture_anthropic_request(self):
        index = build_canonical_index(
            [
                {
                    "model_name": "claude-haiku-vertex",
                    "litellm_params": {"model": "vertex_ai/claude-haiku-4-5"},
                }
            ]
        )
        assert lookup(index, DATED) is None

    def test_ambiguous_identity_fails_closed(self):
        """Two same-provider groups serving one model: resolution must decline."""
        index = build_canonical_index(
            [
                {
                    "model_name": "haiku-prod",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                },
                {
                    "model_name": "haiku-experiments",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                },
            ]
        )
        assert lookup(index, DATED) is None
        assert lookup(index, UNDATED) is None

    def test_request_for_own_group_name_is_not_a_rewrite(self):
        index = build_canonical_index(
            [
                {
                    "model_name": UNDATED,  # group named exactly the canonical name
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                }
            ]
        )
        assert lookup(index, UNDATED) is None

    def test_malformed_deployments_are_skipped_not_fatal(self):
        index = build_canonical_index(
            [
                {"model_name": None, "litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
                {"model_name": "ok-group", "litellm_params": None},
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                },
            ]
        )
        assert lookup(index, DATED) == ANTHROPIC_GROUP


class TestRouterResolveCanonicalModelName:
    def test_claude_code_case(self, anthropic_router: Router):
        assert anthropic_router.resolve_canonical_model_name(DATED) == ANTHROPIC_GROUP

    def test_recognized_model_short_circuits(self, anthropic_router: Router):
        """I1: a name the router already serves is never rewritten."""
        assert anthropic_router.resolve_canonical_model_name(ANTHROPIC_GROUP) is None

    def test_strict_mode_disables_resolution(self):
        from litellm.types.router import RouterGeneralSettings

        router = Router(
            model_list=[
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-test"},
                }
            ],
            router_general_settings=RouterGeneralSettings(model_name_resolution="strict"),
        )
        assert router.resolve_canonical_model_name(DATED) is None

    def test_wildcard_route_outranks_resolution(self):
        """I1: an operator catch-all keeps winning; resolution declines entirely."""
        router = Router(
            model_list=[
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-test"},
                },
                {
                    "model_name": "openai/*",
                    "litellm_params": {"model": "openai/*", "api_key": "sk-test"},
                },
            ]
        )
        assert router.resolve_canonical_model_name(DATED) is None

    def test_default_deployment_outranks_resolution(self):
        router = Router(
            model_list=[
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-test"},
                },
                {
                    "model_name": "*",
                    "litellm_params": {"model": "*", "api_key": "sk-test"},
                },
            ]
        )
        assert router.resolve_canonical_model_name(DATED) is None

    def test_manual_alias_outranks_resolution(self):
        """A model_group_alias for the same spelling wins (is_recognized_model)."""
        router = Router(
            model_list=[
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-test"},
                },
                {
                    "model_name": "other-group",
                    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "sk-test"},
                },
            ],
            model_group_alias={DATED: "other-group"},
        )
        assert router.resolve_canonical_model_name(DATED) is None

    def test_get_canonical_model_index_builds_and_caches(self, anthropic_router: Router):
        """The index is built on demand, memoized, and keyed by identity."""
        index = anthropic_router._get_canonical_model_index()
        assert index[("anthropic", UNDATED)] == ANTHROPIC_GROUP
        # Second call returns the same memoized object (no rebuild).
        assert anthropic_router._get_canonical_model_index() is index

    def test_get_canonical_model_index_survives_build_failure(
        self, anthropic_router: Router, monkeypatch: pytest.MonkeyPatch
    ):
        """A failing index build degrades to 'strict', never raises."""
        import litellm.router as router_module

        def boom(*_args: object, **_kwargs: object) -> dict:
            raise RuntimeError("cost map exploded")

        monkeypatch.setattr(router_module, "build_canonical_index", boom)
        anthropic_router._canonical_model_index = None
        assert anthropic_router._get_canonical_model_index() == {}
        assert anthropic_router.resolve_canonical_model_name(DATED) is None

    def test_index_rebuilds_after_model_list_change(self, anthropic_router: Router):
        assert anthropic_router.resolve_canonical_model_name(DATED) == ANTHROPIC_GROUP
        anthropic_router.set_model_list(
            [
                {
                    "model_name": "gpt-group",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                }
            ]
        )
        assert anthropic_router.resolve_canonical_model_name(DATED) is None


class TestAuthAndOnTarget:
    """I3: an auto-resolved request is allowed iff the TARGET is allowed."""

    def test_target_allowed_grants_requested_spelling(self, anthropic_router: Router):
        assert (
            _can_object_call_model(
                model=DATED,
                llm_router=anthropic_router,
                models=[ANTHROPIC_GROUP],
                object_type="key",
            )
            is True
        )

    def test_target_not_allowed_denies(self, anthropic_router: Router):
        with pytest.raises(ProxyException):
            _can_object_call_model(
                model=DATED,
                llm_router=anthropic_router,
                models=["some-other-model"],
                object_type="key",
            )

    def test_unrelated_model_still_denied(self, anthropic_router: Router):
        with pytest.raises(ProxyException):
            _can_object_call_model(
                model="claude-sonnet-5",
                llm_router=anthropic_router,
                models=[ANTHROPIC_GROUP],
                object_type="key",
            )

    def test_unrestricted_key_unchanged(self, anthropic_router: Router):
        # Empty allowlist = unrestricted; behavior must not change.
        assert (
            _can_object_call_model(
                model=DATED,
                llm_router=anthropic_router,
                models=[],
                object_type="key",
            )
            is True
        )
