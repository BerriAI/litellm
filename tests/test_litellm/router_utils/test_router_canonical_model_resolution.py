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

    def test_ambiguity_guard_not_defeated_by_group_named_like_sentinel(self):
        """Regression: the ambiguity check used to compare against the string
        '__absent__' as an "is this key missing" sentinel. A model group
        literally named '__absent__' collided with that sentinel and could
        silently overwrite a same-identity entry instead of triggering the
        ambiguity decline."""
        index = build_canonical_index(
            [
                {
                    "model_name": "__absent__",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                },
                {
                    "model_name": "haiku-2",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                },
            ]
        )
        assert lookup(index, DATED) is None
        assert lookup(index, UNDATED) is None

    def test_custom_llm_provider_override_decides_provider(self):
        """Regression (I2): a deployment's explicit custom_llm_provider wins over
        whatever the model string implies. A first-party-looking id served via
        OpenRouter/Bedrock/Vertex must not be indexed as Anthropic -- otherwise
        an Anthropic-form request rides the rewrite onto that other provider's
        credentials, quota, and bill, which is exactly the cross-provider hop
        rule 2 forbids."""
        index = build_canonical_index(
            [
                {
                    "model_name": "haiku-via-openrouter",
                    "litellm_params": {
                        "model": "claude-haiku-4-5",
                        "custom_llm_provider": "openrouter",
                    },
                },
            ]
        )
        # Indexed under the real provider, not the one the string implies.
        assert ("openrouter", UNDATED) in index
        assert ("anthropic", UNDATED) not in index
        # An Anthropic-form request must not reach the OpenRouter deployment.
        assert lookup(index, DATED) is None
        assert lookup(index, UNDATED) is None

    def test_custom_llm_provider_override_still_resolves_within_provider(self):
        """The override narrows the provider, it does not disable resolution:
        a request that infers to the same overridden provider still resolves."""
        index = build_canonical_index(
            [
                {
                    "model_name": "haiku-via-bedrock",
                    "litellm_params": {
                        "model": "claude-haiku-4-5",
                        "custom_llm_provider": "bedrock",
                    },
                },
            ]
        )
        assert index[("bedrock", UNDATED)] == "haiku-via-bedrock"

    def test_team_owned_deployment_never_indexed(self):
        """Regression: a team-owned deployment (model_info.team_id set) must
        never enter the global canonical index. Without this, a no-team key
        could request an unclaimed spelling of a model whose only server is a
        team's private deployment and land on that team's credentials/quota --
        a team boundary is an access/billing boundary exactly like the
        cross-provider boundary and must not be crossed by inference."""
        index = build_canonical_index(
            [
                {
                    "model_name": "internal-team-model-xyz",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                    "model_info": {
                        "team_id": "team-A",
                        "team_public_model_name": "claude-haiku-4-5",
                    },
                },
            ]
        )
        assert lookup(index, DATED) is None
        assert lookup(index, UNDATED) is None

    def test_team_owned_deployment_does_not_block_global_sibling(self):
        """A team-owned deployment coexisting with a global deployment of the
        same identity must not suppress resolution to the global one."""
        index = build_canonical_index(
            [
                {
                    "model_name": "internal-team-model-xyz",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                    "model_info": {
                        "team_id": "team-A",
                        "team_public_model_name": "claude-haiku-4-5",
                    },
                },
                {
                    "model_name": ANTHROPIC_GROUP,
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5"},
                },
            ]
        )
        assert lookup(index, DATED) == ANTHROPIC_GROUP

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

    def test_no_team_caller_never_resolves_onto_team_owned_deployment(self):
        """Regression: end-to-end version of the team-leak fix. A no-team
        caller asking for an unclaimed spelling must not resolve onto a
        deployment that is the sole server of that identity but is owned by a
        team -- that would leak the team's credentials/quota to a global key."""
        router = Router(
            model_list=[
                {
                    "model_name": "internal-team-model-xyz",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "team-secret"},
                    "model_info": {"team_id": "team-A", "team_public_model_name": "claude-haiku-4-5"},
                },
            ]
        )
        assert router.resolve_canonical_model_name(DATED, request_team_id=None) is None
        # Even the requesting team's own id must not resolve through this path --
        # team-scoped models are reached via the existing team-route machinery,
        # not via canonical inference.
        assert router.resolve_canonical_model_name(DATED, request_team_id="team-A") is None

    def test_log_dedup_keyed_on_pair_not_target_alone(self, anthropic_router: Router, caplog: pytest.LogCaptureFixture):
        """Regression: a second distinct requested spelling resolving to an
        already-logged target must still get its own log line -- the
        (requested, target) cardinality is the signal used to size demand for
        follow-up resolution rules, so deduping on target alone would
        undercount it."""
        import logging

        with caplog.at_level(logging.INFO, logger="LiteLLM Router"):
            assert anthropic_router.resolve_canonical_model_name(DATED) == ANTHROPIC_GROUP
            assert anthropic_router.resolve_canonical_model_name(UNDATED) == ANTHROPIC_GROUP
            # Re-requesting the same spelling must not double-log.
            assert anthropic_router.resolve_canonical_model_name(DATED) == ANTHROPIC_GROUP
        messages = [r.message for r in caplog.records if "canonical-resolution" in r.message]
        assert any(DATED in m for m in messages)
        assert any(UNDATED in m for m in messages)
        assert sum(1 for m in messages if DATED in m) == 1

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


class TestCanonicalTargetReAuth:
    """The rewrite's re-auth must use the *resolved-model* helper.

    Regression: the hook originally called ``can_key_call_model``, which checks
    only the key's own allowlist. Team, team-member, and project allowlists were
    therefore never re-checked against the resolved target, so an unrestricted
    key on a team whose allowlist held only a stale unserved name could ride the
    rewrite onto a deployment that team was never granted.
    ``can_key_call_resolved_model`` is the helper every other post-resolution
    auth site uses (model_group_alias rewrites, realtime endpoints, auto-router).
    """

    @pytest.mark.asyncio
    async def test_uses_resolved_model_helper_so_team_scope_is_rechecked(self, anthropic_router: Router):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.route_llm_request import _canonical_target_is_allowed

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_resolved_model",
            new=AsyncMock(return_value=None),
        ) as resolved_check:
            allowed = await _canonical_target_is_allowed(
                canonical_target=ANTHROPIC_GROUP,
                llm_router=anthropic_router,
                user_api_key_dict=UserAPIKeyAuth(token="t", team_id="team-A"),
            )

        assert allowed is True
        resolved_check.assert_awaited_once()
        assert resolved_check.await_args.kwargs["model"] == ANTHROPIC_GROUP

    @pytest.mark.asyncio
    async def test_denial_declines_rewrite_rather_than_raising(self, anthropic_router: Router):
        """A denial must return False (request falls through to the usual 400),
        never propagate an exception that would leak the target's existence."""
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth

        from litellm.proxy.route_llm_request import _canonical_target_is_allowed

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_resolved_model",
            new=AsyncMock(side_effect=ProxyException(message="denied", type="auth_error", param=None, code=401)),
        ):
            allowed = await _canonical_target_is_allowed(
                canonical_target=ANTHROPIC_GROUP,
                llm_router=anthropic_router,
                user_api_key_dict=UserAPIKeyAuth(token="t", team_id="team-A"),
            )

        assert allowed is False

    @pytest.mark.asyncio
    async def test_absent_auth_context_fails_closed(self, anthropic_router: Router):
        """Regression: absent key context must DECLINE the rewrite, not allow it.

        Most route_request callers (image generation, rerank, moderation,
        speech, transcription, realtime, Responses WebSocket) are authenticated
        but don't currently forward user_api_key_dict. Returning True here
        would run the rewrite with no target authorization at all on exactly
        those paths -- a key whose allowlist holds only the stale requested
        spelling could reach a target it was never granted. Declining costs
        those endpoints only the convenience rewrite; the AND-on-target
        guarantee stays unconditional."""
        from litellm.proxy.route_llm_request import _canonical_target_is_allowed

        assert (
            await _canonical_target_is_allowed(
                canonical_target=ANTHROPIC_GROUP,
                llm_router=anthropic_router,
                user_api_key_dict=None,
            )
            is False
        )


class TestProxyRouterGeneralSettings:
    """Operators must be able to set model_name_resolution on the proxy.

    Regression: the proxy passed a hardcoded RouterGeneralSettings(async_only_mode=True)
    as an explicit keyword alongside **router_params, so setting
    router_general_settings in config raised "got multiple values for keyword
    argument" at startup -- leaving no way to select 'strict'.
    """

    def test_config_settings_preserved_and_async_only_forced(self):
        from litellm.proxy.proxy_server import _proxy_router_general_settings

        settings = _proxy_router_general_settings({"model_name_resolution": "strict"})
        assert settings.model_name_resolution == "strict"
        assert settings.async_only_mode is True

    def test_async_only_mode_cannot_be_disabled_by_config(self):
        from litellm.proxy.proxy_server import _proxy_router_general_settings

        settings = _proxy_router_general_settings({"async_only_mode": False, "model_name_resolution": "strict"})
        assert settings.async_only_mode is True
        assert settings.model_name_resolution == "strict"

    def test_none_yields_proxy_default(self):
        from litellm.proxy.proxy_server import _proxy_router_general_settings

        settings = _proxy_router_general_settings(None)
        assert settings.async_only_mode is True
        assert settings.model_name_resolution == "canonical"

    def test_model_instance_is_not_mutated(self):
        from litellm.proxy.proxy_server import _proxy_router_general_settings
        from litellm.types.router import RouterGeneralSettings

        original = RouterGeneralSettings(async_only_mode=False, model_name_resolution="strict")
        settings = _proxy_router_general_settings(original)
        assert settings.async_only_mode is True
        # The caller's object must not be rewritten at a distance.
        assert original.async_only_mode is False
