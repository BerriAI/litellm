import logging
from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import litellm.proxy.policy_engine.attachment_registry as attachment_registry_module
import litellm.proxy.policy_engine.policy_registry as policy_registry_module
from litellm.proxy.policy_engine.attachment_registry import AttachmentRegistry
from litellm.proxy.policy_engine.policy_registry import PolicyRegistry
from litellm.rust_bridge.policy_engine import PolicyEngine, build_policy_engine, current_policy_engine
from litellm.types.proxy.policy_engine import (
    GuardrailPipeline,
    PipelineStep,
    Policy,
    PolicyAttachment,
    PolicyCondition,
    PolicyGuardrails,
    PolicyMatchContext,
)


def _context(
    model: str | None = "gpt-4", team: str | None = "t", key: str | None = "k", tags: Sequence[str] | None = None
) -> PolicyMatchContext:
    return PolicyMatchContext(
        team_alias=team, key_alias=key, model=model, tags=list(tags) if tags is not None else None
    )


def _condition_applies(condition: PolicyCondition | None, model: str | None) -> bool:
    policy: Final = Policy(guardrails=PolicyGuardrails(add=["g"]), condition=condition)
    return build_policy_engine({"p": policy}).policy_guardrails("p", _context(model)).guardrails == ["g"]


class TestModelCondition:
    def test_no_condition_always_matches(self) -> None:
        assert _condition_applies(None, "gpt-4") is True

    def test_empty_condition_always_matches(self) -> None:
        assert _condition_applies(PolicyCondition(), "any-model") is True

    @pytest.mark.parametrize(("model", "expected"), [("gpt-4", True), ("gpt-3.5", False)])
    def test_exact_model(self, model: str, expected: bool) -> None:
        assert _condition_applies(PolicyCondition(model="gpt-4"), model) is expected

    @pytest.mark.parametrize(
        ("model", "expected"), [("gpt-4", True), ("gpt-4-turbo", True), ("gpt-4o", True), ("gpt-3.5", False)]
    )
    def test_regex_pattern(self, model: str, expected: bool) -> None:
        assert _condition_applies(PolicyCondition(model="gpt-4.*"), model) is expected

    @pytest.mark.parametrize(("model", "expected"), [("gpt-4", True), ("claude-3", True), ("gpt-3.5", False)])
    def test_list_of_models(self, model: str, expected: bool) -> None:
        assert _condition_applies(PolicyCondition(model=["gpt-4", "gpt-4-turbo", "claude-3"]), model) is expected

    @pytest.mark.parametrize(("model", "expected"), [("gpt-4-turbo", True), ("claude-3", True), ("llama-2", False)])
    def test_list_of_regex_patterns(self, model: str, expected: bool) -> None:
        assert _condition_applies(PolicyCondition(model=["gpt-4.*", "claude-.*"]), model) is expected

    def test_none_model_does_not_match(self) -> None:
        assert _condition_applies(PolicyCondition(model="gpt-4"), None) is False

    @pytest.mark.parametrize(
        ("pattern", "model", "expected"),
        [
            pytest.param("[[]", "[", True, id="syntax-only-python-reads"),
            pytest.param("[a&&b]", "&", True, id="class-python-reads-literally"),
            pytest.param("[a&&b]", "c", False, id="class-python-reads-literally-miss"),
            pytest.param("gpt-4(", "gpt-4(", True, id="invalid-regex-still-matches-literally"),
        ],
    )
    def test_patterns_follow_python_re(self, pattern: str, model: str, expected: bool) -> None:
        assert _condition_applies(PolicyCondition(model=pattern), model) is expected


def _attached(attachment: PolicyAttachment, context: PolicyMatchContext) -> bool:
    engine: Final = build_policy_engine({"p": Policy()}, attachments=(attachment,))
    return [match["policy_name"] for match in engine.attached_policies(context, only_applying=False)] == ["p"]


def _scoped(
    teams: Sequence[str] = ("*",),
    keys: Sequence[str] = ("*",),
    models: Sequence[str] = ("*",),
    tags: Sequence[str] | None = None,
) -> PolicyAttachment:
    return PolicyAttachment(
        policy="p", teams=list(teams), keys=list(keys), models=list(models), tags=list(tags) if tags else None
    )


class TestAttachmentScope:
    @pytest.mark.parametrize(
        ("value", "patterns", "expected"),
        [
            ("healthcare-team", ("healthcare-team",), True),
            ("finance-team", ("healthcare-team",), False),
            ("any-team", ("*",), True),
            ("dev-key-123", ("dev-key-*",), True),
            ("prod-key-123", ("dev-key-*",), False),
            (None, ("*",), True),
            (None, ("specific",), False),
        ],
    )
    def test_team_patterns(self, value: str | None, patterns: Sequence[str], expected: bool) -> None:
        assert _attached(_scoped(teams=patterns), _context(team=value)) is expected

    def test_all_fields_match(self) -> None:
        context: Final = _context(team="healthcare-team", key="any-key", model="gpt-4")
        assert _attached(_scoped(teams=["healthcare-team"], models=["gpt-4"]), context) is True

    def test_team_mismatch(self) -> None:
        context: Final = _context(team="finance-team", key="any-key", model="gpt-4")
        assert _attached(_scoped(teams=["healthcare-team"]), context) is False

    def test_wildcard_patterns(self) -> None:
        context: Final = _context(team="any-team", key="dev-key-123", model="bedrock/claude-3")
        assert _attached(_scoped(keys=["dev-key-*"], models=["bedrock/*"]), context) is True

    def test_global_wildcard(self) -> None:
        assert _attached(_scoped(), _context(team="any-team", key="any-key", model="any-model")) is True

    @pytest.mark.parametrize(
        ("scope_tags", "context_tags", "expected"),
        [
            (["healthcare"], ["healthcare", "internal"], True),
            (["health-*"], ["health-prod"], True),
            (["healthcare"], ["finance"], False),
            (["healthcare"], None, False),
            (None, ["healthcare", "internal"], True),
        ],
    )
    def test_tags(self, scope_tags: Sequence[str] | None, context_tags: Sequence[str] | None, expected: bool) -> None:
        assert _attached(_scoped(tags=scope_tags), _context(tags=context_tags)) is expected

    @pytest.mark.parametrize(
        ("team", "tags", "expected"),
        [("team-a", ["healthcare"], True), ("team-b", ["healthcare"], False), ("team-a", ["finance"], False)],
    )
    def test_tags_and_team_both_required(self, team: str, tags: Sequence[str], expected: bool) -> None:
        assert _attached(_scoped(teams=["team-a"], tags=["healthcare"]), _context(team=team, tags=tags)) is expected


class TestAttachmentRegistryMatching:
    def test_team_and_global_attachments_both_match(self) -> None:
        registry: Final = AttachmentRegistry()
        registry.load_attachments(
            [
                {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
                {"policy": "global-policy", "scope": "*"},
            ]
        )
        attached: Final = registry.get_attached_policies(_context(team="healthcare-team"))
        assert "healthcare-policy" in attached
        assert "global-policy" in attached

    def test_no_match(self) -> None:
        registry: Final = AttachmentRegistry()
        registry.load_attachments([{"policy": "healthcare-policy", "teams": ["healthcare-team"]}])
        assert "healthcare-policy" not in registry.get_attached_policies(_context(team="finance-team"))


class TestInheritance:
    def test_simple_policy(self) -> None:
        engine: Final = build_policy_engine(
            {"global": Policy(guardrails=PolicyGuardrails(add=["pii_blocker", "toxicity_filter"]))}
        )
        resolved: Final = engine.policy_guardrails("global")
        assert set(resolved.guardrails) == {"pii_blocker", "toxicity_filter"}
        assert resolved.inheritance_chain == ["global"]

    def test_child_inherits_and_adds(self) -> None:
        engine: Final = build_policy_engine(
            {
                "base": Policy(guardrails=PolicyGuardrails(add=["pii_blocker"])),
                "healthcare": Policy(inherit="base", guardrails=PolicyGuardrails(add=["hipaa_audit"])),
            }
        )
        resolved: Final = engine.policy_guardrails("healthcare")
        assert set(resolved.guardrails) == {"pii_blocker", "hipaa_audit"}
        assert resolved.inheritance_chain == ["base", "healthcare"]

    def test_child_removes(self) -> None:
        engine: Final = build_policy_engine(
            {
                "base": Policy(guardrails=PolicyGuardrails(add=["pii_blocker", "phi_blocker"])),
                "dev": Policy(
                    inherit="base", guardrails=PolicyGuardrails(add=["toxicity_filter"], remove=["phi_blocker"])
                ),
            }
        )
        assert set(engine.policy_guardrails("dev").guardrails) == {"pii_blocker", "toxicity_filter"}

    def test_deep_chain(self) -> None:
        engine: Final = build_policy_engine(
            {
                "root": Policy(guardrails=PolicyGuardrails(add=["root_guardrail"])),
                "middle": Policy(inherit="root", guardrails=PolicyGuardrails(add=["middle_guardrail"])),
                "leaf": Policy(inherit="middle", guardrails=PolicyGuardrails(add=["leaf_guardrail"])),
            }
        )
        resolved: Final = engine.policy_guardrails("leaf")
        assert set(resolved.guardrails) == {"root_guardrail", "middle_guardrail", "leaf_guardrail"}
        assert resolved.inheritance_chain == ["root", "middle", "leaf"]


class TestConditions:
    @pytest.mark.parametrize(("model", "expected"), [("gpt-4", ["toxicity_filter"]), ("gpt-3.5", [])])
    def test_own_condition(self, model: str, expected: Sequence[str]) -> None:
        engine: Final = build_policy_engine(
            {
                "gpt4-policy": Policy(
                    guardrails=PolicyGuardrails(add=["toxicity_filter"]), condition=PolicyCondition(model="gpt-4.*")
                )
            }
        )
        assert engine.policy_guardrails("gpt4-policy", _context(model)).guardrails == expected

    @pytest.mark.parametrize(
        ("model", "expected"), [("gpt-4", {"pii_blocker", "child_guardrail"}), ("gpt-3.5", {"pii_blocker"})]
    )
    def test_inherited_condition(self, model: str, expected: set[str]) -> None:
        engine: Final = build_policy_engine(
            {
                "base": Policy(guardrails=PolicyGuardrails(add=["pii_blocker"])),
                "child": Policy(
                    inherit="base",
                    guardrails=PolicyGuardrails(add=["child_guardrail"]),
                    condition=PolicyCondition(model="gpt-4"),
                ),
            }
        )
        assert set(engine.policy_guardrails("child", _context(model)).guardrails) == expected

    @pytest.mark.parametrize(("model", "expected"), [("gpt-5.5", {"y"}), ("claude-haiku", {"x", "y"})])
    def test_condition_missing_child_keeps_inherited_parent(self, model: str, expected: set[str]) -> None:
        engine: Final = build_policy_engine(
            {
                "parent": Policy(guardrails=PolicyGuardrails(add=["y"])),
                "child": Policy(
                    inherit="parent",
                    guardrails=PolicyGuardrails(add=["x"]),
                    condition=PolicyCondition(model="claude.*"),
                ),
            }
        )
        assert set(engine.resolve(_context(model), names=["child"]).guardrails) == expected

    def test_pipeline_skipped_when_own_condition_misses(self) -> None:
        pipeline: Final = GuardrailPipeline(mode="pre_call", steps=[PipelineStep(guardrail="child-guard")])
        engine: Final = build_policy_engine(
            {
                "parent": Policy(guardrails=PolicyGuardrails(add=["y"])),
                "child": Policy(inherit="parent", pipeline=pipeline, condition=PolicyCondition(model="gpt-5.5")),
            }
        )
        assert engine.resolve(_context("gpt-4o"), names=["child"]).pipelines == ()
        assert engine.resolve(_context("gpt-5.5"), names=["child"]).pipelines == (("child", pipeline),)

    def test_unknown_policy_name_applies_nothing(self) -> None:
        engine: Final = build_policy_engine(
            {"real": Policy(guardrails=PolicyGuardrails(add=["g"]), condition=PolicyCondition(model="claude.*"))}
        )
        assert engine.resolve(_context("gpt-5.5"), names=["nope"]).applied_policies == ()


def _install(
    monkeypatch: pytest.MonkeyPatch,
    policies_config: Mapping[str, Mapping[str, object]],
    attachments_config: Sequence[Mapping[str, object]],
) -> None:
    policies: Final = PolicyRegistry()
    policies.load_policies({name: dict(config) for name, config in policies_config.items()})
    attachments: Final = AttachmentRegistry()
    attachments.load_attachments([dict(attachment) for attachment in attachments_config])
    monkeypatch.setattr(policy_registry_module, "get_policy_registry", lambda: policies)
    monkeypatch.setattr(attachment_registry_module, "get_attachment_registry", lambda: attachments)


def _matching(context: PolicyMatchContext) -> list[str]:
    return [match["policy_name"] for match in current_policy_engine().attached_policies(context, only_applying=True)]


def _install_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            "guardrail-y": {"guardrails": {"add": ["y"]}},
            "guardrail-x": {"guardrails": {"add": ["x"]}, "condition": {"model": "claude.*"}},
        },
        [{"policy": "guardrail-x", "tags": ["opt-in"]}, {"policy": "guardrail-y", "scope": "*", "default": True}],
    )


def _install_inherited(monkeypatch: pytest.MonkeyPatch, parent_condition: Mapping[str, object] | None = None) -> None:
    _install(
        monkeypatch,
        {
            "parent": {"guardrails": {"add": ["y"]}, **({"condition": parent_condition} if parent_condition else {})},
            "child": {"inherit": "parent", "guardrails": {"add": ["x"]}, "condition": {"model": "claude.*"}},
            "fallback": {"guardrails": {"add": ["z"]}},
        },
        [{"policy": "child", "scope": "*"}, {"policy": "fallback", "scope": "*", "default": True}],
    )


class TestDefaultFallback:
    def test_condition_failing_opt_in_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_opt_in(monkeypatch)
        assert _matching(_context("gpt-5.5", tags=["opt-in"])) == ["guardrail-y"]

    def test_condition_passing_opt_in_suppresses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_opt_in(monkeypatch)
        assert _matching(_context("claude-haiku", tags=["opt-in"])) == ["guardrail-x"]

    def test_condition_missing_child_with_unconditional_parent_still_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_inherited(monkeypatch)
        assert _matching(_context("gpt-5.5")) == ["child"]

    def test_child_whose_whole_chain_misses_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_inherited(monkeypatch, parent_condition={"model": "claude.*"})
        assert _matching(_context("gpt-5.5")) == ["fallback"]


_MODELS: Final = ("gpt-4o", "gpt-5.5", "claude-opus-4-1")


@st.composite
def _forest_and_request(draw: st.DrawFn) -> tuple[Mapping[str, Policy], tuple[str, ...], PolicyMatchContext]:
    names: Final = tuple(f"p{i}" for i in range(draw(st.integers(min_value=1, max_value=6))))
    policies: Final = {
        name: Policy(
            inherit=draw(st.sampled_from((None, *names[:i]))),
            guardrails=PolicyGuardrails(add=[f"g-{name}"]),
            condition=draw(st.sampled_from((None, *(PolicyCondition(model=m) for m in _MODELS)))),
        )
        for i, name in enumerate(names)
    }
    attached: Final = tuple(draw(st.lists(st.sampled_from(sorted(policies)), unique=True)))
    return policies, attached, _context(draw(st.sampled_from(_MODELS)))


def _own_condition_applies(policy: Policy, context: PolicyMatchContext) -> bool:
    return policy.condition is None or policy.condition.model == context.model


def _applicable_chain(
    engine: PolicyEngine, policies: Mapping[str, Policy], name: str, context: PolicyMatchContext
) -> tuple[str, ...]:
    chain: Final = engine.policy_guardrails(name).inheritance_chain
    return tuple(member for member in chain if _own_condition_applies(policies[member], context))


class TestChainMatchingProperties:
    @given(_forest_and_request())
    @settings(max_examples=400, deadline=None)
    def test_chain_matching_only_widens_to_applicable_ancestor_guardrails(
        self, case: tuple[Mapping[str, Policy], tuple[str, ...], PolicyMatchContext]
    ) -> None:
        policies, attached, context = case
        engine: Final = build_policy_engine(policies)
        head: Final = engine.resolve(context, names=attached).applied_policies
        base: Final = tuple(name for name in attached if _own_condition_applies(policies[name], context))
        expected_head: Final = tuple(name for name in attached if _applicable_chain(engine, policies, name, context))

        assert head == expected_head, "a policy applies exactly when some chain member's own condition applies"
        assert frozenset(base) <= frozenset(head), "head must never drop a policy base applied"

        for name in head:
            resolved = engine.policy_guardrails(name, context)
            assert sorted(resolved.guardrails) == sorted(
                f"g-{member}" for member in _applicable_chain(engine, policies, name, context)
            )
            if name not in base:
                assert f"g-{name}" not in resolved.guardrails, "a condition-missed child must not add its own guardrail"


def _ancestor_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "applied through ancestor" in r.getMessage()]


class TestAncestorAdmissionLogging:
    @staticmethod
    def _attached_engine(child_condition: str, parent_condition: str | None = None) -> PolicyEngine:
        return build_policy_engine(
            {
                "parent": Policy(
                    guardrails=PolicyGuardrails(add=["g-parent"]),
                    condition=PolicyCondition(model=parent_condition) if parent_condition else None,
                ),
                "child": Policy(
                    inherit="parent",
                    guardrails=PolicyGuardrails(add=["g-child"]),
                    condition=PolicyCondition(model=child_condition),
                ),
            },
            attachments=(PolicyAttachment(policy="child", scope="*"),),
        )

    def test_logs_when_admitted_through_ancestor_only(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            matches: Final = self._attached_engine("gpt-5.5").attached_policies(_context("gpt-4o"), only_applying=True)
        logs: Final = _ancestor_logs(caplog)
        assert [match["policy_name"] for match in matches] == ["child"]
        assert len(logs) == 1
        assert "applied through ancestor 'parent'" in logs[0]
        assert "'child'" in logs[0]

    def test_no_log_when_own_condition_matches(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            matches: Final = self._attached_engine("gpt-5.5").attached_policies(_context("gpt-5.5"), only_applying=True)
        assert [match["policy_name"] for match in matches] == ["child"]
        assert not _ancestor_logs(caplog)

    def test_no_log_when_no_chain_member_applies(self, caplog: pytest.LogCaptureFixture) -> None:
        engine: Final = self._attached_engine("gpt-5.5", parent_condition="claude-opus-4-1")
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            matches: Final = engine.attached_policies(_context("gpt-4o"), only_applying=True)
        assert matches == ()
        assert not _ancestor_logs(caplog)

    def test_requested_names_log_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        engine: Final = build_policy_engine(
            {
                "parent": Policy(guardrails=PolicyGuardrails(add=["g-parent"])),
                "child": Policy(
                    inherit="parent",
                    guardrails=PolicyGuardrails(add=["g-child"]),
                    condition=PolicyCondition(model="gpt-5.5"),
                ),
            }
        )
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            resolution: Final = engine.resolve(_context("gpt-4o"), names=["child"])
        assert resolution.applied_policies == ("child",)
        assert not _ancestor_logs(caplog)
