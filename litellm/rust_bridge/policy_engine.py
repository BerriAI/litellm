"""The proxy policy engine, resolved in Rust over a snapshot of the policy and attachment registries.

The engine for the live registries is rebuilt whenever a registry replaces one of its containers.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.rust_bridge import _native
from litellm.types.proxy.policy_engine import (
    GuardrailPipeline,
    Policy,
    PolicyAttachment,
    PolicyMatchContext,
    ResolvedPolicy,
)

if TYPE_CHECKING:
    from litellm.proxy.policy_engine.attachment_registry import PolicyAttachmentMatch

_PatternRow: TypeAlias = tuple[str, re.Pattern[str] | None]
_Names: TypeAlias = Sequence[str]
_PolicyRow: TypeAlias = tuple[str | None, _Names, _Names, Sequence[_PatternRow] | None, _Names | None]
_AttachmentRow: TypeAlias = tuple[str, bool, _Names, _Names, _Names, _Names, int | None, bool]
_ContextRow: TypeAlias = tuple[str | None, str | None, str | None, _Names]

_NO_POLICIES: Final[Mapping[str, Policy]] = MappingProxyType({})
_NO_VERSIONS: Final[Mapping[str, tuple[str, Policy]]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class PolicyEngineResolution:
    applied_policies: tuple[str, ...]
    policy_sources: Mapping[str, str]
    guardrails: tuple[str, ...]
    pipelines: tuple[tuple[str, GuardrailPipeline], ...]
    pipeline_managed_guardrails: frozenset[str]


def _compiled(pattern: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _condition_row(policy: Policy) -> tuple[_PatternRow, ...] | None:
    model: Final = policy.condition.model if policy.condition is not None else None
    if model is None:
        return None
    patterns: Final = (model,) if isinstance(model, str) else model
    return tuple((pattern, _compiled(pattern)) for pattern in patterns)


def _policy_row(policy: Policy) -> _PolicyRow:
    return (
        policy.inherit or None,
        policy.guardrails.get_add(),
        policy.guardrails.get_remove(),
        _condition_row(policy),
        tuple(step.guardrail for step in policy.pipeline.steps) if policy.pipeline is not None else None,
    )


def _attachment_row(attachment: PolicyAttachment) -> _AttachmentRow:
    return (
        attachment.policy,
        attachment.is_global(),
        attachment.teams or [],
        attachment.keys or [],
        attachment.models or [],
        attachment.tags or [],
        attachment.priority,
        attachment.default,
    )


def _context_row(context: PolicyMatchContext) -> _ContextRow:
    return (context.team_alias, context.key_alias, context.model, tuple(context.tags or ()))


def _log_ancestor_admissions(admissions: Sequence[tuple[str, str]], context: PolicyMatchContext) -> None:
    for policy_name, ancestor in admissions:
        verbose_proxy_logger.info(
            "Policy '%s' applied through ancestor '%s' although its own condition did not match "
            "(team_alias=%s, key_alias=%s, model=%s)",
            policy_name,
            ancestor,
            context.team_alias,
            context.key_alias,
            context.model,
        )


def _pipeline(policy: Policy) -> GuardrailPipeline:
    pipeline: Final = policy.pipeline
    assert pipeline is not None, "Rust only reports pipelines of policies that have one"
    return pipeline


@dataclass(frozen=True, slots=True)
class PolicyEngine:
    policies: Mapping[str, Policy]
    versions: Mapping[str, tuple[str, Policy]]
    attachments: Sequence[PolicyAttachment]
    native: _native.PolicyEngine

    def resolve(
        self,
        context: PolicyMatchContext,
        names: Sequence[str] = (),
        version_ids: Sequence[str] = (),
    ) -> PolicyEngineResolution:
        """The policies a request applies, from its attachments plus the names and version ids it sent"""
        resolution: Final = self.native.resolve(_context_row(context), tuple(names), tuple(version_ids))
        _log_ancestor_admissions(resolution.ancestor_admissions, context)
        return PolicyEngineResolution(
            applied_policies=tuple(resolution.applied_policies),
            policy_sources=MappingProxyType(dict(resolution.policy_sources)),
            guardrails=tuple(resolution.guardrails),
            pipelines=tuple(
                (name, _pipeline(self.versions[version_id][1] if version_id is not None else self.policies[name]))
                for name, version_id in resolution.pipelines
            ),
            pipeline_managed_guardrails=frozenset(resolution.pipeline_managed_guardrails),
        )

    def policy_guardrails(self, policy_name: str, context: PolicyMatchContext | None = None) -> ResolvedPolicy:
        """One policy's guardrails through its inheritance chain, skipping members whose condition misses the
        context. Without a context every member counts"""
        guardrails, chain = self.native.policy_guardrails(
            policy_name, _context_row(context) if context is not None else None
        )
        return ResolvedPolicy(policy_name=policy_name, guardrails=guardrails, inheritance_chain=chain)

    def attached_policies(
        self, context: PolicyMatchContext, *, only_applying: bool
    ) -> tuple["PolicyAttachmentMatch", ...]:
        """The policies attached to the context, one per policy with the reason it matched. With
        ``only_applying``, attachments whose policy does not apply are dropped before defaults are considered"""
        matches, admissions = self.native.attached_policies(_context_row(context), only_applying)
        _log_ancestor_admissions(admissions, context)
        return tuple({"policy_name": policy_name, "matched_via": matched_via} for policy_name, matched_via in matches)


def build_policy_engine(
    policies: Mapping[str, Policy],
    versions: Mapping[str, tuple[str, Policy]] = _NO_VERSIONS,
    attachments: Sequence[PolicyAttachment] = (),
) -> PolicyEngine:
    return PolicyEngine(
        policies=policies,
        versions=versions,
        attachments=attachments,
        native=_native.PolicyEngine(
            tuple((name, _policy_row(policy)) for name, policy in policies.items()),
            tuple((policy_id, name, _policy_row(policy)) for policy_id, (name, policy) in versions.items()),
            tuple(_attachment_row(attachment) for attachment in attachments),
        ),
    )


class _CurrentPolicyEngine:
    """Holds the engine for the registries' current containers"""

    def __init__(self) -> None:
        self._engine: PolicyEngine | None = None

    def engine_for(
        self,
        policies: Mapping[str, Policy],
        versions: Mapping[str, tuple[str, Policy]],
        attachments: Sequence[PolicyAttachment],
    ) -> PolicyEngine:
        cached: Final = self._engine
        if (
            cached is not None
            and cached.policies is policies
            and cached.versions is versions
            and cached.attachments is attachments
        ):
            return cached
        engine: Final = build_policy_engine(policies, versions, attachments)
        self._engine = engine
        return engine


_CURRENT: Final = _CurrentPolicyEngine()


def current_policy_engine(attachments: Sequence[PolicyAttachment] | None = None) -> PolicyEngine:
    """The engine over the live policy registry and the given attachments, the live attachment registry's
    by default"""
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry

    policy_registry: Final = get_policy_registry()
    return _CURRENT.engine_for(
        policy_registry.current_policies() if policy_registry.is_initialized() else _NO_POLICIES,
        policy_registry.current_versions(),
        attachments if attachments is not None else get_attachment_registry().current_attachments(),
    )
