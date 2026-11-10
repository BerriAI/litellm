"""The proxy policy engine, resolved in Rust over a snapshot of the policy and attachment registries.

The engine for the live registries is rebuilt whenever a registry replaces one of its containers.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, TypeVar

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


class _FullMatcher(Protocol):
    def fullmatch(self, string: str, /) -> object: ...


_PatternRow: TypeAlias = tuple[str, _FullMatcher | None]
_Names: TypeAlias = Sequence[str]
_PolicyRow: TypeAlias = tuple[str | None, _Names, _Names, Sequence[_PatternRow] | None, _Names | None]
_AttachmentRow: TypeAlias = tuple[str, bool, _Names, _Names, _Names, _Names, int | None, bool]
_ContextRow: TypeAlias = tuple[str | None, str | None, str | None, _Names]

_NO_POLICIES: Final[Mapping[str, Policy]] = MappingProxyType({})
_NO_VERSIONS: Final[Mapping[str, tuple[str, Policy]]] = MappingProxyType({})

_FIRST_SURROGATE: Final = 0xD800
_FIRST_STAND_IN: Final = 0x10F800
_TO_STAND_IN: Final = MappingProxyType(
    {code: _FIRST_STAND_IN + code - _FIRST_SURROGATE for code in range(_FIRST_SURROGATE, 0xE000)}
)
_FROM_STAND_IN: Final = MappingProxyType({stand_in: code for code, stand_in in _TO_STAND_IN.items()})
_NOT_UTF8: Final = (UnicodeEncodeError, TypeError)


@dataclass(frozen=True, slots=True)
class PolicyEngineResolution:
    attached_policies: tuple[str, ...]
    unapplied_policies: tuple[str, ...]
    condition_misses: tuple[str, ...]
    policy_guardrails: Mapping[str, tuple[str, ...]]
    applied_policies: tuple[str, ...]
    policy_sources: Mapping[str, str]
    guardrails: tuple[str, ...]
    pipelines: tuple[tuple[str, GuardrailPipeline], ...]
    pipeline_managed_guardrails: frozenset[str]


def _same(text: str) -> str:
    return text


def _stand_in(text: str) -> str:
    return text.translate(_TO_STAND_IN)


def _restore(text: str) -> str:
    return text.translate(_FROM_STAND_IN)


@dataclass(frozen=True, slots=True)
class _RestoringMatcher:
    compiled: re.Pattern[str]

    def fullmatch(self, string: str, /) -> re.Match[str] | None:
        return self.compiled.fullmatch(_restore(string))


@dataclass(frozen=True, slots=True)
class _Text:
    """How strings cross into Rust. Rust strings are UTF-8, so lone surrogates, which Python strings and YAML
    allow, cross as the last 2048 code points of plane 16's private use area and are restored on the way back.
    Text that already holds those private use code points reads as the surrogates they stand in for"""

    into_rust: Callable[[str], str]
    from_rust: Callable[[str], str]

    def optional(self, text: str | None) -> str | None:
        return self.into_rust(text) if text is not None else None

    def names(self, names: Iterable[str]) -> tuple[str, ...]:
        return tuple(map(self.into_rust, names))

    def restored(self, names: Iterable[str]) -> tuple[str, ...]:
        return tuple(map(self.from_rust, names))

    def pairs(self, pairs: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple((self.from_rust(first), self.from_rust(second)) for first, second in pairs)

    def condition_row(self, policy: Policy) -> tuple[_PatternRow, ...] | None:
        model: Final = policy.condition.model if policy.condition is not None else None
        if model is None:
            return None
        patterns: Final = (model,) if isinstance(model, str) else model
        return tuple((self.into_rust(pattern), _matcher(pattern)) for pattern in patterns)

    def policy_row(self, policy: Policy) -> _PolicyRow:
        return (
            self.optional(policy.inherit or None),
            self.names(policy.guardrails.get_add()),
            self.names(policy.guardrails.get_remove()),
            self.condition_row(policy),
            self.names(step.guardrail for step in policy.pipeline.steps) if policy.pipeline is not None else None,
        )

    def attachment_row(self, attachment: PolicyAttachment) -> _AttachmentRow:
        return (
            self.into_rust(attachment.policy),
            attachment.is_global(),
            self.names(attachment.teams or ()),
            self.names(attachment.keys or ()),
            self.names(attachment.models or ()),
            self.names(attachment.tags or ()),
            attachment.priority,
            attachment.default,
        )

    def context_row(self, context: PolicyMatchContext) -> _ContextRow:
        return (
            self.optional(context.team_alias),
            self.optional(context.key_alias),
            self.optional(context.model),
            self.names(context.tags or ()),
        )

    def native_engine(
        self,
        policies: Mapping[str, Policy],
        versions: Mapping[str, tuple[str, Policy]],
        attachments: Sequence[PolicyAttachment],
    ) -> _native.PolicyEngine:
        return _native.PolicyEngine(
            tuple((self.into_rust(name), self.policy_row(policy)) for name, policy in policies.items()),
            tuple(
                (self.into_rust(policy_id), self.into_rust(name), self.policy_row(policy))
                for policy_id, (name, policy) in versions.items()
            ),
            tuple(self.attachment_row(attachment) for attachment in attachments),
        )


_AS_IS: Final = _Text(_same, _same)
_STOOD_IN: Final = _Text(_stand_in, _restore)


def _compiled(pattern: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern)
    except (re.error, OverflowError, RecursionError, ValueError):
        return None


def _matcher(pattern: str) -> _FullMatcher | None:
    compiled: Final = _compiled(pattern)
    return _RestoringMatcher(compiled) if compiled is not None else None


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


_Result: Final = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class PolicyEngine:
    policies: Mapping[str, Policy]
    versions: Mapping[str, tuple[str, Policy]]
    attachments: Sequence[PolicyAttachment]
    native: _native.PolicyEngine
    text: _Text = _AS_IS

    def _call(self, call: Callable[[_Text], _Result]) -> tuple[_Text, _Result]:
        if self.text is _STOOD_IN:
            return _STOOD_IN, call(_STOOD_IN)
        try:
            return _AS_IS, call(_AS_IS)
        except _NOT_UTF8:
            return _STOOD_IN, call(_STOOD_IN)

    def resolve(
        self,
        context: PolicyMatchContext,
        names: Sequence[str] = (),
        version_ids: Sequence[str] = (),
    ) -> PolicyEngineResolution:
        """The policies a request applies, from its attachments plus the names and version ids it sent"""
        text, resolution = self._call(
            lambda text: self.native.resolve(text.context_row(context), text.names(names), text.names(version_ids))
        )
        _log_ancestor_admissions(text.pairs(resolution.ancestor_admissions), context)
        return PolicyEngineResolution(
            attached_policies=text.restored(resolution.attached_policies),
            unapplied_policies=text.restored(resolution.unapplied_policies),
            condition_misses=text.restored(resolution.condition_misses),
            policy_guardrails=MappingProxyType(
                {text.from_rust(name): text.restored(added) for name, added in resolution.policy_guardrails}
            ),
            applied_policies=text.restored(resolution.applied_policies),
            policy_sources=MappingProxyType(dict(text.pairs(resolution.policy_sources))),
            guardrails=text.restored(resolution.guardrails),
            pipelines=tuple(
                self._pipeline_of(text.from_rust(name), version_id, text) for name, version_id in resolution.pipelines
            ),
            pipeline_managed_guardrails=frozenset(text.restored(resolution.pipeline_managed_guardrails)),
        )

    def _pipeline_of(self, name: str, version_id: str | None, text: _Text) -> tuple[str, GuardrailPipeline]:
        policy: Final = self.versions[text.from_rust(version_id)][1] if version_id is not None else self.policies[name]
        return name, _pipeline(policy)

    def policy_guardrails(self, policy_name: str, context: PolicyMatchContext | None = None) -> ResolvedPolicy:
        """One policy's guardrails through its inheritance chain, skipping members whose condition misses the
        context. Without a context every member counts"""
        text, (guardrails, chain) = self._call(
            lambda text: self.native.policy_guardrails(
                text.into_rust(policy_name), text.context_row(context) if context is not None else None
            )
        )
        return ResolvedPolicy(
            policy_name=policy_name,
            guardrails=list(text.restored(guardrails)),
            inheritance_chain=list(text.restored(chain)),
        )

    def attached_policies(
        self, context: PolicyMatchContext, *, only_applying: bool
    ) -> tuple["PolicyAttachmentMatch", ...]:
        """The policies attached to the context, one per policy with the reason it matched. With
        ``only_applying``, attachments whose policy does not apply are dropped before defaults are considered"""
        text, (matches, admissions) = self._call(
            lambda text: self.native.attached_policies(text.context_row(context), only_applying)
        )
        _log_ancestor_admissions(text.pairs(admissions), context)
        return tuple(
            {"policy_name": policy_name, "matched_via": matched_via} for policy_name, matched_via in text.pairs(matches)
        )


def build_policy_engine(
    policies: Mapping[str, Policy],
    versions: Mapping[str, tuple[str, Policy]] = _NO_VERSIONS,
    attachments: Sequence[PolicyAttachment] = (),
) -> PolicyEngine:
    engine: Final = _built_policy_engine(policies, versions, attachments)
    for policy_name in engine.text.restored(engine.native.circular_policies()):
        verbose_proxy_logger.warning("Circular inheritance detected for policy '%s'", policy_name)
    return engine


def _built_policy_engine(
    policies: Mapping[str, Policy],
    versions: Mapping[str, tuple[str, Policy]],
    attachments: Sequence[PolicyAttachment],
) -> PolicyEngine:
    try:
        return PolicyEngine(policies, versions, attachments, _AS_IS.native_engine(policies, versions, attachments))
    except _NOT_UTF8:
        return PolicyEngine(
            policies, versions, attachments, _STOOD_IN.native_engine(policies, versions, attachments), _STOOD_IN
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
