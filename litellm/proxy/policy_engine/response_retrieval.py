from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import MANAGED_OBJECT_STALENESS_CUTOFF_DAYS
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
    add_policy_sources_to_metadata,
    add_policy_to_applied_policies_header,
)
from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.common_utils import resolve_model_group_alias
from litellm.types.proxy.policy_engine import PolicyMatchContext
from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router

PolicyPipelines: TypeAlias = tuple[tuple[str, GuardrailPipeline], ...]

_POLICY_PIPELINES_ADAPTER: Final = TypeAdapter(PolicyPipelines)

_SUBMIT_POLICIES_TTL_SECONDS: Final = MANAGED_OBJECT_STALENESS_CUTOFF_DAYS * 24 * 60 * 60


class SubmitPostCallPolicies(BaseModel):
    model_config = ConfigDict(frozen=True)

    pipelines: PolicyPipelines
    policy_sources: Mapping[str, str]


class SubmitPoliciesCache(Protocol):
    async def async_set_cache(self, key: str, value: object, litellm_parent_otel_span: None, *, ttl: int) -> None: ...

    async def async_get_cache(self, key: str, litellm_parent_otel_span: None) -> object: ...


def _submit_policies_cache_key(response_id: str) -> str:
    upstream_response_id: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(response_id).get(
        "response_id", response_id
    )
    return f"policy_engine:background_response_post_call_policies:{upstream_response_id}"


async def remember_submit_post_call_policies(
    cache: SubmitPoliciesCache, response_id: str, submitted: SubmitPostCallPolicies
) -> None:
    await cache.async_set_cache(
        key=_submit_policies_cache_key(response_id),
        value=submitted.model_dump(mode="json"),
        litellm_parent_otel_span=None,
        ttl=_SUBMIT_POLICIES_TTL_SECONDS,
    )


async def _submit_post_call_policies(cache: SubmitPoliciesCache, response_id: str) -> SubmitPostCallPolicies | None:
    stored: Final = await cache.async_get_cache(
        key=_submit_policies_cache_key(response_id), litellm_parent_otel_span=None
    )
    return None if stored is None else SubmitPostCallPolicies.model_validate(stored)


@dataclass(frozen=True, slots=True)
class UngovernedRetrieval:
    reason: Literal["no router", "response id names no deployment", "deployment no longer in the router"]


def _model_group_for_response_id(response_id: object, llm_router: "Router | None") -> str | UngovernedRetrieval:
    if llm_router is None:
        return UngovernedRetrieval("no router")
    model_id: Final = (
        ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id) if isinstance(response_id, str) else None
    )
    if model_id is None:
        return UngovernedRetrieval("response id names no deployment")
    deployment: Final = llm_router.get_deployment(model_id)
    if deployment is None:
        return UngovernedRetrieval("deployment no longer in the router")
    return deployment.model_name


def _warn_if_submit_model_hidden(response_id: object, model_group: str, llm_router: "Router") -> None:
    hidden_by: Final = _submit_model_hidden_by(model_group, llm_router.model_group_alias)
    if hidden_by is None:
        return
    verbose_proxy_logger.warning(
        "Policy engine: background response %s has no submit-time post_call policies stored and re-matches "
        "policies on retrieval as model group %s (%s), so a policy attached to the model name it was submitted as "
        "does not run on it",
        response_id,
        model_group,
        hidden_by,
    )


def _submit_model_hidden_by(model_group: str, model_group_alias: Mapping[str, object]) -> str | None:
    if "*" in model_group:
        return "a wildcard deployment"
    aliases: Final = tuple(
        alias for alias in model_group_alias if resolve_model_group_alias(model_group_alias, alias) == model_group
    )
    if not aliases:
        return None
    return f"the target of model_group_alias {', '.join(aliases)}"


def _retrieval_context(
    data: Mapping[str, object], user_api_key_dict: "UserAPIKeyAuth", model_group: str
) -> PolicyMatchContext:
    team_alias: Final = user_api_key_dict.team_alias
    key_alias: Final = user_api_key_dict.key_alias
    return PolicyMatchContext(
        team_alias=team_alias if isinstance(team_alias, str) else None,
        key_alias=key_alias if isinstance(key_alias, str) else None,
        model=model_group,
        tags=get_tags_from_request_body(data) or None,
    )


def _post_call_pipelines_for_context(context: PolicyMatchContext) -> tuple[PolicyPipelines, Mapping[str, str]]:
    matches: Final = get_attachment_registry().get_attached_policies_with_reasons(
        context, PolicyMatcher.policy_applies(context)
    )
    if not matches:
        return (), MappingProxyType({})
    applied_policy_names: Final = PolicyMatcher.get_policies_with_matching_conditions(
        policy_names=[match["policy_name"] for match in matches],
        context=context,
    )
    post_call_pipelines: Final = tuple(
        (policy_name, pipeline)
        for policy_name, pipeline in PolicyResolver.resolve_pipelines_for_context(
            context=context, policy_names=applied_policy_names
        )
        if pipeline.mode == "post_call"
    )
    return post_call_pipelines, MappingProxyType({match["policy_name"]: match["matched_via"] for match in matches})


def _retrieval_post_call_pipelines(
    data: Mapping[str, object],
    user_api_key_dict: "UserAPIKeyAuth",
    llm_router: "Router | None",
    submitted: SubmitPostCallPolicies | None,
) -> tuple[PolicyPipelines, Mapping[str, str]]:
    response_id: Final = data.get("response_id")
    model_group: Final = _model_group_for_response_id(response_id, llm_router)
    if isinstance(model_group, UngovernedRetrieval):
        if submitted is None:
            verbose_proxy_logger.warning(
                "Policy engine: background response %s is retrieved without its post_call policy pipelines (%s)",
                response_id,
                model_group.reason,
            )
        return (), MappingProxyType({})
    if submitted is None and llm_router is not None:
        _warn_if_submit_model_hidden(response_id, model_group, llm_router)
    return _post_call_pipelines_for_context(_retrieval_context(data, user_api_key_dict, model_group))


async def attach_post_call_pipelines_to_retrieval(
    data: dict[str, object],  # mutable-ok: request-state dict the policy engine hooks all write in place
    user_api_key_dict: "UserAPIKeyAuth",
    llm_router: "Router | None",
    submit_policies_cache: SubmitPoliciesCache,
) -> None:
    if not get_policy_registry().is_initialized():
        return
    response_id: Final = data.get("response_id")
    submitted: Final = (
        await _submit_post_call_policies(submit_policies_cache, response_id) if isinstance(response_id, str) else None
    )
    retrieval_pipelines, retrieval_sources = _retrieval_post_call_pipelines(
        data, user_api_key_dict, llm_router, submitted
    )
    submitted_pipelines: Final = submitted.pipelines if submitted is not None else ()
    submitted_policy_names: Final = frozenset(policy_name for policy_name, _pipeline in submitted_pipelines)
    post_call_pipelines: Final = (
        *submitted_pipelines,
        *(
            (policy_name, pipeline)
            for policy_name, pipeline in retrieval_pipelines
            if policy_name not in submitted_policy_names
        ),
    )
    policy_sources: Final = {**retrieval_sources, **(submitted.policy_sources if submitted is not None else {})}
    _, bucket = get_or_create_metadata_bucket(data)
    already_attached: Final = _POLICY_PIPELINES_ADAPTER.validate_python(bucket.get("_guardrail_pipelines") or ())
    attached_policy_names: Final = frozenset(policy_name for policy_name, _pipeline in already_attached)
    added: Final = tuple(
        (policy_name, pipeline)
        for policy_name, pipeline in post_call_pipelines
        if policy_name not in attached_policy_names
    )
    if not added:
        return
    pipelines: Final = (*already_attached, *added)
    bucket["_guardrail_pipelines"] = pipelines
    bucket["_pipeline_managed_guardrails"] = frozenset(
        step.guardrail for _policy_name, pipeline in pipelines for step in pipeline.steps
    )
    for policy_name, _pipeline in added:
        add_policy_to_applied_policies_header(request_data=data, policy_name=policy_name)
    for _policy_name, pipeline in added:
        for step in pipeline.steps:
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=step.guardrail)
    add_policy_sources_to_metadata(
        request_data=data,
        policy_sources={
            policy_name: policy_sources[policy_name]
            for policy_name, _pipeline in added
            if policy_name in policy_sources
        },
    )
    verbose_proxy_logger.debug(
        "Policy engine: attached post_call pipelines to the retrieval of background response %s: %s",
        response_id,
        ", ".join(policy_name for policy_name, _pipeline in added),
    )
