from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
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
    hidden_by: Final = _submit_model_hidden_by(deployment.model_name, llm_router.model_group_alias)
    if hidden_by is not None:
        verbose_proxy_logger.warning(
            "Policy engine: background response %s re-matches policies on retrieval as model group %s (%s), "
            "so a policy attached to the model name it was submitted as does not run on it",
            response_id,
            deployment.model_name,
            hidden_by,
        )
    return deployment.model_name


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
    matches: Final = get_attachment_registry().get_attached_policies_with_reasons(context)
    if not matches:
        return (), MappingProxyType({})
    applied_policy_names: Final = PolicyMatcher.get_policies_with_matching_conditions(
        policy_names=[match["policy_name"] for match in matches],  # mutable-ok: the matcher takes a list
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


def attach_post_call_pipelines_to_retrieval(
    data: dict[str, object],  # mutable-ok: request-state dict the policy engine hooks all write in place
    user_api_key_dict: "UserAPIKeyAuth",
    llm_router: "Router | None",
) -> None:
    if not get_policy_registry().is_initialized():
        return
    model_group: Final = _model_group_for_response_id(data.get("response_id"), llm_router)
    if isinstance(model_group, UngovernedRetrieval):
        verbose_proxy_logger.warning(
            "Policy engine: background response %s is retrieved without its post_call policy pipelines (%s)",
            data.get("response_id"),
            model_group.reason,
        )
        return
    context: Final = _retrieval_context(data, user_api_key_dict, model_group)
    post_call_pipelines, policy_sources = _post_call_pipelines_for_context(context)
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
        policy_sources={  # mutable-ok: add_policy_sources_to_metadata takes a dict
            policy_name: policy_sources[policy_name] for policy_name, _pipeline in added
        },
    )
    verbose_proxy_logger.debug(
        "Policy engine: attached post_call pipelines to the retrieval of background response %s (model group %s): %s",
        data.get("response_id"),
        model_group,
        ", ".join(policy_name for policy_name, _pipeline in added),
    )
