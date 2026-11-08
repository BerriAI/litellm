from typing import Final

from litellm._auth import access, admission, managed


def access_decision(facts: access.AccessFacts) -> access.AccessDecision:
    match facts:
        case access.ConcurrentFailures(failures=failures, terminal=terminal):
            return access.AccessDecision(
                failure_index=next(
                    (index for index, failed in enumerate(terminal) if failed),
                    next((index for index, failed in enumerate(failures) if failed), None),
                )
            )
        case access.UserParamRequest(enabled=enabled):
            return access.AccessDecision(read="UserParam" if enabled else None)
        case access.GuardrailRequest(requested_keys=keys):
            modifies: Final = any(
                key in keys
                for key in (
                    "guardrails",
                    "disable_global_guardrails",
                    "disable_global_guardrail",
                    "opted_out_global_guardrails",
                )
            )
            return access.AccessDecision(read="GuardrailPermission" if modifies else None)
        case access.TraceRequest(required=required):
            return access.AccessDecision(next_trace="Gate" if required else None)
        case access.Search(name=name, allowed=allowed, strict=strict):
            return access.AccessDecision(denial="Search" if (strict or allowed) and name not in allowed else None)
        case access.Blocked(blocked=blocked):
            return access.AccessDecision(denial="Blocked" if blocked else None)
        case access.UserParam(
            enabled=enabled, method=method, llm_route=llm_route, mcp_route=mcp_route, has_user=has_user
        ):
            return access.AccessDecision(
                denial="UserParam"
                if enabled and method and method.upper() == "POST" and llm_route and not mcp_route and not has_user
                else None
            )
        case access.ClientTags(enabled=enabled, llm_route=llm_route, has_tags=has_tags):
            return access.AccessDecision(denial="ClientTags" if enabled and llm_route and has_tags else None)
        case access.Guardrails(requested_keys=keys, modify_guardrails=modify):
            return access.AccessDecision(
                denial="Guardrails"
                if modify is False
                and any(
                    key in keys
                    for key in (
                        "guardrails",
                        "disable_global_guardrails",
                        "disable_global_guardrail",
                        "opted_out_global_guardrails",
                    )
                )
                else None
            )
        case access.AgentTrace(required=required, header=header):
            return access.AccessDecision(denial="AgentTrace" if required and not header else None)
        case access.Tools(requested=requested, key_allowed=key, team_allowed=team):
            tool_allowed: Final = key or team
            rejected: Final = tuple(tool for tool in requested if tool_allowed and tool not in tool_allowed)
            return access.AccessDecision(denial="Tools" if rejected else None, rejected=rejected)
        case access.GrantLayers():
            if not facts.deny_by_default or not facts.identity_present or facts.master_key or facts.dashboard:
                return access.AccessDecision()
            layers: Final[tuple[access.GrantLayer, ...]] = (
                *(("key",) if facts.virtual_key and not facts.session_token else ()),
                *(("team",) if facts.team_present else ()),
            )
            return access.AccessDecision(layers=layers or ("user",))
        case access.VectorStores(requested=requested, allowed=allowed, strict=strict):
            if not allowed:
                return access.AccessDecision(denial="MissingVectorGrant" if strict else None)
            rejected_index: Final = next((index for index, store in enumerate(requested) if store not in allowed), None)
            return access.AccessDecision(
                denial="VectorStores" if rejected_index is not None else None, rejected_index=rejected_index
            )
        case access.TeamLookup(dashboard=dashboard, missing=missing, recorded_models=models, allow_unavailable=allow):
            if not dashboard and not missing and not models and allow is None:
                return access.AccessDecision(read="UnavailablePolicy")
            return access.AccessDecision(
                denial="TeamLookup" if not dashboard and (missing or (not models and not allow)) else None
            )
        case access.TraceGate(
            required=required, method=method, llm_route=llm, passthrough=passthrough, telemetry=telemetry
        ):
            return access.AccessDecision(
                next_trace="Header"
                if required and method not in ("GET", "HEAD", "OPTIONS") and (llm or passthrough) and not telemetry
                else None
            )
        case access.TraceStep(source=source, value=value, present=present, headers_only=headers_only):
            if value:
                return access.AccessDecision()
            if source == "Header" and not headers_only:
                return access.AccessDecision(next_trace="Body")
            if source == "Body":
                return access.AccessDecision(next_trace="Selected" if present else "Traceparent")
            if source == "Traceparent":
                return access.AccessDecision(next_trace="Selected")
            if source == "Selected" and not present:
                return access.AccessDecision(next_trace="Other")
            return access.AccessDecision(denial="TeamTrace")
        case access.TeamTrace():
            gate: Final = access_decision(
                access.TraceGate(facts.required, facts.method, facts.llm_route, facts.passthrough, facts.telemetry)
            )
            trace_present: Final = facts.header or (
                not facts.headers_only
                and (
                    facts.body_trace
                    or (not facts.body_trace_present and facts.traceparent_id)
                    or (facts.selected_trace if facts.selected_trace_present else facts.other_trace)
                )
            )
            return access.AccessDecision(denial="TeamTrace" if gate.next_trace and not trace_present else None)


def common_plan(facts: admission.CommonGate) -> admission.CommonPlan:
    if facts.force_virtual_key_checks:
        return admission.CommonPlan(None)
    if not facts.force_virtual_key_checks and (facts.public_route or facts.additional_public_route):
        return admission.CommonPlan("PublicRoute")
    if facts.unauthenticated_passthrough:
        return admission.CommonPlan("UnauthenticatedPassthrough")
    if (
        not facts.master_key_present
        and not facts.jwt_enabled
        and not facts.oauth_enabled
        and not facts.oauth_proxy_enabled
    ):
        return admission.CommonPlan("NoAuthDevelopment")
    if facts.custom_auth and not facts.custom_auth_common_checks:
        return admission.CommonPlan("CustomAuth")
    return admission.CommonPlan(None)


def reservation_plan(facts: admission.Reservation) -> admission.ReservationPlan:
    return admission.ReservationPlan(
        not facts.skip_budget_checks and not facts.disabled,
        facts.apply_user_budget_to_team_keys,
        facts.fail_closed or facts.billing_agent,
    )


def actor(facts: managed.Actor) -> managed.ManagedDecision:
    agent, context = facts.agent, facts.context
    if not agent.enabled or agent.identity is None or not agent.identity.active:
        return managed.ManagedDecision("Deny", "Disabled", None)
    if context is None:
        return managed.ManagedDecision("Deny", "MissingContext", None)
    if context.agent_id != agent.agent_id or context.binding_revision != agent.identity.revision:
        return managed.ManagedDecision("Deny", "ChangedIdentity", None)
    if agent.execution_mode not in (context.mode, "both"):
        return managed.ManagedDecision("Deny", "ExecutionMode", None)
    if context.mode == "delegated" and not context.user_id:
        return managed.ManagedDecision("Deny", "MissingHuman", None)
    return managed.ManagedDecision("Allow", None, None)
