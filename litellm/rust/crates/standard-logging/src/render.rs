use crate::{Caller, HiddenParams, Metadata, StandardLoggingPayload, wire};

fn metadata(
    metadata: Metadata,
    caller: Caller,
    requester_ip_address: Option<String>,
    user_agent: Option<String>,
) -> wire::Metadata {
    let Caller {
        key,
        user,
        team,
        organization,
        project,
        customer,
        end_user_id,
        request_route,
    } = caller;
    wire::Metadata {
        user_api_key_hash: key.hash,
        user_api_key_alias: key.alias,
        user_api_key_spend: key.budget.spend,
        user_api_key_max_budget: key.budget.max,
        user_api_key_budget_reset_at: key.budget_reset_at,
        user_api_key_auth_metadata: key.metadata,
        user_api_key_user_id: user.id,
        user_api_key_user_email: user.email,
        user_api_key_user_spend: user.budget.spend,
        user_api_key_user_max_budget: user.budget.max,
        user_api_key_team_id: team.id,
        user_api_key_team_alias: team.alias,
        user_api_key_team_spend: team.budget.spend,
        user_api_key_team_max_budget: team.budget.max,
        user_api_key_org_id: organization.id,
        user_api_key_org_alias: organization.alias,
        user_api_key_project_id: project.id,
        user_api_key_project_alias: project.alias,
        user_api_key_customer_member_id: customer.member.id,
        user_api_key_customer_member_alias: customer.member.alias,
        user_api_key_customer_team_id: customer.team.id,
        user_api_key_customer_team_alias: customer.team.alias,
        user_api_key_customer_org_id: customer.organization.id,
        user_api_key_customer_org_alias: customer.organization.alias,
        user_api_key_customer_budget_scopes: customer.budget_scopes,
        user_api_key_customer_hierarchy: customer.hierarchy,
        user_api_key_end_user_id: end_user_id,
        user_api_key_request_route: request_route,
        spend_logs_metadata: metadata.spend_logs_metadata,
        requester_ip_address,
        user_agent,
        requester_metadata: metadata.requester_metadata,
        requester_custom_headers: metadata.requester_custom_headers,
        prompt_management_metadata: metadata.prompt_management_metadata,
        applied_guardrails: metadata.applied_guardrails,
        mcp_tool_call_metadata: metadata.mcp_tool_call_metadata,
        vector_store_request_metadata: metadata.vector_store_request_metadata,
        routing_decision: metadata.routing_decision,
        kubernetes_pod_routing: metadata.kubernetes_pod_routing,
        usage_object: metadata.usage_object,
        cold_storage_object_key: metadata.cold_storage_object_key,
        team_alias: metadata.team_alias,
        team_id: metadata.team_id,
        used_client_oauth_token: metadata.used_client_oauth_token,
        claude_subscription_usage: metadata.claude_subscription_usage,
    }
}

fn hidden_params(hidden_params: HiddenParams, cache_key: Option<String>) -> wire::HiddenParams {
    wire::HiddenParams {
        model_id: hidden_params.model_id,
        cache_key,
        api_base: hidden_params.api_base,
        response_cost: hidden_params.response_cost,
        additional_headers: hidden_params.additional_headers,
        litellm_overhead_time_ms: hidden_params.litellm_overhead_time_ms,
        batch_models: hidden_params.batch_models,
        batch_successful_requests: hidden_params.batch_successful_requests,
        batch_failed_requests: hidden_params.batch_failed_requests,
        litellm_model_name: hidden_params.litellm_model_name,
        usage_object: hidden_params.usage_object,
    }
}

impl From<StandardLoggingPayload> for wire::Payload {
    fn from(payload: StandardLoggingPayload) -> Self {
        Self {
            id: payload.id,
            litellm_call_id: payload.litellm_call_id,
            trace_id: payload.trace_id,
            session_id: payload.session_id,
            call_type: payload.call_type,
            cache_hit: payload.cache_hit,
            stream: payload.stream,
            status: payload.status,
            status_fields: wire::StatusFields {
                llm_api_status: payload.status,
                guardrail_status: payload.guardrail_status,
            },
            custom_llm_provider: payload.custom_llm_provider,
            saved_cache_cost: payload.saved_cache_cost,
            start_time: payload.start_time,
            end_time: payload.end_time,
            completion_start_time: payload.completion_start_time,
            response_time: payload.response_time,
            model: payload.model,
            metadata: metadata(
                payload.metadata,
                payload.caller,
                payload.requester_ip_address.clone(),
                payload.user_agent.clone(),
            ),
            cache_key: payload.cache_key.clone(),
            cost_finalized: payload.cost_finalized,
            response_cost: payload.response_cost,
            cost_breakdown: payload.cost_breakdown,
            autorouter_savings: payload.autorouter_savings,
            autorouter_savings_estimate: payload.autorouter_savings_estimate,
            autorouter_baseline_observation: payload.autorouter_baseline_observation,
            total_tokens: payload.total_tokens,
            prompt_tokens: payload.prompt_tokens,
            completion_tokens: payload.completion_tokens,
            request_tags: payload.request_tags,
            request_model_access_groups: payload.request_model_access_groups,
            end_user: payload.end_user,
            api_base: payload.api_base,
            model_group: payload.model_group,
            model_id: payload.model_id,
            requester_ip_address: payload.requester_ip_address,
            user_agent: payload.user_agent,
            messages: payload.messages,
            response: payload.response,
            model_parameters: payload.model_parameters,
            hidden_params: hidden_params(payload.hidden_params, payload.cache_key),
            model_map_information: payload.model_map_information,
            error_str: payload.error_str,
            error_information: payload.error_information,
            response_cost_failure_debug_info: payload.response_cost_failure_debug_info,
            zero_cost_diagnostic: payload.zero_cost_diagnostic,
            guardrail_information: payload.guardrail_information,
            standard_built_in_tools_params: payload.standard_built_in_tools_params,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use crate::{
        Budget, Caller, ClaudeSubscriptionUsage, Customer, ErrorInformation, GuardrailStatus,
        HiddenParams, Key, Metadata, ModelInformation, Named, StandardLoggingPayload, Status, Team,
        User,
    };

    fn object(value: Value) -> Map<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    fn text(value: &str) -> Option<String> {
        Some(value.to_owned())
    }

    fn named(id: &str, alias: &str) -> Named {
        Named {
            id: text(id),
            alias: text(alias),
        }
    }

    fn budget(spend: f64, max: f64) -> Budget {
        Budget {
            spend: Some(spend),
            max: Some(max),
        }
    }

    fn caller() -> Caller {
        Caller {
            key: Key {
                hash: text("88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"),
                alias: text("ci-key"),
                budget: budget(1.5, 10.0),
                budget_reset_at: text("2026-02-01T00:00:00+00:00"),
                metadata: Some(object(json!({"cost_center": "research"}))),
            },
            user: User {
                id: text("user-1"),
                email: text("user@example.com"),
                budget: budget(2.5, 20.0),
            },
            team: Team {
                id: text("team-1"),
                alias: text("platform"),
                budget: budget(3.5, 30.0),
            },
            organization: named("org-1", "acme"),
            project: named("project-1", "search"),
            customer: Customer {
                member: named("customer-member-1", "customer-member"),
                team: named("customer-team-1", "customer-team"),
                organization: named("customer-org-1", "customer-org"),
                budget_scopes: text("[]"),
                hierarchy: Some(json!({
                    "customer_member_alias": "customer-member",
                    "customer_member_id": "customer-member-1",
                    "customer_org_alias": "customer-org",
                    "customer_org_id": "customer-org-1",
                    "customer_team_alias": "customer-team",
                    "customer_team_id": "customer-team-1",
                    "scopes": [],
                    "team_scopes": [],
                })),
            },
            end_user_id: text("customer-1"),
            request_route: text("/v1/messages"),
        }
    }

    fn success() -> StandardLoggingPayload {
        let usage = json!({
            "completion_tokens": 5,
            "prompt_tokens": 10,
            "total_tokens": 15,
            "completion_tokens_details": null,
            "prompt_tokens_details": null,
        });
        StandardLoggingPayload {
            id: "msg_1".to_owned(),
            litellm_call_id: text("call-1"),
            trace_id: "trace-1".to_owned(),
            session_id: String::new(),
            call_type: "anthropic_messages".to_owned(),
            cache_hit: Some(false),
            cache_key: text("cache-key-1"),
            stream: None,
            status: Status::Success,
            guardrail_status: GuardrailStatus::NotRun,
            custom_llm_provider: text("anthropic"),
            saved_cache_cost: 0.0,
            start_time: 1_767_268_800.0,
            end_time: 1_767_268_802.5,
            completion_start_time: 1_767_268_801.0,
            response_time: 2.5,
            model: "claude-unlisted".to_owned(),
            caller: caller(),
            requester_ip_address: text("10.0.0.1"),
            user_agent: text("curl/8.7.1"),
            metadata: Metadata {
                spend_logs_metadata: Some(object(json!({"project": "alpha"}))),
                requester_custom_headers: Some(BTreeMap::from([(
                    "x-team".to_owned(),
                    "platform".to_owned(),
                )])),
                usage_object: Some(object(usage.clone())),
                ..Metadata::default()
            },
            cost_finalized: false,
            response_cost: 0.000_125,
            cost_breakdown: None,
            autorouter_savings: None,
            autorouter_savings_estimate: None,
            autorouter_baseline_observation: None,
            total_tokens: 15,
            prompt_tokens: 10,
            completion_tokens: 5,
            request_tags: vec![
                json!("batch-job"),
                json!("User-Agent: curl"),
                json!("User-Agent: curl/8.7.1"),
            ],
            request_model_access_groups: Vec::new(),
            end_user: text("customer-1"),
            api_base: "https://api.anthropic.com".to_owned(),
            model_group: text("claude"),
            model_id: text("deployment-1"),
            messages: json!([{"role": "user", "content": "Hey"}]),
            response: json!({
                "id": "msg_1",
                "created": 1_767_268_800,
                "model": "claude-unlisted",
                "object": "chat.completion",
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hi",
                        "role": "assistant",
                        "tool_calls": null,
                        "function_call": null,
                    },
                }],
                "usage": usage,
            }),
            model_parameters: object(json!({"max_tokens": 64, "temperature": 0.5})),
            hidden_params: HiddenParams {
                model_id: text("deployment-1"),
                api_base: text("https://api.anthropic.com/"),
                response_cost: Some(0.000_125),
                additional_headers: Some(object(json!({
                    "x_ratelimit_remaining_requests": 99,
                    "llm_provider-request-id": "req_1",
                }))),
                litellm_overhead_time_ms: Some(3.5),
                litellm_model_name: text("claude-unlisted"),
                ..HiddenParams::default()
            },
            model_map_information: ModelInformation {
                model_map_key: "claude-unlisted".to_owned(),
                model_map_value: None,
            },
            error_str: None,
            error_information: Some(ErrorInformation {
                error_code: text(""),
                error_class: text(""),
                llm_provider: text(""),
                traceback: text(""),
                error_message: text(""),
                ..ErrorInformation::default()
            }),
            response_cost_failure_debug_info: None,
            zero_cost_diagnostic: None,
            guardrail_information: None,
            standard_built_in_tools_params: None,
        }
    }

    fn failure() -> StandardLoggingPayload {
        let served = success();
        StandardLoggingPayload {
            id: "call-1".to_owned(),
            cache_key: None,
            status: Status::Failure,
            metadata: Metadata {
                usage_object: Some(Map::new()),
                ..served.metadata
            },
            response_cost: 0.0,
            total_tokens: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            response: json!({}),
            hidden_params: HiddenParams {
                response_cost: Some(0.0),
                ..HiddenParams::default()
            },
            model_map_information: ModelInformation {
                model_map_key: String::new(),
                model_map_value: None,
            },
            error_str: text("litellm.RateLimitError: slow down"),
            error_information: Some(ErrorInformation {
                error_code: text("429"),
                error_class: text("RateLimitError"),
                llm_provider: text("anthropic"),
                traceback: text(""),
                error_message: text("litellm.RateLimitError: slow down"),
                error_rate_limit_category: text("vendor_rate_limit"),
                normalized_error: text("429_RATE_LIMIT_EXCEEDED"),
                ..ErrorInformation::default()
            }),
            ..served
        }
    }

    #[rstest]
    #[case::allowance(
        ClaudeSubscriptionUsage {
            is_overage: false,
            status: text("allowed"),
            overage_status: None,
        },
        json!({"is_overage": false, "status": "allowed", "overage_status": null})
    )]
    #[case::overage(
        ClaudeSubscriptionUsage {
            is_overage: true,
            status: text("rejected"),
            overage_status: text("allowed"),
        },
        json!({"is_overage": true, "status": "rejected", "overage_status": "allowed"})
    )]
    fn claude_subscription_usage_is_logged_as_python_logs_it(
        #[case] usage: ClaudeSubscriptionUsage,
        #[case] logged: Value,
    ) {
        let served = success();
        let payload = StandardLoggingPayload {
            metadata: Metadata {
                used_client_oauth_token: Some(true),
                claude_subscription_usage: Some(usage),
                ..served.metadata
            },
            ..served
        };

        let document = serde_json::to_value(payload).unwrap();

        assert_eq!(document["metadata"]["claude_subscription_usage"], logged);
        assert_eq!(document["metadata"]["used_client_oauth_token"], json!(true));
    }

    #[rstest]
    #[case::success(
        success(),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/success.json"))
    )]
    #[case::failure(
        failure(),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/failure.json"))
    )]
    fn a_payload_serializes_to_the_document_python_logs_for_the_same_call(
        #[case] payload: StandardLoggingPayload,
        #[case] document: &str,
    ) {
        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            serde_json::from_str::<Value>(document).unwrap()
        );
    }
}
