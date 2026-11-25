//! Renders a [`StandardLoggingPayload`] into the wire shape of `wire`, the document
//! `get_standard_logging_object_payload` in litellm/litellm_core_utils/litellm_logging.py logs.

use crate::{App, Caller, HiddenParams, Metadata, StandardLoggingPayload, wire};

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
        app,
        end_user_id,
        request_route,
    } = caller;
    let (user_api_key_app_id, user_api_key_app_group_id, user_api_key_app_alias) = match app {
        Some(App {
            id,
            group_id,
            alias,
        }) => (Some(id), group_id, Some(alias)),
        None => (None, None, None),
    };
    wire::Metadata {
        user_api_key_hash: key.hash,
        user_api_key_alias: key.alias,
        user_api_key_spend: key.budget.spend,
        user_api_key_max_budget: key.budget.max,
        user_api_key_budget_reset_at: key.budget_reset_at,
        user_api_key_user_spend: user.budget.spend,
        user_api_key_user_max_budget: user.budget.max,
        user_api_key_team_spend: team.budget.spend,
        user_api_key_team_max_budget: team.budget.max,
        user_api_key_team_id: team.id,
        user_api_key_org_id: organization.id,
        user_api_key_org_alias: organization.alias,
        user_api_key_project_id: project.id,
        user_api_key_project_alias: project.alias,
        user_api_key_user_id: user.id,
        user_api_key_team_alias: team.alias,
        user_api_key_user_email: user.email,
        user_api_key_end_user_id: end_user_id,
        user_api_key_customer_member_id: customer.member.id,
        user_api_key_customer_team_id: customer.team.id,
        user_api_key_customer_org_id: customer.organization.id,
        user_api_key_request_route: request_route,
        spend_logs_metadata: metadata.spend_logs_metadata,
        requester_ip_address,
        user_agent,
        requester_metadata: metadata.requester_metadata,
        prompt_management_metadata: metadata.prompt_management_metadata,
        applied_guardrails: metadata.applied_guardrails,
        mcp_tool_call_metadata: metadata.mcp_tool_call_metadata,
        vector_store_request_metadata: metadata.vector_store_request_metadata,
        routing_decision: metadata.routing_decision,
        usage_object: metadata.usage_object,
        requester_custom_headers: metadata.requester_custom_headers,
        cold_storage_object_key: metadata.cold_storage_object_key,
        user_api_key_auth_metadata: key.metadata,
        team_alias: metadata.team_alias,
        team_id: metadata.team_id,
        kubernetes_pod_routing: metadata.kubernetes_pod_routing,
        used_client_oauth_token: metadata.used_client_oauth_token,
        is_silent_experiment: metadata.is_silent_experiment,
        user_api_key_app_id,
        user_api_key_app_group_id,
        user_api_key_app_alias,
        user_api_key_customer_org_alias: customer.organization.alias,
        user_api_key_customer_team_alias: customer.team.alias,
        user_api_key_customer_member_alias: customer.member.alias,
        user_api_key_customer_budget_scopes: customer.budget_scopes,
        user_api_key_customer_hierarchy: customer.hierarchy,
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
        upstream_provider: hidden_params.upstream_provider,
    }
}

impl From<StandardLoggingPayload> for wire::Payload {
    fn from(payload: StandardLoggingPayload) -> Self {
        Self {
            classifier_input: payload.classifier_audit.classifier_input,
            originating_request_masked: payload.classifier_audit.originating_request_masked,
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
        App, Budget, Caller, ClassifierAudit, ClaudeSubscriptionUsage, ClientDisconnected,
        Customer, ErrorInformation, GuardrailStatus, HiddenParams, Key, Metadata, ModelInformation,
        Named, StandardLoggingPayload, Status, Team, User,
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
            app: None,
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
            classifier_audit: ClassifierAudit::default(),
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
            error_information: Some(
                ErrorInformation {
                    error_code: text(""),
                    error_class: text(""),
                    llm_provider: text(""),
                    traceback: text(""),
                    error_message: text(""),
                    ..ErrorInformation::default()
                }
                .into(),
            ),
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
            error_information: Some(
                ErrorInformation {
                    error_code: text("429"),
                    error_class: text("RateLimitError"),
                    llm_provider: text("anthropic"),
                    traceback: text(""),
                    error_message: text("litellm.RateLimitError: slow down"),
                    error_rate_limit_category: text("vendor_rate_limit"),
                    normalized_error: text("429_RATE_LIMIT_EXCEEDED"),
                    ..ErrorInformation::default()
                }
                .into(),
            ),
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

    fn with_audit(classifier_input: bool) -> StandardLoggingPayload {
        StandardLoggingPayload {
            classifier_audit: ClassifierAudit {
                classifier_input: classifier_input.then(|| object(json!({"messages": []}))),
                originating_request_masked: Some(object(json!({"model": "auto"}))),
            },
            ..success()
        }
    }

    fn with_app(group_id: Option<String>) -> StandardLoggingPayload {
        let served = success();
        StandardLoggingPayload {
            caller: Caller {
                app: Some(App {
                    id: "app-1".to_owned(),
                    group_id,
                    alias: "billing-app".to_owned(),
                }),
                ..served.caller
            },
            ..served
        }
    }

    fn silent_experiment() -> StandardLoggingPayload {
        let served = success();
        StandardLoggingPayload {
            metadata: Metadata {
                is_silent_experiment: true,
                ..served.metadata
            },
            ..served
        }
    }

    fn routed_upstream() -> StandardLoggingPayload {
        let served = success();
        StandardLoggingPayload {
            hidden_params: HiddenParams {
                upstream_provider: text("Anthropic"),
                ..served.hidden_params
            },
            ..served
        }
    }

    /// The keys logged just before and just after `key` in the object at `pointer`, or `None` when
    /// the object holds no such key.
    fn neighbours(
        payload: StandardLoggingPayload,
        pointer: &str,
        key: &str,
    ) -> Option<(Option<String>, Option<String>)> {
        let document = serde_json::to_value(payload).unwrap();
        let keys: Vec<&String> = document
            .pointer(pointer)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .collect();
        let at = keys.iter().position(|logged| *logged == key)?;
        Some((
            at.checked_sub(1).map(|before| keys[before].clone()),
            keys.get(at + 1).map(|after| (*after).clone()),
        ))
    }

    #[rstest]
    #[case::classifier_input_leads_the_payload(
        with_audit(true),
        "",
        "classifier_input",
        None,
        text("originating_request_masked")
    )]
    #[case::originating_request_alone_leads_the_payload(
        with_audit(false),
        "",
        "originating_request_masked",
        None,
        text("id")
    )]
    #[case::silent_experiment_follows_the_oauth_flag(
        silent_experiment(),
        "/metadata",
        "is_silent_experiment",
        text("used_client_oauth_token"),
        text("user_api_key_customer_org_alias")
    )]
    #[case::app_group_sits_between_app_id_and_alias(
        with_app(text("group-1")),
        "/metadata",
        "user_api_key_app_group_id",
        text("user_api_key_app_id"),
        text("user_api_key_app_alias")
    )]
    #[case::app_without_group_logs_id_then_alias(
        with_app(None),
        "/metadata",
        "user_api_key_app_alias",
        text("user_api_key_app_id"),
        text("user_api_key_customer_org_alias")
    )]
    #[case::upstream_provider_closes_hidden_params(
        routed_upstream(),
        "/hidden_params",
        "upstream_provider",
        text("usage_object"),
        None
    )]
    fn a_key_python_logs_only_when_set_lands_where_python_puts_it(
        #[case] payload: StandardLoggingPayload,
        #[case] pointer: &str,
        #[case] key: &str,
        #[case] before: Option<String>,
        #[case] after: Option<String>,
    ) {
        assert_eq!(neighbours(payload, pointer, key), Some((before, after)));
    }

    #[rstest]
    #[case::classifier_input("", "classifier_input")]
    #[case::originating_request_masked("", "originating_request_masked")]
    #[case::silent_experiment("/metadata", "is_silent_experiment")]
    #[case::app_id("/metadata", "user_api_key_app_id")]
    #[case::app_group_id("/metadata", "user_api_key_app_group_id")]
    #[case::app_alias("/metadata", "user_api_key_app_alias")]
    #[case::upstream_provider("/hidden_params", "upstream_provider")]
    fn a_key_python_logs_only_when_set_is_absent_when_unset(
        #[case] pointer: &str,
        #[case] key: &str,
    ) {
        assert_eq!(neighbours(success(), pointer, key), None);
    }

    #[test]
    fn an_app_outside_a_group_logs_no_group_id() {
        assert_eq!(
            neighbours(with_app(None), "/metadata", "user_api_key_app_group_id"),
            None
        );
    }

    const KEY_HASH: &str = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b";
    const SERVED_MODEL: &str = "anthropic/claude-sonnet-4-5-20250929";
    const PROVIDER_ERROR: &str = r#"litellm.BadRequestError: AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: must be greater than or equal to 1"}}"#;
    const LIMIT_RESET: &str = "Limit resets at: 2026-01-01 12:01:00 UTC";

    /// Builds one scenario's payload, taking from its golden only the values this crate carries
    /// as opaque JSON that other crates build: the logged response, the usage object, the cost
    /// breakdown and the model map entry.
    type Scenario = fn(&Value) -> StandardLoggingPayload;

    fn opaque(golden: &Value, pointer: &str) -> Option<Value> {
        golden
            .pointer(pointer)
            .filter(|value| !value.is_null())
            .cloned()
    }

    fn opaque_object(golden: &Value, pointer: &str) -> Option<Map<String, Value>> {
        opaque(golden, pointer).map(|value| serde_json::from_value(value).unwrap())
    }

    /// A virtual key with no user, team, organization, project, customer or budget behind it.
    fn bare_key() -> Caller {
        Caller {
            key: Key {
                hash: text(KEY_HASH),
                alias: text("ci-key"),
                budget: Budget {
                    spend: Some(0.25),
                    max: None,
                },
                budget_reset_at: None,
                metadata: Some(Map::new()),
            },
            request_route: text("/v1/messages"),
            ..Caller::default()
        }
    }

    fn served_without_error() -> ErrorInformation {
        ErrorInformation {
            error_code: text(""),
            error_class: text(""),
            llm_provider: text(""),
            traceback: text(""),
            error_message: text(""),
            ..ErrorInformation::default()
        }
    }

    fn unary_success(golden: &Value) -> StandardLoggingPayload {
        StandardLoggingPayload {
            classifier_audit: ClassifierAudit::default(),
            id: "msg_01".to_owned(),
            litellm_call_id: text("call-1"),
            trace_id: "trace-1".to_owned(),
            session_id: String::new(),
            call_type: "anthropic_messages".to_owned(),
            cache_hit: Some(false),
            cache_key: None,
            stream: None,
            status: Status::Success,
            guardrail_status: GuardrailStatus::NotRun,
            custom_llm_provider: text("anthropic"),
            saved_cache_cost: 0.0,
            start_time: 1_767_268_800.0,
            end_time: 1_767_268_802.5,
            completion_start_time: 1_767_268_802.5,
            response_time: 2.5,
            model: SERVED_MODEL.to_owned(),
            caller: bare_key(),
            requester_ip_address: text("10.0.0.1"),
            user_agent: text("curl/8.7.1"),
            metadata: Metadata {
                requester_metadata: Some(Map::new()),
                requester_custom_headers: Some(BTreeMap::new()),
                usage_object: opaque_object(golden, "/metadata/usage_object"),
                ..Metadata::default()
            },
            cost_finalized: false,
            response_cost: 0.000_105,
            cost_breakdown: opaque(golden, "/cost_breakdown"),
            autorouter_savings: None,
            autorouter_savings_estimate: None,
            autorouter_baseline_observation: None,
            total_tokens: 15,
            prompt_tokens: 10,
            completion_tokens: 5,
            request_tags: vec![json!("User-Agent: curl"), json!("User-Agent: curl/8.7.1")],
            request_model_access_groups: Vec::new(),
            end_user: None,
            api_base: "https://api.anthropic.com".to_owned(),
            model_group: text("claude-sonnet-4-5"),
            model_id: text("deployment-1"),
            messages: json!([{"role": "user", "content": "Hey"}]),
            response: golden["response"].clone(),
            model_parameters: object(json!({"max_tokens": 64})),
            hidden_params: HiddenParams {
                response_cost: Some(0.000_105),
                additional_headers: Some(Map::new()),
                litellm_model_name: text(SERVED_MODEL),
                ..HiddenParams::default()
            },
            model_map_information: ModelInformation {
                model_map_key: SERVED_MODEL.to_owned(),
                model_map_value: opaque(golden, "/model_map_information/model_map_value"),
            },
            error_str: None,
            error_information: Some(served_without_error().into()),
            response_cost_failure_debug_info: None,
            zero_cost_diagnostic: None,
            guardrail_information: None,
            standard_built_in_tools_params: None,
        }
    }

    fn stream_success(golden: &Value) -> StandardLoggingPayload {
        StandardLoggingPayload {
            stream: Some(true),
            completion_start_time: 1_767_268_800.75,
            response_time: 0.75,
            model_parameters: object(json!({"max_tokens": 64, "stream": true})),
            ..unary_success(golden)
        }
    }

    fn cache_hit(golden: &Value) -> StandardLoggingPayload {
        let served = unary_success(golden);
        StandardLoggingPayload {
            id: "msg_01_cache_hit<time>".to_owned(),
            cache_hit: Some(true),
            saved_cache_cost: 0.000_105,
            response_cost: 0.0,
            hidden_params: HiddenParams {
                response_cost: Some(0.0),
                ..served.hidden_params
            },
            ..served
        }
    }

    fn unserved(served: StandardLoggingPayload) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: "call-1".to_owned(),
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
            ..served
        }
    }

    fn provider_4xx(golden: &Value) -> StandardLoggingPayload {
        let failed = unserved(unary_success(golden));
        StandardLoggingPayload {
            hidden_params: HiddenParams {
                response_cost: Some(0.0),
                additional_headers: None,
                ..failed.hidden_params
            },
            error_str: text(PROVIDER_ERROR),
            error_information: Some(
                ErrorInformation {
                    error_code: text("400"),
                    error_class: text("BadRequestError"),
                    llm_provider: text("anthropic"),
                    error_message: text(PROVIDER_ERROR),
                    normalized_error: text("400_INVALID_REQUEST"),
                    ..served_without_error()
                }
                .into(),
            ),
            ..failed
        }
    }

    fn key_limit_429(golden: &Value) -> StandardLoggingPayload {
        StandardLoggingPayload {
            custom_llm_provider: None,
            model: "claude-sonnet-4-5".to_owned(),
            api_base: String::new(),
            model_group: text(""),
            model_id: text(""),
            hidden_params: HiddenParams {
                response_cost: Some(0.0),
                additional_headers: Some(object(json!({
                    "retry-after": "60",
                    "rate_limit_type": "requests",
                    "rate_limited_by": "api_key",
                    "reset_at": "2026-01-01 12:01:00 UTC",
                }))),
                ..HiddenParams::default()
            },
            model_map_information: ModelInformation {
                model_map_key: String::new(),
                model_map_value: None,
            },
            error_str: Some(format!(
                "429: Rate limit exceeded for api_key: {KEY_HASH}. Limit type: requests. Current limit: 1, Remaining: 0. {LIMIT_RESET}"
            )),
            error_information: Some(
                ErrorInformation {
                    error_code: text("429"),
                    error_class: text("ProxyRateLimitError"),
                    llm_provider: text("litellm_proxy"),
                    error_message: Some(format!(
                        "litellm.RateLimitError: Rate limit exceeded for REDACTED Limit type: requests. Current limit: 1, Remaining: 0. {LIMIT_RESET}"
                    )),
                    error_rate_limit_category: text("litellm_rate_limit"),
                    normalized_error: text("429_RATE_LIMIT_EXCEEDED"),
                    ..served_without_error()
                }
                .into(),
            ),
            ..unserved(unary_success(golden))
        }
    }

    fn client_disconnect(golden: &Value) -> StandardLoggingPayload {
        let streamed = stream_success(golden);
        StandardLoggingPayload {
            response_cost: 0.000_06,
            total_tokens: 12,
            completion_tokens: 2,
            hidden_params: HiddenParams {
                response_cost: Some(0.000_06),
                ..streamed.hidden_params
            },
            error_str: text("Client disconnected the request"),
            error_information: Some(ClientDisconnected.into()),
            ..streamed
        }
    }

    #[rstest]
    #[case::unary_success(
        unary_success,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/unary_success.json"))
    )]
    #[case::stream_success(
        stream_success,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/stream_success.json"))
    )]
    #[case::cache_hit(
        cache_hit,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/cache_hit.json"))
    )]
    #[case::provider_4xx(
        provider_4xx,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/provider_4xx.json"))
    )]
    #[case::key_limit_429(
        key_limit_429,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/key_limit_429.json"))
    )]
    #[case::client_disconnect(
        client_disconnect,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/client_disconnect.json"))
    )]
    fn a_messages_call_serializes_to_the_bytes_python_logs_for_it(
        #[case] scenario: Scenario,
        #[case] golden: &str,
    ) {
        let document: Value = serde_json::from_str(golden).unwrap();

        assert_eq!(
            serde_json::to_string(&scenario(&document)).unwrap(),
            golden.trim_end()
        );
    }
}
