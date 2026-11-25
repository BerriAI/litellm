//! The `StandardLoggingPayload` of a `/v1/messages` call the gateway served or failed, built from
//! the call's record the way `get_standard_logging_object_payload` in
//! `litellm/litellm_core_utils/litellm_logging.py` builds it for the same call, and its
//! `StandardLoggingPayloadSetup` helpers.
//!
//! Every input that would change a key from the value set here is one admit sends to Python: a
//! key with a team, owner, end user row, budget or access group, any `x-litellm-*` header but the
//! call id, body `metadata` beyond `user_id`, and any `litellm_settings` or `general_settings`
//! key the config types do not name (`request_correlation_in_logs`, cold storage,
//! `extra_spend_tag_headers`, guardrails, prompt management, MCP, vector stores). So the keys
//! those inputs fill (`session_id`, the team, organization, project, customer and app keys,
//! `spend_logs_metadata`, `cold_storage_object_key`, `guardrail_information`, the autorouter
//! keys, `classifier_*`) keep the values Python logs without them.
//!
//! The record carries no cost yet, so the cost keys are what Python logs for a call whose cost
//! it could not work out: `response_cost` 0, no `cost_breakdown`, `saved_cache_cost` 0, and the
//! served model's cost map key with no entry.

use std::time::SystemTime;

use litellm_gateway_inference::record::{
    AttemptRecord, CallFailure, CallState, Charged, Ended, ServedMessage, unix,
};
use litellm_redaction::SecretRedactor;
use litellm_router_core::Deployment;
use litellm_standard_logging::{
    Budget, CLIENT_DISCONNECTED_MESSAGE, Caller, ClassifierAudit, ClientDisconnected,
    ErrorInformation, GuardrailStatus, HiddenParams, Key, LoggedErrorInformation, Metadata,
    ModelInformation, StandardLoggingPayload, Status, User, additional_headers,
    anthropic_response_headers, api_base, cache_hit_id, claude_subscription_usage, httpx_items,
    messages_response::{self, LoggedResponse, ResponseContext},
    reconstruct_model_name, used_client_oauth_token,
};
use serde_json::{Map, Value};

use super::fields::route;

/// What the payload reads besides the call's record.
pub struct Context<'a> {
    /// The deployment a call that ended on an error last ran on, which the caller looks up in the
    /// router's registry by [`failed_on`]. A served call's own deployment is on its record.
    pub failed_on: Option<&'a Deployment>,
    /// Python's `redact_string`, over the logged error message, traceback and `error_str`.
    pub redactor: &'a SecretRedactor,
    /// When the payload is built: the `created` of the logged response and a cache hit's id.
    pub now: SystemTime,
    /// Python's `token_counter(text=..., count_response_tokens=True)`, which the logged
    /// response's usage counts reasoning with when the answer does not report it.
    pub count_tokens: &'a dyn Fn(&str) -> u64,
}

/// A call the payload is not built for.
#[derive(Debug, thiserror::Error)]
pub enum Unbuilt {
    #[error("the call has not ended")]
    NotEnded,
    #[error("the call's answer is not logged as Python logs it")]
    Response(#[from] messages_response::Error),
}

/// The id of the deployment a call that ended on an error last ran on, for
/// [`Context::failed_on`]. `None` for a call refused before it was routed.
pub fn failed_on(entries: &CallState) -> Option<&str> {
    entries.routing().map(|routing| routing.deployment_id())
}

/// The provider `deployment` is called through, as Python's `custom_llm_provider` names it.
pub(super) fn deployment_provider(deployment: &Deployment) -> Option<String> {
    deployment
        .resolved_provider
        .clone()
        .or_else(|| deployment.params.custom_llm_provider.clone())
        .or_else(|| {
            deployment
                .params
                .model
                .split_once('/')
                .map(|(provider, _)| provider.to_owned())
        })
}

/// `user_api_key_*` of the key the call is charged to, as `get_sanitized_user_information_from_key`
/// stamps it for a bare virtual key or the master key: no team, organization, project,
/// customer or app. `end_user` is the request's end user, which auth names on the key.
fn caller(charged: &Charged, request_route: Option<&str>, end_user: Option<&str>) -> Caller {
    Caller {
        key: Key {
            hash: charged.key_hash.clone(),
            alias: charged.key_alias.clone(),
            budget: Budget {
                spend: charged.spend,
                max: charged.max_budget,
            },
            budget_reset_at: charged.budget_reset_at.clone(),
            metadata: charged.metadata.clone(),
        },
        user: User {
            id: charged.user_id.clone(),
            ..User::default()
        },
        end_user_id: end_user.map(str::to_owned),
        request_route: request_route.map(str::to_owned),
        ..Caller::default()
    }
}

/// `get_error_information(None)`: what a call that raised nothing logs as `error_information`.
fn nothing_raised() -> ErrorInformation {
    ErrorInformation {
        error_code: Some(String::new()),
        error_class: Some(String::new()),
        llm_provider: Some(String::new()),
        traceback: Some(String::new()),
        error_message: Some(String::new()),
        ..ErrorInformation::default()
    }
}

/// `get_error_information_for_logging_payload`: the exception's `error_information` and the
/// failure handler's `redact_string(str(exception))`, or the fixed 499 of a client that left.
fn error(
    failure: Option<&CallFailure>,
    redact: &impl Fn(&str) -> String,
) -> (LoggedErrorInformation, Option<String>) {
    match failure {
        Some(raised @ CallFailure::Raised { .. }) => (
            raised.error_information(redact).into(),
            Some(raised.error_str(redact)),
        ),
        Some(CallFailure::ClientDisconnected) => (
            ClientDisconnected.into(),
            Some(CLIENT_DISCONNECTED_MESSAGE.to_owned()),
        ),
        None => (nothing_raised().into(), None),
    }
}

/// The `ModelResponse` Python logs for the answer that served the call:
/// `_handle_anthropic_messages_response_logging` over a complete answer or a stream.
/// `model` is `model_call_details["model"]`, which names a stream's response.
fn logged_response(
    served: &ServedMessage,
    entries: &CallState,
    model: &str,
    created: u64,
    count_tokens: &dyn Fn(&str) -> u64,
) -> Result<LoggedResponse, messages_response::Error> {
    let minted = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let fallback_id = match served {
        ServedMessage::Complete(_) => minted.as_str(),
        ServedMessage::Streamed(_) => entries.call_id.as_deref().unwrap_or_default(),
    };
    let context = ResponseContext {
        created,
        fallback_id,
        requested_speed: entries
            .logged_request
            .model_parameters
            .get("speed")
            .and_then(Value::as_str),
        count_tokens,
    };
    served.logged_response(model, &context)
}

/// A count of the logged usage, `usage_dict.get(name, 0)`.
fn tokens(usage: &Map<String, Value>, name: &str) -> u64 {
    usage.get(name).and_then(Value::as_u64).unwrap_or_default()
}

/// The `hidden_params.additional_headers` Python logs: the exception's response headers on a
/// failure (`_extract_response_obj_and_hidden_params`, none when it carried none), those
/// `AnthropicConfig.transform_response` keeps on a complete answer, and none on a stream, whose
/// joined `ModelResponse` carries no hidden params.
fn logged_headers(
    failure: Option<&CallFailure>,
    served: Option<&ServedMessage>,
    attempt: Option<&AttemptRecord>,
) -> Option<Map<String, Value>> {
    if let Some(CallFailure::Raised { headers, .. }) = failure {
        return (!headers.is_empty()).then(|| additional_headers(&httpx_items(headers)));
    }
    match served? {
        ServedMessage::Complete(_) => {
            let provider_headers = attempt.map_or(&[][..], |attempt| &attempt.provider_headers);
            Some(additional_headers(&anthropic_response_headers(
                provider_headers,
            )))
        }
        ServedMessage::Streamed(_) => None,
    }
}

/// The call's `StandardLoggingPayload`, as `get_standard_logging_object_payload` builds it for a
/// `/v1/messages` call that ended: with `status="success"` and the served answer, or
/// `status="failure"` and the exception the call raised.
pub fn standard_logging_payload(
    entries: &CallState,
    context: &Context<'_>,
) -> Result<StandardLoggingPayload, Unbuilt> {
    let end = entries.end().ok_or(Unbuilt::NotEnded)?;
    let served = entries.served();
    let deployment = match served {
        Some(served) => served.deployment.as_deref(),
        None => context.failed_on,
    };
    let provider = deployment.and_then(deployment_provider);
    let raised = matches!(entries.failure, Some(CallFailure::Raised { .. }))
        || entries.ended() == Some(Ended::Failed);
    let call_id = entries.call_id.clone().unwrap_or_default();

    // `model_call_details["model"]`: the deployment's model without its provider once routed,
    // else the model the request named.
    let requested = entries.model().unwrap_or_default();
    let routed_model = deployment.map(|deployment| {
        let model = deployment.params.model.as_str();
        model.split_once('/').map_or(model, |(_, model)| model)
    });
    let kwargs_model = routed_model.unwrap_or(requested);
    let model = reconstruct_model_name(
        kwargs_model,
        provider.as_deref(),
        deployment.map(|deployment| deployment.params.model.as_str()),
    );

    let served_message = entries.served_message.as_ref().filter(|_| !raised);
    let created = context
        .now
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let response = served_message
        .map(|message| {
            logged_response(
                message,
                entries,
                kwargs_model,
                created,
                context.count_tokens,
            )
        })
        .transpose()?;
    let usage = match &response {
        Some(response) => match serde_json::to_value(&response.usage) {
            Ok(Value::Object(usage)) => usage,
            _ => Map::new(),
        },
        None => Map::new(),
    };

    let cache_hit = entries
        .attempt
        .as_ref()
        .and_then(AttemptRecord::cache_key)
        .is_some();
    let answered_id = response
        .as_ref()
        .map(|response| response.id.as_str())
        .filter(|id| !id.is_empty());
    let id = answered_id.unwrap_or(&call_id);
    let id = match cache_hit {
        true => cache_hit_id(id, unix(context.now)),
        false => id.to_owned(),
    };

    let start = unix(entries.start());
    let end = unix(end);
    let completion_start = served
        .and_then(|served| served.first_event)
        .or(entries.first_event)
        .map_or(end, unix);
    let response_time = match entries.stream() {
        true => completion_start - start,
        false => end - start,
    };

    let attempt = match served {
        Some(_) => entries.attempt.as_ref(),
        None => entries.failed_attempts.last(),
    };
    let facts = &entries.request_facts;
    let client_sent_oauth_token = Some(facts.client_sent_oauth_token);
    let served_headers = served
        .and(entries.attempt.as_ref())
        .map_or(&[][..], |attempt| &attempt.provider_headers);
    let logged = &entries.logged_request;
    let redact = |text: &str| context.redactor.redact(text);
    let (error_information, error_str) = error(entries.failure.as_ref(), &redact);

    // A failure logs `response_cost` 0 (`_failure_handler_helper_fn`) and a cache hit its own
    // cost, both copied into the hidden params; a served call's cost is not on the record.
    let hidden_response_cost = (raised || cache_hit).then_some(0.0);

    Ok(StandardLoggingPayload {
        classifier_audit: ClassifierAudit::default(),
        id,
        litellm_call_id: entries.call_id.clone(),
        trace_id: entries.trace_id.clone(),
        session_id: String::new(),
        call_type: entries
            .call_type
            .map(<&str>::from)
            .unwrap_or_default()
            .to_owned(),
        cache_hit: Some(cache_hit),
        cache_key: None,
        stream: (entries.stream() && response.is_some()).then_some(true),
        status: match raised {
            true => Status::Failure,
            false => Status::Success,
        },
        guardrail_status: GuardrailStatus::NotRun,
        custom_llm_provider: provider.clone(),
        saved_cache_cost: 0.0,
        start_time: start,
        end_time: end,
        completion_start_time: completion_start,
        response_time,
        model,
        caller: caller(
            &entries.charged,
            entries.call_type.map(route),
            logged.end_user.as_deref(),
        ),
        requester_ip_address: Some(facts.requester_ip_address.clone()),
        user_agent: Some(facts.user_agent.clone()),
        metadata: Metadata {
            requester_metadata: logged.requester_metadata.clone(),
            requester_custom_headers: facts.requester_custom_headers.clone(),
            usage_object: Some(usage.clone()),
            used_client_oauth_token: used_client_oauth_token(
                client_sent_oauth_token,
                provider.as_deref(),
            ),
            claude_subscription_usage: claude_subscription_usage(
                provider.as_deref(),
                client_sent_oauth_token,
                served_headers,
            ),
            ..Metadata::default()
        },
        cost_finalized: false,
        response_cost: 0.0,
        cost_breakdown: None,
        autorouter_savings: None,
        autorouter_savings_estimate: None,
        autorouter_baseline_observation: None,
        total_tokens: tokens(&usage, "total_tokens"),
        prompt_tokens: tokens(&usage, "prompt_tokens"),
        completion_tokens: tokens(&usage, "completion_tokens"),
        request_tags: facts
            .user_agent_tags
            .iter()
            .map(|tag| Value::from(tag.as_str()))
            .collect(),
        request_model_access_groups: Vec::new(),
        end_user: logged.end_user.clone(),
        api_base: api_base(attempt.and_then(|attempt| attempt.url.as_deref())),
        model_group: Some(match (served, deployment) {
            (Some(served), _) => served.model_group.clone(),
            (None, Some(deployment)) => deployment.model_name.clone(),
            (None, None) => String::new(),
        }),
        model_id: Some(deployment.map_or_else(String::new, |deployment| deployment.id.clone())),
        messages: logged.messages.clone(),
        response: response
            .map(|response| serde_json::to_value(response).unwrap_or_default())
            .unwrap_or_else(|| Value::Object(Map::new())),
        model_parameters: logged.model_parameters.clone(),
        hidden_params: HiddenParams {
            response_cost: hidden_response_cost,
            additional_headers: logged_headers(
                entries.failure.as_ref(),
                served_message,
                entries.attempt.as_ref(),
            ),
            litellm_model_name: deployment.map(|deployment| deployment.params.model.clone()),
            ..HiddenParams::default()
        },
        model_map_information: ModelInformation {
            model_map_key: deployment
                .map(|deployment| deployment.params.model.clone())
                .unwrap_or_default(),
            model_map_value: None,
        },
        error_str,
        error_information: Some(error_information),
        response_cost_failure_debug_info: None,
        zero_cost_diagnostic: None,
        guardrail_information: None,
        standard_built_in_tools_params: None,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use litellm_gateway_inference::{
        CallType, ServedCall,
        record::{self, CallRecord, LoggedRequest, RequestFacts},
    };
    use litellm_router_core::DeploymentSpec;
    use litellm_standard_logging::{
        ExceptionClass, LoggedException, RateLimitCategory, messages_response::StreamedMessage,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const KEY_HASH: &str = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b";
    const API_BASE: &str = "https://api.anthropic.com";
    const MODEL_GROUP: &str = "claude-sonnet-4-5";
    const PROVIDER_ERROR: &str = r#"litellm.BadRequestError: AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: must be greater than or equal to 1"}}"#;
    const LIMIT: &str = "Limit type: requests. Current limit: 1, Remaining: 0. Limit resets at: 2026-01-01 12:01:00 UTC";

    /// The deployment `record.py` routes its calls to.
    fn deployment() -> Arc<Deployment> {
        let spec = DeploymentSpec {
            model_name: MODEL_GROUP.to_owned(),
            litellm_params: Map::from_iter([
                (
                    "model".to_owned(),
                    json!("anthropic/claude-sonnet-4-5-20250929"),
                ),
                ("api_base".to_owned(), json!(API_BASE)),
            ]),
            model_info: Map::from_iter([("id".to_owned(), json!("deployment-1"))]),
            extra: Map::new(),
            provider: Some("anthropic".to_owned()),
        };
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    fn pairs(headers: &[(&str, &str)]) -> Vec<(String, String)> {
        headers
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn object(value: Value) -> Map<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    /// The `/v1/messages` request `record.py` makes with its bare virtual key, as the record
    /// holds it once read.
    fn requested(entries: &mut CallState, stream: bool) {
        entries.call_type = Some(CallType::AnthropicMessages);
        entries.call_id = Some("call-1".to_owned());
        entries.trace_id = "trace-1".to_owned();
        entries.charged = Charged {
            key_hash: Some(KEY_HASH.to_owned()),
            key_alias: Some("ci-key".to_owned()),
            spend: Some(0.25),
            metadata: Some(Map::new()),
            ..Charged::default()
        };
        let mut body = object(json!({
            "model": MODEL_GROUP,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hey"}],
        }));
        let mut model_parameters = object(json!({"max_tokens": 64}));
        if stream {
            body.insert("stream".to_owned(), json!(true));
            model_parameters.insert("stream".to_owned(), json!(true));
        }
        entries.hold(body);
        entries.release();
        entries.request_facts = RequestFacts {
            requester_ip_address: "10.0.0.1".to_owned(),
            user_agent: "curl/8.7.1".to_owned(),
            user_agent_tags: vec![
                "User-Agent: curl".to_owned(),
                "User-Agent: curl/8.7.1".to_owned(),
            ],
            requester_custom_headers: Some(BTreeMap::new()),
            client_sent_oauth_token: false,
        };
        entries.logged_request = LoggedRequest {
            messages: json!([{"role": "user", "content": "Hey"}]),
            model_parameters,
            requester_metadata: Some(Map::new()),
            end_user: None,
        };
    }

    fn sent_to_the_provider() -> AttemptRecord {
        AttemptRecord {
            url: Some(API_BASE.to_owned()),
            ..AttemptRecord::default()
        }
    }

    fn served(entries: &mut CallState, deployment: &Arc<Deployment>, message: ServedMessage) {
        let first_event = entries
            .stream()
            .then(|| entries.start() + Duration::from_millis(750));
        entries.served_message = Some(message);
        entries.attempt = Some(sent_to_the_provider());
        entries.serve(ServedCall {
            model_group: MODEL_GROUP.to_owned(),
            deployment: Some(deployment.clone()),
            response_id: Some("msg_01".to_owned()),
            usage: None,
            first_event,
        });
    }

    /// The answer `record.py` serves, which it logs for a stream too, as its assembled response.
    fn answer() -> ServedMessage {
        ServedMessage::Complete(object(json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        })))
    }

    fn unary_success(entries: &mut CallState, deployment: &Arc<Deployment>) {
        requested(entries, false);
        served(entries, deployment, answer());
    }

    fn stream_success(entries: &mut CallState, deployment: &Arc<Deployment>) {
        requested(entries, true);
        served(entries, deployment, answer());
    }

    fn failed(
        entries: &mut CallState,
        exception: LoggedException,
        error_str: String,
        headers: Vec<(String, String)>,
    ) {
        entries.failure = Some(CallFailure::Raised {
            exception,
            error_str,
            headers,
        });
        entries.fail();
    }

    fn provider_4xx(entries: &mut CallState, _: &Arc<Deployment>) {
        requested(entries, false);
        entries.failed_attempts = vec![sent_to_the_provider()];
        failed(
            entries,
            LoggedException {
                class: ExceptionClass::BadRequest,
                code: "400".to_owned(),
                llm_provider: "anthropic".to_owned(),
                message: PROVIDER_ERROR.to_owned(),
                provider_request_id: None,
                rate_limit_category: None,
                rate_limit_type: None,
                proxy_type: None,
                traceback: String::new(),
            },
            PROVIDER_ERROR.to_owned(),
            Vec::new(),
        );
    }

    /// The key's rpm limit refused the call before it was routed, with the exception
    /// `record.py` raises, which names no limit type.
    fn key_limit_429(entries: &mut CallState, _: &Arc<Deployment>) {
        requested(entries, false);
        failed(
            entries,
            LoggedException {
                class: ExceptionClass::ProxyRateLimit,
                code: "429".to_owned(),
                llm_provider: "litellm_proxy".to_owned(),
                message: format!(
                    "litellm.RateLimitError: Rate limit exceeded for api_key: {KEY_HASH}. {LIMIT}"
                ),
                provider_request_id: None,
                rate_limit_category: Some(RateLimitCategory::LitellmRateLimit),
                rate_limit_type: None,
                proxy_type: None,
                traceback: String::new(),
            },
            format!("429: Rate limit exceeded for api_key: {KEY_HASH}. {LIMIT}"),
            pairs(&[
                ("retry-after", "60"),
                ("rate_limit_type", "requests"),
                ("rate_limited_by", "api_key"),
                ("reset_at", "2026-01-01 12:01:00 UTC"),
            ]),
        );
    }

    type Scenario = fn(&mut CallState, &Arc<Deployment>);

    /// The payload built for the call `scenario` records, as the JSON document it renders to.
    /// `failed_on` hands the builder the deployment the router registry names for a call that
    /// ended on a failed attempt.
    fn built(scenario: Scenario, failed_on: bool) -> Result<Value, Unbuilt> {
        let deployment = deployment();
        let redactor = SecretRedactor::new(16);
        let count_tokens = |_: &str| 0;
        let scoped = || {
            record::write(|entries| {
                scenario(entries, &deployment);
                let context = Context {
                    failed_on: failed_on.then_some(&*deployment),
                    redactor: &redactor,
                    now: SystemTime::now(),
                    count_tokens: &count_tokens,
                };
                standard_logging_payload(entries, &context)
                    .map(|payload| serde_json::to_value(payload).unwrap())
            })
            .unwrap()
        };
        futures_util::FutureExt::now_or_never(record::scope(CallRecord::new(), async { scoped() }))
            .unwrap()
    }

    /// The values the goldens hold that the record cannot: the times, which `record.py` fixes,
    /// the cost keys, which the record does not carry yet, and the response's `created`, which
    /// the goldens mask to 0.
    const FROM_THE_GOLDEN: [&str; 9] = [
        "/startTime",
        "/endTime",
        "/completionStartTime",
        "/response_time",
        "/saved_cache_cost",
        "/response_cost",
        "/cost_breakdown",
        "/hidden_params/response_cost",
        "/model_map_information/model_map_value",
    ];

    fn masked(mut built: Value, golden: &Value) -> Value {
        for pointer in FROM_THE_GOLDEN {
            *built.pointer_mut(pointer).unwrap() = golden.pointer(pointer).unwrap().clone();
        }
        if let Some(created) = built.pointer_mut("/response/created") {
            *created = json!(0);
        }
        built
    }

    /// The golden as the proxy logs the same call: `add_litellm_data_to_request` always stamps
    /// whether the client sent an OAuth token, which `record.py` leaves out of its metadata, and
    /// the failure handler redacts `error_str`, which `record.py` passes as raised.
    fn as_the_proxy_logs(mut golden: Value, error_str: Option<&str>) -> Value {
        golden["metadata"]["used_client_oauth_token"] = json!(false);
        if let Some(error_str) = error_str {
            golden["error_str"] = json!(error_str);
        }
        golden
    }

    #[rstest]
    #[case::unary_success(
        unary_success,
        false,
        None,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../standard-logging/tests/goldens/unary_success.json"))
    )]
    #[case::stream_success(
        stream_success,
        false,
        None,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../standard-logging/tests/goldens/stream_success.json"))
    )]
    #[case::provider_4xx(
        provider_4xx,
        true,
        None,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../standard-logging/tests/goldens/provider_4xx.json"))
    )]
    #[case::key_limit_429(
        key_limit_429,
        false,
        Some(concat!("429: Rate limit exceeded for REDACTED ", "Limit type: requests. Current limit: 1, Remaining: 0. Limit resets at: 2026-01-01 12:01:00 UTC")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../standard-logging/tests/goldens/key_limit_429.json"))
    )]
    fn a_messages_call_logs_the_payload_python_logs_for_it(
        #[case] scenario: Scenario,
        #[case] failed_on: bool,
        #[case] redacted_error_str: Option<&str>,
        #[case] golden: &str,
    ) {
        let golden: Value = serde_json::from_str(golden).unwrap();
        let built = masked(built(scenario, failed_on).unwrap(), &golden);
        assert_eq!(
            serde_json::to_string(&built).unwrap(),
            serde_json::to_string(&as_the_proxy_logs(golden, redacted_error_str)).unwrap()
        );
    }

    fn seconds(document: &Value, key: &str) -> f64 {
        document[key].as_f64().unwrap()
    }

    /// The seconds from the start a call logs as its completion start and its response time.
    fn timed(scenario: Scenario) -> (f64, f64, f64) {
        let document = built(scenario, false).unwrap();
        let start = seconds(&document, "startTime");
        (
            seconds(&document, "completionStartTime") - start,
            seconds(&document, "endTime") - start,
            seconds(&document, "response_time"),
        )
    }

    #[rstest]
    fn a_complete_answer_starts_and_takes_until_it_ends() {
        let (completion_start, end, response_time) = timed(unary_success);
        assert_eq!((completion_start, response_time), (end, end));
    }

    #[rstest]
    fn a_stream_starts_and_takes_until_its_first_event() {
        let (completion_start, _, response_time) = timed(stream_success);
        assert!((completion_start - 0.75).abs() < 1e-3 && (response_time - 0.75).abs() < 1e-3);
    }

    const STREAM_START: &str = r#"{"type":"message_start","message":{"id":"msg_s","model":"claude-sonnet-4-5-20250929","usage":{"input_tokens":3,"output_tokens":1}}}"#;
    const STREAM_TEXT: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
    const STREAM_END: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#;

    fn streamed(entries: &mut CallState, deployment: &Arc<Deployment>) {
        requested(entries, true);
        let mut stream = StreamedMessage::default();
        for event in [STREAM_START, STREAM_TEXT, STREAM_END] {
            stream.see(event);
        }
        served(
            entries,
            deployment,
            ServedMessage::Streamed(Box::new(stream)),
        );
    }

    #[rstest]
    fn a_stream_logs_the_response_its_events_add_up_to_without_headers() {
        let document = built(streamed, false).unwrap();
        assert_eq!(
            (
                &document["id"],
                &document["stream"],
                &document["response"]["model"],
                &document["total_tokens"],
                &document["hidden_params"]["additional_headers"],
            ),
            (
                &json!("msg_s"),
                &json!(true),
                &json!("claude-sonnet-4-5-20250929"),
                &json!(5),
                &Value::Null,
            )
        );
    }

    fn answered_with_headers(entries: &mut CallState, deployment: &Arc<Deployment>) {
        unary_success(entries, deployment);
        if let Some(attempt) = entries.attempt.as_mut() {
            attempt.provider_headers = pairs(&[
                ("request-id", "req_1"),
                ("anthropic-ratelimit-requests-remaining", "99"),
            ]);
        }
    }

    #[rstest]
    fn a_complete_answer_logs_the_provider_headers_as_the_anthropic_transform_keeps_them() {
        assert_eq!(
            built(answered_with_headers, false).unwrap()["hidden_params"]["additional_headers"],
            json!({
                "x_ratelimit_remaining_requests": 99,
                "llm_provider-request-id": "req_1",
                "llm_provider-anthropic-ratelimit-requests-remaining": "99",
            })
        );
    }

    fn disconnected(entries: &mut CallState, deployment: &Arc<Deployment>) {
        streamed(entries, deployment);
        entries.failure = Some(CallFailure::ClientDisconnected);
    }

    #[rstest]
    fn a_client_that_left_mid_stream_logs_what_was_served_and_pythons_499() {
        let document = built(disconnected, false).unwrap();
        assert_eq!(
            (
                &document["status"],
                &document["error_str"],
                &document["error_information"],
                &document["total_tokens"],
            ),
            (
                &json!("success"),
                &json!("Client disconnected the request"),
                &json!({
                    "error_code": "499",
                    "error_message": "Client disconnected the request",
                    "error_class": "ClientDisconnected",
                }),
                &json!(5),
            )
        );
    }

    fn unended(entries: &mut CallState, _: &Arc<Deployment>) {
        requested(entries, false);
    }

    #[rstest]
    fn a_call_that_has_not_ended_has_no_payload() {
        assert!(matches!(built(unended, false), Err(Unbuilt::NotEnded)));
    }
}
