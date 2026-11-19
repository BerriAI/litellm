//! A served call as the logging callbacks receive it, folded from what the route observed.

use std::{num::NonZeroUsize, sync::Arc};

use futures_util::future::BoxFuture;
use litellm_host::{
    interceptors::{Cost, ExecutionFacts, RawResponse, ResultSource},
    lifecycle::{CallEvent, ExecutionEvent, Timing},
    observation::{ObservationSender, observation_channel},
};
use litellm_standard_logging::{
    Caller, GuardrailStatus, HiddenParams, Metadata, ModelInformation, StandardLoggingPayload,
    Status,
};
use serde_json::{Map, Value};
use tokio::sync::mpsc::Receiver;

const QUEUED_EVENTS: NonZeroUsize = NonZeroUsize::new(16).unwrap();

/// Where the gateway sends each call it logs.
pub trait CallLog: Send + Sync + 'static {
    fn record(&self, payload: StandardLoggingPayload) -> BoxFuture<'static, ()>;
}

/// What the route knows about a call before it runs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Request {
    pub(crate) call_id: Option<String>,
    pub(crate) call_type: &'static str,
    pub(crate) model_group: String,
    pub(crate) model: String,
    pub(crate) api_base: Option<String>,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) messages: Value,
    pub(crate) model_parameters: Map<String, Value>,
    pub(crate) user_agent: Option<String>,
    pub(crate) caller: Caller,
}

pub(crate) struct Observed {
    log: Arc<dyn CallLog>,
    events: Receiver<CallEvent>,
}

pub(crate) fn observe(log: Arc<dyn CallLog>) -> (ObservationSender, Observed) {
    let (sender, events) = observation_channel(QUEUED_EVENTS);
    (sender, Observed { log, events })
}

impl Observed {
    /// Logs `request` once its call has ended, which the route's last event says.
    pub(crate) async fn record(mut self, request: Request) {
        let mut events = Vec::new();
        while let Some(event) = self.events.recv().await {
            let ended = matches!(
                event,
                CallEvent::Succeeded { .. }
                    | CallEvent::Failed { .. }
                    | CallEvent::Cancelled { .. }
            );
            events.push(event);
            if ended {
                break;
            }
        }
        if let Some(payload) = payload(request, &events) {
            self.log.record(payload).await;
        }
    }
}

fn succeeded(events: &[CallEvent]) -> Option<Timing> {
    events.iter().find_map(|event| match event {
        CallEvent::Succeeded { timing, .. } => Some(*timing),
        _ => None,
    })
}

fn provider_response(events: &[CallEvent]) -> Option<&RawResponse> {
    events.iter().find_map(|event| match event {
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => Some(raw),
        _ => None,
    })
}

fn execution_facts(events: &[CallEvent]) -> Option<&ExecutionFacts> {
    events.iter().find_map(|event| match event {
        CallEvent::Execution(ExecutionEvent::ResultReady { facts }) => Some(facts),
        _ => None,
    })
}

fn tokens(usage: Option<&Value>, name: &str) -> u64 {
    usage
        .and_then(|usage| usage.get(name))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// The payload of a call that succeeded, and nothing for one that failed or was cancelled.
pub(crate) fn payload(request: Request, events: &[CallEvent]) -> Option<StandardLoggingPayload> {
    let timing = succeeded(events)?;
    let facts = execution_facts(events);
    let response: Value = provider_response(events)
        .and_then(|raw| serde_json::from_str(&raw.body).ok())
        .unwrap_or(Value::Null);
    let usage = response.get("usage");
    let prompt_tokens = tokens(usage, "input_tokens")
        + tokens(usage, "cache_read_input_tokens")
        + tokens(usage, "cache_creation_input_tokens");
    let completion_tokens = tokens(usage, "output_tokens");
    let reported_cost = facts.and_then(|facts| match &facts.cost {
        Cost::Reported { amount } => amount.as_f64(),
        Cost::CacheHit | Cost::Deferred => None,
    });
    let cache_key = facts.and_then(|facts| match &facts.source {
        ResultSource::Cache { key } => Some(key.clone()),
        ResultSource::Provider | ResultSource::Host => None,
    });
    let response_id = response.get("id").and_then(Value::as_str);
    Some(StandardLoggingPayload {
        id: response_id
            .or(request.call_id.as_deref())
            .unwrap_or_default()
            .to_owned(),
        trace_id: request.call_id.clone().unwrap_or_default(),
        litellm_call_id: request.call_id,
        session_id: String::new(),
        call_type: request.call_type.to_owned(),
        cache_hit: cache_key.is_some().then_some(true),
        cache_key,
        stream: None,
        status: Status::Success,
        guardrail_status: GuardrailStatus::NotRun,
        custom_llm_provider: facts
            .map(|facts| facts.provider.provider.clone())
            .or(request.custom_llm_provider),
        saved_cache_cost: 0.0,
        start_time: timing.start_time,
        end_time: timing.end_time,
        completion_start_time: timing.end_time,
        response_time: timing.end_time - timing.start_time,
        model_map_information: ModelInformation {
            model_map_key: request.model.clone(),
            model_map_value: None,
        },
        model: request.model,
        caller: request.caller,
        requester_ip_address: None,
        user_agent: request.user_agent,
        metadata: Metadata {
            usage_object: usage.and_then(Value::as_object).cloned(),
            ..Metadata::default()
        },
        cost_finalized: reported_cost.is_some(),
        response_cost: reported_cost.unwrap_or(0.0),
        cost_breakdown: None,
        autorouter_savings: None,
        autorouter_savings_estimate: None,
        autorouter_baseline_observation: None,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens,
        completion_tokens,
        request_tags: Vec::new(),
        request_model_access_groups: Vec::new(),
        end_user: None,
        api_base: request.api_base.unwrap_or_default(),
        model_group: Some(request.model_group),
        model_id: None,
        messages: request.messages,
        response,
        model_parameters: request.model_parameters,
        hidden_params: HiddenParams {
            response_cost: reported_cost,
            ..HiddenParams::default()
        },
        error_str: None,
        error_information: None,
        response_cost_failure_debug_info: None,
        zero_cost_diagnostic: None,
        guardrail_information: None,
        standard_built_in_tools_params: None,
    })
}

#[cfg(test)]
mod tests {
    use litellm_host::{interceptors::ProviderIdentity, lifecycle::FailureOrigin};
    use litellm_standard_logging::Key;
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    const TIMING: Timing = Timing {
        start_time: 1_767_268_800.0,
        end_time: 1_767_268_802.5,
    };

    #[fixture]
    fn request() -> Request {
        Request {
            call_id: Some("call-1".to_owned()),
            call_type: "anthropic_messages",
            model_group: "claude".to_owned(),
            model: "anthropic/claude-haiku-5-5".to_owned(),
            api_base: Some("https://api.anthropic.com".to_owned()),
            custom_llm_provider: None,
            messages: json!([{"role": "user", "content": "Hey"}]),
            model_parameters: json!({"max_tokens": 32}).as_object().cloned().unwrap(),
            user_agent: Some("curl/8.7.1".to_owned()),
            caller: Caller {
                key: Key {
                    hash: Some("key-1".to_owned()),
                    ..Key::default()
                },
                ..Caller::default()
            },
        }
    }

    fn body() -> Value {
        json!({
            "id": "msg_1",
            "type": "message",
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 5,
            },
        })
    }

    fn ready(source: ResultSource, cost: Cost) -> CallEvent {
        CallEvent::Execution(ExecutionEvent::ResultReady {
            facts: ExecutionFacts {
                provider: ProviderIdentity {
                    model: "claude-haiku-5-5".to_owned(),
                    provider: "anthropic".to_owned(),
                },
                source,
                cost,
            },
        })
    }

    fn served(source: ResultSource, cost: Cost) -> Vec<CallEvent> {
        vec![
            CallEvent::Started {
                start_time: TIMING.start_time,
            },
            CallEvent::Execution(ExecutionEvent::ProviderResponseReceived {
                raw: RawResponse::new(body().to_string()),
            }),
            ready(source, cost),
            CallEvent::Succeeded {
                timing: TIMING,
                response: (),
            },
        ]
    }

    #[rstest]
    fn a_served_call_is_logged_with_what_the_provider_answered(request: Request) {
        let logged = payload(
            request.clone(),
            &served(ResultSource::Provider, Cost::Deferred),
        )
        .unwrap();
        assert_eq!(
            logged,
            StandardLoggingPayload {
                id: "msg_1".to_owned(),
                litellm_call_id: Some("call-1".to_owned()),
                trace_id: "call-1".to_owned(),
                session_id: String::new(),
                call_type: "anthropic_messages".to_owned(),
                cache_hit: None,
                cache_key: None,
                stream: None,
                status: Status::Success,
                guardrail_status: GuardrailStatus::NotRun,
                custom_llm_provider: Some("anthropic".to_owned()),
                saved_cache_cost: 0.0,
                start_time: 1_767_268_800.0,
                end_time: 1_767_268_802.5,
                completion_start_time: 1_767_268_802.5,
                response_time: 2.5,
                model: "anthropic/claude-haiku-5-5".to_owned(),
                caller: request.caller,
                requester_ip_address: None,
                user_agent: Some("curl/8.7.1".to_owned()),
                metadata: Metadata {
                    usage_object: body()["usage"].as_object().cloned(),
                    ..Metadata::default()
                },
                cost_finalized: false,
                response_cost: 0.0,
                cost_breakdown: None,
                autorouter_savings: None,
                autorouter_savings_estimate: None,
                autorouter_baseline_observation: None,
                total_tokens: 20,
                prompt_tokens: 15,
                completion_tokens: 5,
                request_tags: Vec::new(),
                request_model_access_groups: Vec::new(),
                end_user: None,
                api_base: "https://api.anthropic.com".to_owned(),
                model_group: Some("claude".to_owned()),
                model_id: None,
                messages: request.messages,
                response: body(),
                model_parameters: request.model_parameters,
                hidden_params: HiddenParams::default(),
                model_map_information: ModelInformation {
                    model_map_key: "anthropic/claude-haiku-5-5".to_owned(),
                    model_map_value: None,
                },
                error_str: None,
                error_information: None,
                response_cost_failure_debug_info: None,
                zero_cost_diagnostic: None,
                guardrail_information: None,
                standard_built_in_tools_params: None,
            }
        );
    }

    #[rstest]
    fn a_cost_the_provider_reports_is_final(request: Request) {
        let amount = serde_json::Number::from_f64(0.25).unwrap();
        let logged = payload(
            request,
            &served(ResultSource::Provider, Cost::Reported { amount }),
        )
        .unwrap();
        assert_eq!(
            (
                logged.cost_finalized,
                logged.response_cost,
                logged.hidden_params.response_cost
            ),
            (true, 0.25, Some(0.25))
        );
    }

    #[rstest]
    fn a_call_served_from_the_cache_names_its_key(request: Request) {
        let source = ResultSource::Cache {
            key: "cache-key-1".to_owned(),
        };
        let logged = payload(request, &served(source, Cost::CacheHit)).unwrap();
        assert_eq!(
            (logged.cache_hit, logged.cache_key),
            (Some(true), Some("cache-key-1".to_owned()))
        );
    }

    #[rstest]
    #[case::failed(CallEvent::Failed { timing: TIMING, origin: FailureOrigin::Call, error: () })]
    #[case::cancelled(CallEvent::Cancelled { timing: TIMING })]
    fn a_call_that_did_not_succeed_is_not_logged(request: Request, #[case] end: CallEvent) {
        assert_eq!(payload(request, &[end]), None);
    }

    #[derive(Default)]
    struct Recorded(parking_lot::Mutex<Vec<StandardLoggingPayload>>);

    impl CallLog for Recorded {
        fn record(&self, payload: StandardLoggingPayload) -> BoxFuture<'static, ()> {
            self.0.lock().push(payload);
            Box::pin(async {})
        }
    }

    #[rstest]
    #[tokio::test]
    async fn the_log_gets_the_call_once_its_last_event_arrives(request: Request) {
        let recorded = Arc::new(Recorded::default());
        let (sender, observed) = observe(recorded.clone());
        for event in served(ResultSource::Provider, Cost::Deferred) {
            sender.emit(event);
        }
        observed.record(request).await;
        let ids: Vec<String> = recorded
            .0
            .lock()
            .iter()
            .map(|payload| payload.id.clone())
            .collect();
        assert_eq!(ids, ["msg_1"]);
    }
}
