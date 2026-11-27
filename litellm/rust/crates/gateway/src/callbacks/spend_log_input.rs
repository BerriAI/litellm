//! The `spend_log_input` field of a call the gateway served, worked out by `fields.rs` the way
//! `rust_spend_log_inputs.py` builds it for the proxy: who the call is charged to, what kind of
//! call it was, and the row builder's input document for its one spend log row.
//!
//! The fields the document leaves at their defaults, and why each default holds, are noted where
//! they are left out or set. It builds no logging payload yet, so the row reads the call from the
//! request's metadata and kwargs the way Python's builder does for a call that logged none.

use std::time::SystemTime;

use jiff::{Timestamp, tz::TimeZone};
use litellm_gateway_inference::{
    CallType, ServedCall,
    record::{Charged, unix},
};
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use serde::Serialize;

/// `cache_key` of a request the proxy's cache did not see.
const CACHE_OFF: &str = "Cache OFF";

/// `SpendLogInput::Success`, as the spend log callback reads it.
///
/// The fields it leaves out are read at their defaults, each held by what the gateway serves:
/// - `settings`: `Settings::default()`. Every key behind it (`disable_spend_logs`,
///   `disable_spend_updates`, `disable_error_logs`, `disable_responses_id_security`,
///   `store_prompts_in_spend_logs`, `turn_off_message_logging`, the row's redaction and length
///   settings, a proxy `max_budget`, `apply_search_free_tier`, `spend_logs_metadata_fields`) is
///   one the config types do not name, and `admit/checks/config.rs` sends any config that sets
///   one to Python. So no prompt is stored, the row is written, and nothing is charged to a proxy
///   budget. `BACKGROUND_INTERACTION_COST_POLLING_ENABLED` is a variable `Environment` does not
///   read, so admit cannot see it, but it only reaches background Responses jobs, which this
///   route never serves.
/// - `end_user_max_budget` and `window_resets`: none. `host/auth` declines any key with a team,
///   owner, end user, budget or access group, so only the master key or a bare key is served.
/// - `batch_accounting`: false. `/v1/messages` is the only route served, and it runs no batch.
/// - `response` and `kwargs_response_cost`: none. The gateway builds no logging payload yet, and
///   the cost comes from `Field::Cost`, which nothing on the native route fills yet.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SpendLogInput<'a> {
    Success {
        identity: Identity<'a>,
        call_type: &'static str,
        stream: bool,
        stream_complete: bool,
        row: RowRequest<'a>,
    },
}

/// Who the call is charged to, from the proxy-stamped `user_api_key_*` metadata.
#[derive(Serialize)]
struct Identity<'a> {
    token: Option<&'a str>,
    user_id: Option<&'a str>,
    key_alias: Option<&'a str>,
}

/// The call's one row. It names no model access group to charge, since `host/auth` declines a
/// key that has one.
#[derive(Serialize)]
struct RowRequest<'a> {
    row: Row<'a>,
}

/// A time of the call as the row builder reads a Python `datetime`: `raw` is `isoformat()` of
/// the proxy's naive local time, `utc` the same instant in UTC with its offset. The gateway runs
/// in UTC, as the proxy's pods do.
#[derive(Serialize)]
struct Moment {
    raw: String,
    utc: String,
}

impl Moment {
    fn of(at: SystemTime) -> Option<Self> {
        let at = Timestamp::try_from(at).ok()?.to_zoned(TimeZone::UTC);
        let fraction = match at.subsec_nanosecond() / 1_000 {
            0 => String::new(),
            micros => format!(".{micros:06}"),
        };
        let raw = format!("{}{fraction}", at.strftime("%Y-%m-%dT%H:%M:%S"));
        Some(Self {
            utc: format!("{raw}+00:00"),
            raw,
        })
    }
}

/// The `user_api_key_*` and routing metadata the proxy stamps on a request, as far as a call the
/// Rust gateway serves has any.
#[derive(Serialize)]
struct Metadata<'a> {
    user_api_key: Option<&'a str>,
    user_api_key_hash: Option<&'a str>,
    user_api_key_alias: Option<&'a str>,
    user_api_key_user_id: Option<&'a str>,
    user_api_key_request_route: &'static str,
    model_group: &'a str,
    model_info: ModelInfo<'a>,
    deployment: Option<&'a str>,
    status: &'static str,
}

#[derive(Serialize)]
struct ModelInfo<'a> {
    id: Option<&'a str>,
}

#[derive(Serialize)]
struct LitellmParams<'a> {
    api_base: Option<&'a str>,
    litellm_call_id: Option<&'a str>,
}

#[derive(Serialize)]
struct Kwargs<'a> {
    model: Option<&'a str>,
    custom_llm_provider: Option<&'a str>,
    litellm_call_id: Option<&'a str>,
    litellm_trace_id: &'a str,
}

/// The row builder's input (`spend_log_row_input`), without the `request_id` the callback reads
/// from `spend_log_id`.
#[derive(Serialize)]
struct Row<'a> {
    call_type: &'static str,
    cache_hit: bool,
    now: f64,
    fallback_session_id: String,
    start_time: Moment,
    end_time: Moment,
    completion_start_time: Moment,
    usage: Option<&'a ChatCompletionsUsage>,
    metadata: Metadata<'a>,
    litellm_params: LitellmParams<'a>,
    kwargs: Kwargs<'a>,
    cache_key: &'static str,
}

/// What a served call tells its spend log input.
pub(super) struct Served<'a> {
    pub(super) call_type: CallType,
    pub(super) route: &'static str,
    pub(super) charged: &'a Charged,
    pub(super) call_id: Option<&'a str>,
    /// The request's `litellm_trace_id`, which a row that names no session is grouped under.
    pub(super) trace_id: &'a str,
    pub(super) stream: bool,
    pub(super) start: SystemTime,
    pub(super) end: SystemTime,
    pub(super) provider: Option<&'a str>,
    pub(super) call: &'a ServedCall,
}

/// The `spend_log_input` document of `served`.
pub(super) fn success(served: &Served<'_>) -> Option<serde_json::Value> {
    let deployment = served.call.deployment.as_deref();
    let model = deployment.map(|deployment| deployment.params.model.as_str());
    let charged = served.charged;
    let now = SystemTime::now();
    let row = Row {
        call_type: served.call_type.into(),
        // The gateway serves no cache, so a call is never a cache hit.
        cache_hit: false,
        now: unix(now),
        fallback_session_id: uuid::Uuid::new_v4().to_string(),
        start_time: Moment::of(served.start)?,
        end_time: Moment::of(served.end)?,
        completion_start_time: Moment::of(served.call.first_event.unwrap_or(served.end))?,
        usage: served.call.usage.as_ref(),
        metadata: Metadata {
            user_api_key: charged.key_hash.as_deref(),
            user_api_key_hash: charged.key_hash.as_deref(),
            user_api_key_alias: charged.key_alias.as_deref(),
            user_api_key_user_id: charged.user_id.as_deref(),
            user_api_key_request_route: served.route,
            model_group: &served.call.model_group,
            model_info: ModelInfo {
                id: deployment.map(|deployment| deployment.id.as_str()),
            },
            deployment: model,
            status: "success",
        },
        litellm_params: LitellmParams {
            api_base: deployment.and_then(|deployment| deployment.params.api_base.as_deref()),
            litellm_call_id: served.call_id,
        },
        kwargs: Kwargs {
            model,
            custom_llm_provider: served.provider,
            litellm_call_id: served.call_id,
            litellm_trace_id: served.trace_id,
        },
        // And the proxy's cache never saw the request.
        cache_key: CACHE_OFF,
    };
    let input = SpendLogInput::Success {
        identity: Identity {
            token: charged.key_hash.as_deref(),
            user_id: charged.user_id.as_deref(),
            key_alias: charged.key_alias.as_deref(),
        },
        call_type: served.call_type.into(),
        stream: served.stream,
        stream_complete: served.stream,
        row: RowRequest { row },
    };
    serde_json::to_value(input).ok()
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use litellm_callbacks_spend_log::{RowConfig, RowInput, SpendLogRowBuilder, SpendLogStatus};
    use litellm_callbacks_spend_logger::SpendLogInput as Parsed;
    use litellm_llms_types::formats::chat_completions::PromptTokensDetails;
    use litellm_redaction::TokenRedactor;
    use litellm_router_core::{Deployment, DeploymentSpec};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn deployment() -> Arc<Deployment> {
        let spec = DeploymentSpec {
            model_name: "claude".into(),
            litellm_params: json!({"model": "anthropic/claude-haiku-5-5",
                "api_base": "https://api.anthropic.com", "api_key": "test-key"})
            .as_object()
            .cloned()
            .unwrap(),
            model_info: json!({"id": "claude-a"}).as_object().cloned().unwrap(),
            ..DeploymentSpec::default()
        };
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    fn at(seconds: u64, micros: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(seconds) + Duration::from_micros(micros)
    }

    fn call(first_event: Option<SystemTime>) -> ServedCall {
        ServedCall {
            model_group: "claude".to_owned(),
            deployment: Some(deployment()),
            response_id: Some("msg_1".to_owned()),
            usage: Some(ChatCompletionsUsage {
                prompt_tokens: 108,
                completion_tokens: 14,
                total_tokens: 122,
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens: Some(100),
                    ..PromptTokensDetails::default()
                }),
                ..ChatCompletionsUsage::default()
            }),
            first_event,
        }
    }

    fn key() -> Charged {
        Charged {
            key_hash: Some("a".repeat(64)),
            key_alias: Some("ci".to_owned()),
            user_id: None,
        }
    }

    fn input(charged: &Charged, call: &ServedCall, stream: bool) -> Value {
        success(&Served {
            call_type: CallType::AnthropicMessages,
            route: "/v1/messages",
            charged,
            call_id: Some("call-1"),
            trace_id: "trace-1",
            stream,
            start: at(1_795_000_000, 250_000),
            end: at(1_795_000_002, 0),
            provider: Some("anthropic"),
            call,
        })
        .unwrap()
    }

    /// The row the spend log callback builds from `input`, under the id `spend_log_id` names.
    fn row(input: &Value) -> litellm_callbacks_spend_log::SpendLogRow {
        let mut document = input["row"]["row"].clone();
        document["request_id"] = json!("msg_1");
        let builder = SpendLogRowBuilder::new(
            RowConfig::default(),
            Arc::new(TokenRedactor::new(0, vec![])),
        );
        builder
            .build_row(
                &RowInput::from_value(&litellm_python_compat::json::from_json(document)).unwrap(),
            )
            .unwrap()
    }

    #[rstest]
    fn the_spend_log_callback_reads_a_served_calls_input_as_a_success() {
        let charged = key();
        let parsed: Parsed = serde_json::from_value(input(&charged, &call(None), false)).unwrap();
        let Parsed::Success(success) = parsed else {
            panic!("a served call is a success")
        };
        assert_eq!(
            (
                success.identity.token,
                success.identity.key_alias,
                success.call_type.as_deref(),
                success.stream,
                success.settings,
            ),
            (
                Some("a".repeat(64)),
                Some("ci".to_owned()),
                Some("anthropic_messages"),
                false,
                litellm_callbacks_spend_logger::hook_input::Settings::default(),
            )
        );
    }

    #[rstest]
    #[case::a_complete_answer(None, "2026-11-18T11:06:42+00:00")]
    #[case::a_stream(Some(at(1_795_000_001, 5)), "2026-11-18T11:06:41.000005+00:00")]
    fn a_served_calls_row_names_its_key_deployment_tokens_and_times(
        #[case] first_event: Option<SystemTime>,
        #[case] completion_start: &str,
    ) {
        let charged = key();
        let row = row(&input(&charged, &call(first_event), first_event.is_some()));
        assert_eq!(
            (
                (
                    row.request_id.as_str(),
                    row.call_type.as_str(),
                    row.api_key.as_str(),
                    row.model.as_str(),
                    row.model_group.as_str(),
                    row.model_id.as_str(),
                    row.api_base.as_str(),
                    row.custom_llm_provider.as_str(),
                ),
                (row.prompt_tokens, row.completion_tokens, row.total_tokens),
                row.start_time.as_str(),
                row.end_time.as_str(),
                row.completion_start_time.as_deref(),
                row.request_duration_ms,
                row.litellm_call_id.as_deref(),
                row.session_id.as_deref(),
                row.status,
            ),
            (
                (
                    "msg_1",
                    "anthropic_messages",
                    "a".repeat(64).as_str(),
                    "anthropic/claude-haiku-5-5",
                    "claude",
                    "claude-a",
                    "https://api.anthropic.com",
                    "anthropic",
                ),
                (108, 14, 122),
                "2026-11-18T11:06:40.250000+00:00",
                "2026-11-18T11:06:42+00:00",
                Some(completion_start),
                Some(1750),
                Some("call-1"),
                Some("trace-1"),
                SpendLogStatus::Success,
            )
        );
    }

    #[rstest]
    fn the_master_keys_row_is_logged_under_pythons_alias_for_it() {
        let charged = Charged {
            key_hash: Some("litellm_proxy_master_key".to_owned()),
            key_alias: None,
            user_id: Some("default_user_id".to_owned()),
        };
        let row = row(&input(&charged, &call(None), false));
        assert_eq!(
            (row.api_key.as_str(), row.user.as_str()),
            ("litellm_proxy_master_key", "default_user_id")
        );
    }
}
