//! The `spend_log_input` field, typed once at the boundary. The Python derivation
//! (`litellm/proxy/spend_tracking/rust_spend_log_inputs.py`) hands over only what Rust cannot read
//! from the typed contract fields: the proxy flags, the identity the request is charged to, a few
//! facts about the response, and each row's builder input. The row ids and costs come from the
//! `spend_log_id`, `billed_retry_attempts` and `cost` fields; every decision over these facts is
//! made in Rust ([`super::policy`]).
//!
//! Each row's builder input stays a JSON document until the row is built, because the enrichment
//! and the search free tier rewrite parts of it first; it is parsed into
//! [`litellm_callbacks_spend_log::RowInput`] once, at build time.

use litellm_callbacks_budgets::CustomerScopeSpend;
use litellm_callbacks_spend_log::RowConfig;
use litellm_redis_store::spend_counters::SpendCounter;
use litellm_spend_log_writer::Retention;
use serde::{Deserialize, Deserializer, de::Error as _};
use serde_json::Value;

/// `SpendLogRowConfig`, the row builder's settings, as the derivation sends them.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RowSettings {
    pub latest_turn_only: bool,
    pub redact_tokens: bool,
    pub drop_requester_ip_address: bool,
    pub omit_missing_session_id: bool,
    pub max_string_length_prompt_in_db: i64,
    pub disable_end_user_cost_tracking: bool,
    pub request_correlation_in_logs: bool,
    pub turn_off_message_logging: bool,
}

impl Default for RowSettings {
    fn default() -> Self {
        let config = RowConfig::default();
        Self {
            latest_turn_only: config.latest_turn_only,
            redact_tokens: config.redact_tokens,
            drop_requester_ip_address: config.drop_requester_ip_address,
            omit_missing_session_id: config.omit_missing_session_id,
            max_string_length_prompt_in_db: config.max_string_length_prompt_in_db,
            disable_end_user_cost_tracking: config.disable_end_user_cost_tracking,
            request_correlation_in_logs: config.request_correlation_in_logs,
            turn_off_message_logging: config.turn_off_message_logging,
        }
    }
}

impl From<&RowSettings> for RowConfig {
    fn from(settings: &RowSettings) -> Self {
        Self {
            latest_turn_only: settings.latest_turn_only,
            redact_tokens: settings.redact_tokens,
            drop_requester_ip_address: settings.drop_requester_ip_address,
            omit_missing_session_id: settings.omit_missing_session_id,
            max_string_length_prompt_in_db: settings.max_string_length_prompt_in_db,
            disable_end_user_cost_tracking: settings.disable_end_user_cost_tracking,
            request_correlation_in_logs: settings.request_correlation_in_logs,
            turn_off_message_logging: settings.turn_off_message_logging,
        }
    }
}

/// The proxy settings the cost callback and `update_database` read on each request.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub disable_spend_updates: bool,
    pub disable_spend_logs: bool,
    pub disable_error_logs: bool,
    pub disable_responses_id_security: bool,
    pub row_config: RowSettings,
    /// `litellm_proxy_budget_name` when a proxy `max_budget` is set.
    pub proxy_budget_name: Option<String>,
    /// `litellm.apply_search_free_tier`.
    pub apply_search_free_tier: bool,
    /// `BACKGROUND_INTERACTION_COST_POLLING_ENABLED`.
    pub background_interaction_cost_polling: bool,
    /// `general_settings.spend_logs_metadata_fields`, read on each request like the Python
    /// claim reads it, for the job cost rows the callback claims outside the queue.
    pub spend_logs_metadata_fields: Option<MetadataFields>,
}

/// `SpendLogsMetadataFields`, validated in Python to set exactly one of the two lists.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MetadataFields {
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
}

impl Settings {
    /// The metadata [`Retention`] a claimed job row is written with.
    pub(crate) fn retention(&self) -> Option<Retention> {
        self.spend_logs_metadata_fields
            .clone()
            .map(|fields| Retention {
                include: fields.include,
                exclude: fields.exclude,
            })
    }
}

/// Who a request is charged to, from the proxy-stamped `user_api_key_*` metadata (success) or
/// the auth object (failure).
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
#[serde(default)]
pub struct Identity {
    /// The key as logged: a hash, or a raw key the writer hashes. `Debug` hides it, since it can
    /// be the raw key.
    #[redact(with = "[REDACTED]")]
    pub token: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub end_user_id: Option<String>,
    pub key_alias: Option<String>,
    pub agent_id: Option<String>,
    /// The app the request was authenticated as, which alone makes a request tracked.
    pub app_id: Option<String>,
}

fn counter_keys<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<SpendCounter>>, D::Error> {
    Option::<Vec<String>>::deserialize(deserializer)?
        .map(|keys| {
            keys.iter()
                .map(|key| SpendCounter::parse(key).map_err(D::Error::custom))
                .collect()
        })
        .transpose()
}

/// One spend log row to write: the row builder's input (without its `request_id`, which comes
/// from the typed `spend_log_id` field) and what the row's counter step charges.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
#[serde(default)]
pub struct RowRequest {
    /// The row builder's input. `Debug` hides it whole, since its request body and metadata can
    /// carry credentials.
    #[redact(with = "[REDACTED]")]
    pub row: Value,
    /// `debitable_model_access_groups` over the served model, which the counter step charges.
    pub model_access_groups: Vec<String>,
    /// The entity counters whose committed twin the request carried, `None` when it carried
    /// none and the accounting context warms them.
    #[serde(deserialize_with = "counter_keys")]
    pub committed_tracked: Option<Vec<SpendCounter>>,
    /// `user_api_key_customer_hierarchy`'s scopes, settled first by the counter step.
    pub customer_hierarchy: Vec<CustomerScopeSpend>,
    /// `request_customer_cost_margin` over the row's request: the fraction its customer is
    /// surcharged, which reads the gateway's `cost_margin_config` and so is resolved by the
    /// derivation. `None` when the request runs for no customer.
    pub customer_margin: Option<f64>,
    /// The row's logging payload carries `cost_finalized`, so its cost takes no surcharge.
    pub cost_finalized: bool,
}

/// What only the response object can tell: a batch whose cost is not final yet
/// (`defers_batch_cost`), and an interaction create with no usage
/// (`_is_unbilled_interaction_response`) that may still be running in the background.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ResponseFacts {
    pub defers_batch_cost: bool,
    pub unbilled_interaction: bool,
    pub interaction_in_progress: bool,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Success {
    pub settings: Settings,
    pub identity: Identity,
    pub call_type: Option<String>,
    pub response: ResponseFacts,
    /// The logging payload's `cache_hit`, which the search free tier skips.
    pub cache_hit: bool,
    /// The kwargs' `response_cost`, read only when the call logged no payload (and so no `cost`).
    pub kwargs_response_cost: Option<f64>,
    /// `kwargs["stream"] is True`.
    pub stream: bool,
    /// The kwargs carry `complete_streaming_response`.
    pub stream_complete: bool,
    pub end_user_max_budget: Option<f64>,
    /// A CheckBatchCost costing waits on this request's `BatchAccountingOutcome`.
    pub batch_accounting: bool,
    /// The `window_reset_at` of each reservation entry, which the typed reservation lacks.
    pub window_resets: Vec<Option<String>>,
    pub row: RowRequest,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Failure {
    pub settings: Settings,
    pub identity: Identity,
    /// The failure is no `DBLookupDeadlineExceeded`, so the key identity may be looked up.
    pub enrich: bool,
    /// The failure is one the proxy writes rows for (error logs on, a trackable route).
    pub track: bool,
    pub nested_tool_call: bool,
    pub window_resets: Vec<Option<String>>,
    /// The final attempt's row, then one per billed retry attempt, in `billed_retry_attempts`
    /// order.
    pub rows: Vec<RowRequest>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpendLogInput {
    Success(Box<Success>),
    Failure(Box<Failure>),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Identity, RowRequest};

    #[test]
    fn identity_debug_hides_the_key() {
        let identity: Identity =
            serde_json::from_value(json!({"token": "credential-value", "team_id": "team"}))
                .unwrap();
        let printed = format!("{identity:?}");
        assert!(!printed.contains("credential-value"));
        assert!(printed.contains("[REDACTED]"));
    }

    #[test]
    fn row_request_debug_hides_the_row() {
        let request: RowRequest = serde_json::from_value(
            json!({"row": {"proxy_server_request": {"api_key": "credential-value"}}}),
        )
        .unwrap();
        let printed = format!("{request:?}");
        assert!(!printed.contains("credential-value"));
        assert!(printed.contains("[REDACTED]"));
    }
}
