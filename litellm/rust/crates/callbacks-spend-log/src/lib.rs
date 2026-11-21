//! The `LiteLLM_SpendLogs` row, and the accounting payload the spend logs ledger stores with it,
//! built in Rust for the proxy's ledger path.
//!
//! The Python glue (`litellm/proxy/spend_tracking/rust_spend_log_rows.py`) turns the logging
//! kwargs and response into one JSON input document, bounded with `safe_json_structure` and
//! carrying the request body and response only when the request's prompts are stored, plus the
//! few values that need Python-only state (the request id rule, the prompt storage decision, the
//! router registry, the cache instance, provider usage normalization). This crate parses that
//! document into a [`RowInput`] and applies the whole row policy:
//!
//! | Module | What it decides |
//! |---|---|
//! | [`row`] | the row's columns ([`SpendLogRow`]) and the metadata JSON |
//! | [`request`] | the stored request body, response and realtime messages |
//! | [`redaction`] | message redaction when message logging is turned off |
//! | [`compaction`] | bounding the summaries inside compaction blocks |
//! | [`privacy`] | credential masking, truncation, guardrail and error redaction |
//! | [`text`] | prompt-leak, IP address and raw model scrubbing in error text |
//! | [`stamps`] | end user, caller session, oauth, pod routing, spillover and classifier fields |
//! | [`accounting`] | the ledger's spend, delivery id and accounting payload ([`LedgerColumns`]) |
//! | [`consts`] | the field and key tables copied from Python, pinned by `tests/goldens/constants.json` |
//!
//! Equality with Python is pinned by `tests/goldens.rs`, which replays rows and ledger columns
//! recorded from Python's own `get_logging_payload` and `build_accounting_payload` by
//! `scripts/record_goldens.py`, and by the helper goldens each module replays inline. The
//! recorder runs the Python implementation that the ledger-off path still uses; once that path is
//! deleted, the committed goldens become fixed fixtures and the recorder goes with it.

mod accounting;
mod compaction;
mod consts;
mod input;
mod privacy;
mod py;
mod redaction;
mod request;
mod row;
mod stamps;
#[cfg(test)]
mod test_support;
mod text;

use std::sync::Arc;

use litellm_tracing::TokenRedactor;

pub use accounting::{
    AccountingSavings, AccountingWindow, LedgerColumns, LedgerInput, guardrail_information_cost,
};
pub use consts::VIDEO_STATUS_CALL_TYPES;
pub use input::RowInput;
pub use row::{SpendLogRow, SpendLogStatus, hash_token, is_sha256_hex, strip_bearer};

/// The default of `MAX_STRING_LENGTH_PROMPT_IN_DB`.
pub const DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB: i64 = 2048;

/// Why an input could not become a row: its JSON did not parse, or a field the glue always sends
/// (a timestamp, `now`, the fallback session) is missing or of the wrong type.
#[derive(Debug, thiserror::Error)]
pub enum RowError {
    #[error("spend log input field {0} is missing or malformed")]
    Input(&'static str),
    #[error(transparent)]
    Json(#[from] litellm_python_compat::Error),
}

/// The proxy settings the row policy reads, resolved in Python by the helpers `get_logging_payload`
/// itself uses and rebuilt only when one of them changes. The prompt storage decision is per
/// request, so it rides on the [`RowInput`] instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowConfig {
    /// `stores_latest_turn_only()`: `spend_logs_prompt_storage: latest_turn`.
    pub latest_turn_only: bool,
    /// `general_settings.redact_tokens_in_spend_logs is True`.
    pub redact_tokens: bool,
    /// `requester_ip_address_logging_disabled()`.
    pub drop_requester_ip_address: bool,
    /// `general_settings.missing_session_id == "omit"`.
    pub omit_missing_session_id: bool,
    /// `_get_max_string_length_prompt_in_db()`.
    pub max_string_length_prompt_in_db: i64,
    /// `litellm.disable_end_user_cost_tracking`.
    pub disable_end_user_cost_tracking: bool,
    /// `litellm.request_correlation_in_logs`.
    pub request_correlation_in_logs: bool,
    /// `litellm.turn_off_message_logging is True`.
    pub turn_off_message_logging: bool,
}

impl Default for RowConfig {
    fn default() -> Self {
        Self {
            latest_turn_only: false,
            redact_tokens: false,
            drop_requester_ip_address: false,
            omit_missing_session_id: false,
            max_string_length_prompt_in_db: DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB,
            disable_end_user_cost_tracking: false,
            request_correlation_in_logs: false,
            turn_off_message_logging: false,
        }
    }
}

/// Builds spend log rows under one [`RowConfig`], redacting tokens with the process's shared
/// [`TokenRedactor`].
pub struct SpendLogRowBuilder {
    config: RowConfig,
    tokens: Arc<TokenRedactor>,
}

impl SpendLogRowBuilder {
    pub fn new(config: RowConfig, tokens: Arc<TokenRedactor>) -> Self {
        Self { config, tokens }
    }

    /// The settings this builder was made with.
    pub fn config(&self) -> &RowConfig {
        &self.config
    }

    /// `get_logging_payload(kwargs, response_obj, start_time, end_time)`: the row for one request.
    pub fn build_row(&self, input: &RowInput) -> Result<SpendLogRow, RowError> {
        row::build(self, input)
    }
}
