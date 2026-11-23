//! The calls the spend log callback makes back into the Python proxy, for state only Python
//! holds. Each request names the row it is about by its position in the hook's rows, so the host
//! reads that row's logging kwargs and the call's response from the call it is serving instead
//! of having them shipped through Rust.
//!
//! Awaited, because the row or the charge needs the answer: the key identity enrichment, the
//! search free tier counter and the accounting context. The search free tier discount is awaited
//! too, so the callbacks after this one log the discounted cost. Fire and forget, because Python only
//! schedules or records them: the post-write work, the after-charge work, the failed tracking
//! alert and the batch accounting outcome.

use litellm_callbacks::HostCall;
use litellm_callbacks_budgets::CustomerWindow;
use litellm_callbacks_spend_log::{AccountingSavings, AccountingWindow, SpendLogRow};
use litellm_redis_store::spend_counters::SpendCounter;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

/// `_enrich_failure_metadata_with_key_info`: the key's and team's identity from the auth cache
/// or DB, for a request whose key carried no user.
#[derive(Debug, Serialize)]
pub struct EnrichKeyIdentity {
    pub api_key: String,
    pub resolve_missing_key_identity: bool,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
}

/// The enriched `user_api_key_*` metadata, each `None` when still unknown.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct EnrichedKeyIdentity {
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub team_alias: Option<String>,
}

impl HostCall for EnrichKeyIdentity {
    const NAME: &'static str = "enrich_key_identity";
    type Reply = EnrichedKeyIdentity;
}

/// The built row's columns the accounting context reads: the savings pricing
/// (`_savings_or_zero`), the served model's access groups and the Responses chaining check.
#[derive(Debug, Serialize)]
pub struct AccountingRow {
    pub request_id: String,
    pub call_type: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub model_id: String,
    #[serde(rename = "endTime")]
    pub end_time: String,
    /// The row's metadata column, as JSON text.
    pub metadata: String,
}

impl From<&SpendLogRow> for AccountingRow {
    fn from(row: &SpendLogRow) -> Self {
        Self {
            request_id: row.request_id.clone(),
            call_type: row.call_type.clone(),
            model: row.model.clone(),
            custom_llm_provider: row.custom_llm_provider.clone(),
            model_id: row.model_id.clone(),
            end_time: row.end_time.clone(),
            metadata: row.metadata.clone(),
        }
    }
}

/// What a charge needs from the auth cache, the DB, the router and the pricing map. The customer
/// window counters and the warmed committed twins are resolved for every charge, even one whose
/// row did not build; the accounting windows (`accounting_window_keys`), the savings, the access
/// groups the served model debits and whether a Responses turn is replayed from its spend log
/// need the built row.
#[derive(Debug, Serialize)]
pub struct AccountingContext {
    /// The row's position in the hook's rows.
    pub position: usize,
    pub built: Option<AccountingRow>,
    /// The key as the ledger stores it (`hash_token` of a raw key).
    pub token: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    /// The row charges a cost, so the customer and budget windows are resolved.
    pub charges: bool,
    /// The request carried no committed twins and charges a positive cost, so the host warms
    /// them (`warm_committed_counter_keys`) when committed spend is tracked.
    pub warm_committed: bool,
}

fn counter_keys<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<SpendCounter>, D::Error> {
    Vec::<String>::deserialize(deserializer)?
        .iter()
        .map(|key| SpendCounter::parse(key).map_err(D::Error::custom))
        .collect()
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Accounting {
    pub windows: Vec<AccountingWindow>,
    /// `customer_hierarchy_ids_from_request(...).entities()`.
    pub customer_entities: Vec<(String, Option<String>)>,
    pub customer_windows: Vec<CustomerWindow>,
    #[serde(deserialize_with = "counter_keys")]
    pub committed_tracked: Vec<SpendCounter>,
    /// `None` when no row was built.
    pub savings: Option<AccountingSavings>,
    pub model_access_groups: Vec<String>,
    pub chains_through_db: bool,
}

impl HostCall for AccountingContext {
    const NAME: &'static str = "accounting_context";
    type Reply = Accounting;
}

/// The rest of `update_database` once a ledger row is recorded: the autorouter turn and baseline
/// accounting, the session spend total, tool usage (when logs are kept), the request error count,
/// the tool registry, and the guardrail usage of a job's claimed cost row, which skips the queue
/// whose flush counts it for every other row.
#[derive(Debug, Serialize)]
pub struct AfterWrite {
    pub position: usize,
    pub row: SpendLogRow,
    pub token: Option<String>,
    pub team_id: Option<String>,
    pub billable_cost: f64,
    pub spend_logs_kept: bool,
    pub claimed: bool,
    /// The claimed cost row's accounting payload, which tells the guardrail usage rollup that the
    /// ledger counts the row's usage units.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accounting_payload: Option<serde_json::Value>,
}

impl HostCall for AfterWrite {
    const NAME: &'static str = "after_write";
    type Reply = ();
}

/// The cost callback's work once a success is charged: `update_cache` (the key's projected soft
/// budget alert, the cached objects' spend and the proxy spend) and the customer spend alert.
#[derive(Debug, Serialize)]
pub struct AfterCharge {
    pub token: Option<String>,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub end_user_id: Option<String>,
    pub team_id: Option<String>,
    pub tags: Option<Vec<String>>,
    pub response_cost: f64,
    pub end_user_max_budget: Option<f64>,
}

impl HostCall for AfterCharge {
    const NAME: &'static str = "after_charge";
    type Reply = ();
}

/// `proxy_logging_obj.failed_tracking_alert`, raised when the cost callback cannot track a
/// request's cost.
#[derive(Debug, Serialize)]
pub struct FailedTrackingAlert {
    pub error_message: String,
    pub failing_model: String,
}

impl HostCall for FailedTrackingAlert {
    const NAME: &'static str = "failed_tracking_alert";
    type Reply = ();
}

/// The `BatchAccountingOutcome` CheckBatchCost reads to decide whether a batch's costing may be
/// retried. It lives in a context variable of the task that logs the batch, so only Python can
/// set it; the host call runs inline in that task.
#[derive(Debug, Serialize)]
pub struct BatchAccounting {
    pub write_started: bool,
    pub failed: bool,
}

impl HostCall for BatchAccounting {
    const NAME: &'static str = "batch_accounting";
    type Reply = ();
}

/// The search free tier's shared allowance counter in the internal usage cache's Redis, which
/// the Python path also counts in and which may not be the spend counter store's Redis. The reply
/// is the total after the increment, or `None` when the proxy has no Redis.
#[derive(Debug, Serialize)]
pub struct IncrementSearchFreeTier {
    pub key: String,
    pub units: i64,
    pub ttl: i64,
}

impl HostCall for IncrementSearchFreeTier {
    const NAME: &'static str = "increment_search_free_tier";
    type Reply = Option<f64>;
}

/// The success's charged cost written back to the live logging kwargs and
/// `standard_logging_object`, which only Python holds: `record_freed_cost` for a search free
/// tier draw, then `apply_callback_customer_cost_margin`, so the callbacks after this one log the
/// discounted and surcharged `response_cost` and `cost_breakdown`, as they do on main.
#[derive(Debug, Serialize)]
pub struct LoggedCost {
    /// The priced cost, before the search free tier.
    pub cost: f64,
    /// The search free tier discount, `None` when the allowance was not drawn from.
    pub discount: Option<f64>,
    pub customer_margin: Option<f64>,
}

impl HostCall for LoggedCost {
    const NAME: &'static str = "logged_cost";
    type Reply = ();
}
