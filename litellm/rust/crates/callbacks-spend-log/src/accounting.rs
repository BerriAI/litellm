//! The spend logs ledger's half of the row: the billable spend, the delivery id that dedupes the
//! accounting worker's writes, and the accounting payload (`build_accounting_payload` from
//! `litellm/accounting/snapshot.py`), which carries the guardrail share of spend and the
//! entities, windows and savings the worker debits.
//!
//! [`SpendLogRow::ledger_columns`] is the only entry point. It reads the row's accounting columns
//! and its metadata JSON, and returns the three ledger columns, so the caller merges them into the
//! row it already holds.

use litellm_python_compat::{Value, json, number, pydantic};
use serde::{Deserialize, Serialize};

use crate::consts::{
    ACCOUNTING_FIELDS, ACCOUNTING_METADATA_FIELDS, BATCH_COST_REQUEST_ID_SUFFIX,
    GUARDRAIL_ACCOUNTING_FIELDS, GUARDRAIL_METADATA_FIELDS, VIDEO_COST_REQUEST_ID_SUFFIX,
    VIDEO_STATUS_CALL_TYPES,
};
use crate::py::{
    Dict, SafeJsonOptions, as_str, big_to_f64, entries, get, get_or_none, lookup,
    safe_json_structure, text,
};
use crate::{RowError, SpendLogRow, SpendLogStatus};

const JSON_SAFE_INTEGER: i64 = (1 << 53) - 1;

/// One budget window the accounting worker debits: `accounting_window_keys(...)` in Python.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccountingWindow {
    pub entity_type: String,
    pub entity_id: String,
    pub window_duration: String,
    /// The window start as `datetime.isoformat()`.
    pub window_start: String,
    /// When a custom customer window resets, as `datetime.isoformat()`; absent for the others.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_at: Option<String>,
}

/// `SavingsSpend._asdict()`: the savings Python priced for the request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AccountingSavings {
    pub compression: f64,
    pub prompt_caching: f64,
    #[serde(default)]
    pub autorouter: f64,
    #[serde(default)]
    pub gateway_injected_caching: f64,
}

/// What the writer resolved for one row before it is stored on the ledger path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LedgerInput {
    /// `_billable_cost(response_cost, payload)`; `None`, zero and non-finite costs store `0.0`.
    pub billable_cost: Option<f64>,
    /// The organization the request resolved to; stored when non-empty.
    pub org_id: Option<String>,
    /// The team the request resolved to; stored when non-empty.
    pub team_id: Option<String>,
    /// A fresh delivery id, used unless the row settles a job's cost once.
    pub delivery_id: String,
    /// The entities the worker debits, in order: `api_key`, `user`, `end_user`, ... to their ids.
    pub entities: Vec<(String, Option<String>)>,
    pub model_access_groups: Vec<String>,
    pub windows: Vec<AccountingWindow>,
    pub savings: AccountingSavings,
}

/// The ledger columns of one row, which the writer merges into the row it stores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerColumns {
    /// The billable cost, `0.0` when there is none.
    pub spend: f64,
    /// `None` for a row that settles a batch or video cost once: its claim dedupes it instead.
    pub accounting_delivery_id: Option<String>,
    /// The accounting payload the worker projects, as the JSON document Postgres stores.
    pub accounting_payload: serde_json::Value,
}

fn entry_cost(entry: &Value) -> f64 {
    let Value::Dict(fields) = entry else {
        return 0.0;
    };
    let cost = match lookup(fields, "guardrail_cost") {
        None | Some(Value::None) => None,
        Some(found) => match pydantic::lax_float(found) {
            Some(cost) => Some(cost),
            None => return 0.0,
        },
    };
    let in_spend = match lookup(fields, "guardrail_cost_in_spend") {
        None => Some(true),
        Some(Value::None) => None,
        Some(found) => match pydantic::lax_bool(found) {
            Some(flag) => Some(flag),
            None => return 0.0,
        },
    };
    if in_spend == Some(false) {
        return 0.0;
    }
    match cost {
        Some(cost) if cost.is_finite() && cost > 0.0 => cost,
        _ => 0.0,
    }
}

/// `guardrail_information_cost(guardrail_information)`: the billable cost of the guardrail records.
pub fn guardrail_information_cost(information: &Value) -> f64 {
    match information {
        Value::None => 0.0,
        Value::List(items) => number::sum(items.iter().map(entry_cost)),
        other => entry_cost(other),
    }
}

/// `guardrail_share_of_spend(metadata, spend)`: the guardrail cost, capped at the spend.
fn guardrail_share_of_spend(metadata: &Value, spend: f64) -> f64 {
    let bounded_spend = spend.max(0.0);
    let cost = guardrail_information_cost(get_or_none(metadata, "guardrail_information"));
    if bounded_spend < cost {
        bounded_spend
    } else {
        cost
    }
}

/// `customer_margin_share_of_spend(metadata, spend)`: the recorded customer surcharge, capped at
/// the spend. A breakdown without a numeric `customer_margin_amount` gives `0.0`, and an integer
/// amount that is not capped stays an integer, as Python's `min` and `max` keep it.
fn customer_margin_share_of_spend(metadata: &Value, spend: f64) -> Value {
    let amount = match get(
        get_or_none(metadata, "cost_breakdown"),
        "customer_margin_amount",
    ) {
        Some(Value::Float(amount)) if amount.is_finite() => Value::Float(*amount),
        Some(Value::Int(amount)) => Value::Int(amount.clone()),
        _ => return Value::Float(0.0),
    };
    let amount_number = match &amount {
        Value::Int(amount) => big_to_f64(amount).unwrap_or(f64::INFINITY),
        Value::Float(amount) => *amount,
        _ => 0.0,
    };
    let floored = if amount_number >= 0.0 {
        amount
    } else {
        Value::Float(0.0)
    };
    let floored_number = amount_number.max(0.0);
    let cap = spend.max(0.0);
    if floored_number <= cap {
        floored
    } else {
        Value::Float(cap)
    }
}

/// The cache read and creation costs of the row's recorded cost breakdown, in its order.
fn cache_cost_breakdown(metadata: &Value) -> Value {
    let mut kept = Dict::default();
    for (key, value) in entries(get_or_none(metadata, "cost_breakdown")).unwrap_or_default() {
        if as_str(key).is_some_and(|name| matches!(name, "cache_read_cost" | "cache_creation_cost"))
        {
            kept.set_value(key.clone(), value.clone());
        }
    }
    kept.into_value()
}

fn accounting_guardrail_entry(entry: &[(Value, Value)]) -> Value {
    let mut kept = Dict::default();
    for (key, value) in entry {
        if as_str(key).is_some_and(|name| GUARDRAIL_ACCOUNTING_FIELDS.contains(&name)) {
            kept.set_value(key.clone(), value.clone());
        }
    }
    if let Some(response @ Value::Dict(_)) = lookup(entry, "guardrail_response")
        && let Some(tokens_saved) = get(response, "tokens_saved")
    {
        kept.set(
            "guardrail_response",
            Value::Dict(vec![(text("tokens_saved"), tokens_saved.clone())]),
        );
    }
    kept.into_value()
}

fn accounting_guardrail_entries(value: &Value) -> Value {
    let listed: &[Value] = match value {
        Value::Dict(_) => std::slice::from_ref(value),
        Value::List(items) => items,
        _ => return Value::None,
    };
    let Some(validated) = listed.iter().map(entries).collect::<Option<Vec<_>>>() else {
        return Value::None;
    };
    Value::List(
        validated
            .into_iter()
            .map(accounting_guardrail_entry)
            .collect(),
    )
}

fn optional_text(value: Option<&str>) -> Value {
    value.map_or(Value::None, text)
}

fn integer(value: i64) -> Value {
    Value::Int(value.into())
}

impl SpendLogRow {
    /// `_is_settle_once_cost_row(payload)`: a successful poll that carries a batch's or a video's
    /// cost, which is charged once by the first poll that sees it.
    pub fn settles_once(&self) -> bool {
        if self.status != SpendLogStatus::Success {
            return false;
        }
        match self.call_type.as_str() {
            "aretrieve_batch" => self.request_id.ends_with(BATCH_COST_REQUEST_ID_SUFFIX),
            kind if VIDEO_STATUS_CALL_TYPES.contains(&kind) => {
                self.request_id.ends_with(VIDEO_COST_REQUEST_ID_SUFFIX)
            }
            _ => false,
        }
    }

    /// The row's columns the accounting payload keeps, after the ledger set its spend and the
    /// resolved organization and team.
    fn accounting_columns(&self, spend: f64, ledger: &LedgerInput) -> Vec<(&'static str, Value)> {
        let resolved = |resolved: &Option<String>, stored: &str| {
            text(
                resolved
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .unwrap_or(stored),
            )
        };
        let columns = [
            ("request_id", text(&self.request_id)),
            ("call_type", text(&self.call_type)),
            ("api_key", text(&self.api_key)),
            ("startTime", text(&self.start_time)),
            ("model", text(&self.model)),
            ("user", text(&self.user)),
            ("team_id", resolved(&ledger.team_id, &self.team_id)),
            (
                "organization_id",
                resolved(&ledger.org_id, &self.organization_id),
            ),
            ("spend", Value::Float(spend)),
            ("prompt_tokens", integer(self.prompt_tokens)),
            ("completion_tokens", integer(self.completion_tokens)),
            ("request_tags", text(&self.request_tags)),
            ("end_user", text(&self.end_user)),
            ("model_group", text(&self.model_group)),
            ("agent_id", optional_text(self.agent_id.as_deref())),
            ("custom_llm_provider", text(&self.custom_llm_provider)),
            (
                "request_duration_ms",
                self.request_duration_ms.map_or(Value::None, integer),
            ),
            (
                "status",
                text(match self.status {
                    SpendLogStatus::Success => "success",
                    SpendLogStatus::Failure => "failure",
                }),
            ),
            (
                "mcp_namespaced_tool_name",
                optional_text(self.mcp_namespaced_tool_name.as_deref()),
            ),
        ];
        debug_assert!(
            columns
                .iter()
                .all(|(name, _)| ACCOUNTING_FIELDS.contains(name))
        );
        columns.into()
    }

    /// The row as the writer stores it on the ledger path: spend set to the billable cost, a
    /// delivery id unless the row settles a job's cost once, and the accounting payload built
    /// from the row with the organization and team the request resolved to.
    pub fn ledger_columns(&self, ledger: &LedgerInput) -> Result<LedgerColumns, RowError> {
        let spend = ledger
            .billable_cost
            .filter(|cost| cost.is_finite() && *cost != 0.0)
            .unwrap_or(0.0);
        let metadata = json::loads(&self.metadata)?;
        let mut accounting_metadata = Dict::default();
        for (key, value) in entries(&metadata).unwrap_or_default() {
            let Some(name) = as_str(key).filter(|name| ACCOUNTING_METADATA_FIELDS.contains(name))
            else {
                continue;
            };
            let stored = if GUARDRAIL_METADATA_FIELDS.contains(&name) {
                accounting_guardrail_entries(value)
            } else {
                value.clone()
            };
            accounting_metadata.set_value(key.clone(), stored);
        }
        accounting_metadata.set(
            "accounting_entities",
            Value::Dict(
                ledger
                    .entities
                    .iter()
                    .map(|(entity, id)| (text(entity), optional_text(id.as_deref())))
                    .collect(),
            ),
        );
        let savings = ledger.savings;
        accounting_metadata.set(
            "accounting_savings",
            Value::Dict(vec![
                (text("compression"), Value::Float(savings.compression)),
                (text("prompt_caching"), Value::Float(savings.prompt_caching)),
                (text("autorouter"), Value::Float(savings.autorouter)),
                (
                    text("gateway_injected_caching"),
                    Value::Float(savings.gateway_injected_caching),
                ),
            ]),
        );
        accounting_metadata.set(
            "accounting_guardrail_spend",
            Value::Float(guardrail_share_of_spend(&metadata, spend)),
        );
        accounting_metadata.set(
            "accounting_customer_margin_spend",
            customer_margin_share_of_spend(&metadata, spend),
        );
        accounting_metadata.set("cache_cost_breakdown", cache_cost_breakdown(&metadata));
        accounting_metadata.set(
            "accounting_model_access_groups",
            Value::List(
                ledger
                    .model_access_groups
                    .iter()
                    .map(|group| text(group))
                    .collect(),
            ),
        );
        accounting_metadata.set(
            "accounting_windows",
            Value::List(
                ledger
                    .windows
                    .iter()
                    .map(|window| {
                        let mut fields = vec![
                            (text("entity_type"), text(&window.entity_type)),
                            (text("entity_id"), text(&window.entity_id)),
                            (text("window_duration"), text(&window.window_duration)),
                            (text("window_start"), text(&window.window_start)),
                        ];
                        if let Some(reset_at) = &window.reset_at {
                            fields.push((text("reset_at"), text(reset_at)));
                        }
                        Value::Dict(fields)
                    })
                    .collect(),
            ),
        );
        let mut payload = Dict::default();
        for (name, value) in self.accounting_columns(spend, ledger) {
            payload.set(name, value);
        }
        payload.set("metadata", accounting_metadata.into_value());
        let accounting_payload = json::to_json(&safe_json_structure(
            &payload.into_value(),
            SafeJsonOptions {
                non_finite_as_null: true,
                integer_limit: Some(JSON_SAFE_INTEGER),
            },
        ))?;
        Ok(LedgerColumns {
            spend,
            accounting_delivery_id: (!self.settles_once()).then(|| ledger.delivery_id.clone()),
            accounting_payload,
        })
    }
}
