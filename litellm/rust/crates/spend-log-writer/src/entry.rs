//! A queued request: its `LiteLLM_SpendLogs` row and the ledger charge it carries, sized the way
//! Python's `json.dumps` writes the row, which is what the queue and statement budgets count.

use litellm_callbacks_spend_log::{LedgerColumns, SpendLogRow};
use litellm_python_compat::{
    json,
    safe_json::{SafeJsonOptions, safe_json_structure},
};
use serde::Serialize;
use serde_json::{Map, Value};

const ACCOUNTING_PAYLOAD: &str = "accounting_payload";
const ACCOUNTING_DELIVERY_ID: &str = "accounting_delivery_id";
const ACCOUNTING_LOG_DISABLED: &str = "accounting_log_disabled";

/// The charge a request leaves for the ledger worker: its accounting payload and the delivery id
/// it settles under. A batch or video job that settles once has no delivery id and settles under
/// its request id.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Charge {
    #[serde(
        rename = "accounting_delivery_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub delivery_id: Option<String>,
    #[serde(rename = "accounting_payload")]
    pub payload: Map<String, Value>,
}

/// Why a row cannot be queued whole.
#[derive(Debug, thiserror::Error)]
pub enum EntryError {
    #[error("spend log row is not a JSON object the writer can read")]
    NotAnObject,
    #[error("spend log row has no string request_id and call_type")]
    MissingIds,
    #[error("accounting_payload or accounting_delivery_id is malformed")]
    InvalidCharge,
    #[error("spend log row does not fit LiteLLM_SpendLogs: {0}")]
    InvalidRow(String),
}

/// A row Python queued that the writer refused, with the accounting-only entry that still
/// charges the request when the row carried a valid charge.
#[derive(Debug)]
pub struct RejectedRow {
    pub request_id: Option<String>,
    pub error: EntryError,
    pub charge: Option<Box<QueuedSpendLog>>,
}

/// One queued request. `row` is `None` when only the charge is kept (`accounting_log_disabled`),
/// and `charge` is `None` off the ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct QueuedSpendLog {
    request_id: String,
    call_type: String,
    row: Option<SpendLogRow>,
    charge: Option<Charge>,
    bytes: usize,
}

/// The row as Python's queue held it: the row's columns, then the ledger keys.
#[derive(Serialize)]
#[serde(untagged)]
enum Wire<'a> {
    Logged {
        #[serde(flatten)]
        row: &'a SpendLogRow,
        #[serde(flatten)]
        charge: Option<&'a Charge>,
        #[serde(skip_serializing_if = "Option::is_none")]
        accounting_log_disabled: Option<bool>,
    },
    ChargeOnly {
        request_id: &'a str,
        call_type: &'a str,
        #[serde(flatten)]
        charge: Option<&'a Charge>,
        accounting_log_disabled: bool,
    },
}

fn string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text.clone()),
        _ => None,
    }
}

fn charge(fields: &mut Map<String, Value>) -> Result<Option<Charge>, EntryError> {
    let delivery_id = match fields.remove(ACCOUNTING_DELIVERY_ID) {
        None | Some(Value::Null) => None,
        Some(Value::String(delivery_id)) => Some(delivery_id),
        Some(_) => return Err(EntryError::InvalidCharge),
    };
    let payload = match fields.remove(ACCOUNTING_PAYLOAD) {
        None => return Ok(None),
        Some(Value::Object(payload)) => payload,
        Some(Value::String(text)) => match serde_json::from_str(&text) {
            Ok(Value::Object(payload)) => payload,
            _ => return Err(EntryError::InvalidCharge),
        },
        Some(_) => return Err(EntryError::InvalidCharge),
    };
    Ok(Some(Charge {
        delivery_id,
        payload,
    }))
}

impl QueuedSpendLog {
    /// A row built in Rust. `ledger` is its charge on the ledger path; with `logged` false only
    /// that charge is kept. The ledger's billable spend replaces the row's `spend`.
    pub fn new(
        mut row: SpendLogRow,
        ledger: Option<LedgerColumns>,
        logged: bool,
    ) -> Result<Self, EntryError> {
        let charge = match ledger {
            None => None,
            Some(LedgerColumns {
                spend,
                accounting_delivery_id,
                accounting_payload: Value::Object(payload),
            }) => {
                row.spend = Some(spend);
                Some(Charge {
                    delivery_id: accounting_delivery_id,
                    payload,
                })
            }
            Some(_) => return Err(EntryError::InvalidCharge),
        };
        Ok(Self {
            request_id: row.request_id.clone(),
            call_type: row.call_type.clone(),
            row: logged.then_some(row),
            charge,
            bytes: 0,
        }
        .measured())
    }

    /// A row Python queued as its `json.dumps` text. Non-finite numbers become null and NUL
    /// characters are dropped, as `safe_dumps` does before Postgres sees them. A row that is not a
    /// whole `LiteLLM_SpendLogs` row is refused, and keeps its charge when it carries one.
    pub fn from_python(text: &str) -> Result<Self, RejectedRow> {
        let refused = |request_id, error| RejectedRow {
            request_id,
            error,
            charge: None,
        };
        let Some(Value::Object(mut fields)) = json::loads(text).ok().and_then(|value| {
            json::to_json(&safe_json_structure(
                &value,
                SafeJsonOptions {
                    non_finite_as_null: true,
                    integer_limit: None,
                },
            ))
            .ok()
        }) else {
            return Err(refused(None, EntryError::NotAnObject));
        };
        let request_id = string(fields.get("request_id"));
        let (Some(id), Some(call_type)) = (request_id.clone(), string(fields.get("call_type")))
        else {
            return Err(refused(request_id, EntryError::MissingIds));
        };
        let log_disabled = fields.remove(ACCOUNTING_LOG_DISABLED) == Some(Value::Bool(true));
        let charge = charge(&mut fields).map_err(|error| refused(request_id.clone(), error))?;
        let entry = Self {
            request_id: id,
            call_type,
            row: None,
            charge,
            bytes: text.len(),
        };
        if log_disabled {
            return Ok(entry);
        }
        match serde_json::from_value::<SpendLogRow>(Value::Object(fields)) {
            Ok(row) => Ok(Self {
                row: Some(row),
                ..entry
            }),
            Err(error) => Err(RejectedRow {
                request_id,
                error: EntryError::InvalidRow(error.to_string()),
                charge: entry.accounting_only().map(Box::new),
            }),
        }
    }

    fn measured(mut self) -> Self {
        self.bytes = json::dumps_len(&self.wire()).unwrap_or(0);
        self
    }

    fn wire(&self) -> Wire<'_> {
        match &self.row {
            Some(row) => Wire::Logged {
                row,
                charge: self.charge.as_ref(),
                accounting_log_disabled: self.charge.as_ref().map(|_| false),
            },
            None => Wire::ChargeOnly {
                request_id: &self.request_id,
                call_type: &self.call_type,
                charge: self.charge.as_ref(),
                accounting_log_disabled: true,
            },
        }
    }

    /// The entry as the JSON object Python's queue holds, for a caller that hands queued rows
    /// back to Python.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.wire()).unwrap_or_default()
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn row(&self) -> Option<&SpendLogRow> {
        self.row.as_ref()
    }

    pub fn charge(&self) -> Option<&Charge> {
        self.charge.as_ref()
    }

    /// The bytes `json.dumps` writes for the row, which the queue and statement budgets count.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The entry reduced to its charge, marked as not logged, or `None` when it carries none.
    pub fn accounting_only(&self) -> Option<Self> {
        let charge = self.charge.clone()?;
        Some(
            Self {
                request_id: self.request_id.clone(),
                call_type: self.call_type.clone(),
                row: None,
                charge: Some(charge),
                bytes: 0,
            }
            .measured(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{EntryError, QueuedSpendLog};
    use crate::test_support::{full_row, python_row};

    #[rstest]
    #[case::logged(python_row("r", r#", "accounting_delivery_id": "d", "accounting_payload": {"spend": 1.0}, "accounting_log_disabled": false"#), true, true)]
    #[case::uncharged(python_row("r", ""), true, false)]
    #[case::charge_only(r#"{"request_id": "r", "call_type": "c", "accounting_payload": {"spend": 1.0}, "accounting_log_disabled": true}"#.to_owned(), false, true)]
    #[case::payload_as_text(
        python_row("r", r#", "accounting_payload": "{\"spend\": 1.0}""#),
        true,
        true
    )]
    fn reads_what_python_queued(#[case] text: String, #[case] logged: bool, #[case] charged: bool) {
        let entry = QueuedSpendLog::from_python(&text).expect("row is queued");

        assert_eq!(entry.row().is_some(), logged);
        assert_eq!(entry.charge().is_some(), charged);
        assert_eq!(entry.bytes(), text.len());
    }

    #[rstest]
    #[case::not_an_object("[1]", None, false)]
    #[case::lone_surrogate(r#"{"request_id": "r", "m": "\ud800"}"#, None, false)]
    #[case::null_request_id(r#"{"request_id": null, "call_type": "c"}"#, None, false)]
    #[case::payload_not_an_object(&python_row("r", r#", "accounting_payload": [1]"#), Some("r"), false)]
    #[case::sparse_row_keeps_its_charge(r#"{"request_id": "r", "call_type": "c", "spend": 1.0, "accounting_payload": {"spend": 1.0}}"#, Some("r"), true)]
    fn refuses_a_row_it_cannot_hold(
        #[case] text: &str,
        #[case] request_id: Option<&str>,
        #[case] keeps_charge: bool,
    ) {
        let rejected = QueuedSpendLog::from_python(text).expect_err("row is refused");

        assert_eq!(rejected.request_id.as_deref(), request_id);
        assert_eq!(rejected.charge.is_some(), keeps_charge);
        if keeps_charge {
            assert!(matches!(rejected.error, EntryError::InvalidRow(_)));
        }
    }

    #[rstest]
    #[case::charged(r#", "accounting_delivery_id": "d", "accounting_payload": {"spend": 1.0}, "accounting_log_disabled": false"#)]
    #[case::uncharged("")]
    fn writes_back_the_text_python_queued(#[case] ledger: &str) {
        let text = python_row("r", ledger);

        let entry = QueuedSpendLog::from_python(&text).expect("row is queued");

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&entry.to_json()).expect("json"),
            serde_json::from_str::<serde_json::Value>(&text).expect("json")
        );
    }

    fn python_len(entry: &QueuedSpendLog) -> usize {
        let value = serde_json::from_str(&entry.to_json()).expect("json");
        litellm_python_compat::json::dumps(&litellm_python_compat::json::from_json(value))
            .expect("dumps")
            .len()
    }

    #[rstest]
    fn measures_an_accounting_only_entry_like_json_dumps() {
        let entry = QueuedSpendLog::from_python(&python_row(
            "r",
            r#", "accounting_delivery_id": "d", "accounting_payload": {"spend": 1.0}"#,
        ))
        .expect("row is queued");

        let charge = entry.accounting_only().expect("charged");

        assert!(charge.row().is_none());
        assert_eq!(charge.bytes(), python_len(&charge));
    }

    #[rstest]
    fn a_rust_row_takes_the_ledger_spend() {
        let entry = QueuedSpendLog::new(
            full_row("r"),
            Some(litellm_callbacks_spend_log::LedgerColumns {
                spend: 2.5,
                accounting_delivery_id: Some("d".into()),
                accounting_payload: serde_json::json!({"spend": 2.5}),
            }),
            true,
        )
        .expect("entry builds");

        assert_eq!(entry.row().and_then(|row| row.spend), Some(2.5));
        assert_eq!(entry.bytes(), python_len(&entry));
    }
}
