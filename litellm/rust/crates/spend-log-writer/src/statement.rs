//! Turning a queued request into what the repository writes: the typed `LiteLLM_SpendLogs` record
//! with its metadata cut to the retained fields, and its `LiteLLM_SpendLogDelivery` row.

use chrono::{DateTime, NaiveDateTime};
use litellm_callbacks_spend_log::{SpendLogRow, SpendLogStatus};
use litellm_python_compat::Value;
use litellm_python_compat::{
    json,
    safe_json::{SafeJsonOptions, safe_json_structure},
};
use litellm_repositories::{JsonText, SpendLogDelivery, SpendLogRecord, SpendLogWrite};

use crate::QueuedSpendLog;

/// `SPEND_LOGS_METADATA_ALWAYS_KEPT_FIELDS`: metadata fields no retention setting drops.
const ALWAYS_KEPT: [&str; 3] = [
    "cold_storage_object_key",
    "litellm_previous_response_ids",
    "status",
];

/// Which metadata fields a logged row keeps (`general_settings.spend_logs_metadata_fields`),
/// validated in Python to set exactly one of the two lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retention {
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
}

impl Retention {
    /// `SpendLogsMetadataFields.keeps(name)`.
    pub fn keeps(&self, name: &str) -> bool {
        if ALWAYS_KEPT.contains(&name) {
            return true;
        }
        match (&self.include, &self.exclude) {
            (Some(include), _) => include.iter().any(|field| field == name),
            (None, exclude) => !exclude.iter().flatten().any(|field| field == name),
        }
    }
}

/// A queued row the `LiteLLM_SpendLogs` table cannot hold, refused before it reaches Postgres.
#[derive(Debug, thiserror::Error)]
pub enum RowError {
    #[error("Spend log column {0} is not valid JSON")]
    InvalidJson(&'static str),
    #[error("Spend log column {0} is not an ISO 8601 timestamp")]
    InvalidTimestamp(&'static str),
    #[error("Spend log column {0} is out of range")]
    OutOfRange(&'static str),
    #[error("Spend log column spend is null")]
    NullSpend,
}

fn timestamp(column: &'static str, text: &str) -> Result<NaiveDateTime, RowError> {
    DateTime::parse_from_rfc3339(text)
        .map(|timestamp| timestamp.naive_utc())
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f"))
        .map_err(|_| RowError::InvalidTimestamp(column))
}

fn int(column: &'static str, value: i64) -> Result<i32, RowError> {
    i32::try_from(value).map_err(|_| RowError::OutOfRange(column))
}

/// The JSON text of a `jsonb` column as the value Postgres stores: non-finite numbers become
/// null, NUL characters are dropped, nesting past `safe_dumps`'s depth is marked and integers
/// keep every digit, as the Python insert's `safe_dumps` does. `retention` cuts the fields of an
/// object to the retained ones.
fn json_column(
    column: &'static str,
    text: &str,
    retention: Option<&Retention>,
) -> Result<JsonText, RowError> {
    json::loads(text)
        .map(|value| {
            safe_json_structure(
                &value,
                SafeJsonOptions {
                    non_finite_as_null: true,
                    integer_limit: None,
                },
            )
        })
        .map(|value| retained(value, retention))
        .and_then(|value| json::dumps_compact(&value))
        .ok()
        .and_then(|text| JsonText::new(text).ok())
        .ok_or(RowError::InvalidJson(column))
}

fn retained(value: Value, retention: Option<&Retention>) -> Value {
    match (value, retention) {
        (Value::Dict(fields), Some(retention)) => Value::Dict(
            fields
                .into_iter()
                .filter(|(name, _)| matches!(name, Value::Str(name) if retention.keeps(name)))
                .collect(),
        ),
        (value, _) => value,
    }
}

/// A text column as the Python insert's `safe_dumps` writes it, with NUL characters dropped,
/// since Postgres cannot store them in `text`.
fn text(value: &str) -> String {
    value.replace('\0', "")
}

fn optional_text(value: Option<&str>) -> Option<String> {
    value.map(text)
}

fn record(
    row: &SpendLogRow,
    retention: Option<&Retention>,
    now: NaiveDateTime,
) -> Result<SpendLogRecord, RowError> {
    Ok(SpendLogRecord {
        request_id: text(&row.request_id),
        call_type: text(&row.call_type),
        api_key: text(&row.api_key),
        spend: row.spend.ok_or(RowError::NullSpend)?,
        total_tokens: int("total_tokens", row.total_tokens)?,
        prompt_tokens: int("prompt_tokens", row.prompt_tokens)?,
        completion_tokens: int("completion_tokens", row.completion_tokens)?,
        start_time: timestamp("startTime", &row.start_time)?,
        end_time: timestamp("endTime", &row.end_time)?,
        request_duration_ms: row
            .request_duration_ms
            .map(|value| int("request_duration_ms", value))
            .transpose()?,
        completion_start_time: row
            .completion_start_time
            .as_deref()
            .map(|text| timestamp("completionStartTime", text))
            .transpose()?,
        model: text(&row.model),
        model_id: Some(text(&row.model_id)),
        model_group: Some(text(&row.model_group)),
        custom_llm_provider: Some(text(&row.custom_llm_provider)),
        api_base: Some(text(&row.api_base)),
        user: Some(text(&row.user)),
        metadata: Some(json_column("metadata", &row.metadata, retention)?),
        cache_hit: Some(text(&row.cache_hit)),
        cache_key: Some(text(&row.cache_key)),
        request_tags: Some(json_column("request_tags", &row.request_tags, None)?),
        team_id: Some(text(&row.team_id)),
        organization_id: Some(text(&row.organization_id)),
        end_user: Some(text(&row.end_user)),
        requester_ip_address: optional_text(row.requester_ip_address.as_deref()),
        messages: Some(json_column("messages", &row.messages, None)?),
        response: Some(json_column("response", &row.response, None)?),
        session_id: optional_text(row.session_id.as_deref()),
        status: Some(
            match row.status {
                SpendLogStatus::Success => "success",
                SpendLogStatus::Failure => "failure",
            }
            .to_owned(),
        ),
        mcp_namespaced_tool_name: optional_text(row.mcp_namespaced_tool_name.as_deref()),
        billing_agent_id: optional_text(row.billing_agent_id.as_deref()),
        agent_id: optional_text(row.agent_id.as_deref()),
        proxy_server_request: Some(json_column(
            "proxy_server_request",
            &row.proxy_server_request,
            None,
        )?),
        litellm_call_id: optional_text(row.litellm_call_id.as_deref()),
        created_at: now,
        updated_at: now,
    })
}

/// The delivery a queued request's charge writes, or `None` when it carries no charge. A
/// settle-once job's delivery takes its request id.
pub fn delivery(entry: &QueuedSpendLog) -> Option<SpendLogDelivery> {
    let charge = entry.charge()?;
    Some(SpendLogDelivery {
        delivery_id: charge
            .delivery_id
            .clone()
            .unwrap_or_else(|| entry.request_id().to_owned()),
        request_id: entry.request_id().to_owned(),
        accounting_payload: charge.payload.clone(),
        is_batch: charge.delivery_id.is_none(),
    })
}

/// What one queued request writes, stamped with `now` as its `created_at` and `updated_at`.
pub fn spend_log_write(
    entry: &QueuedSpendLog,
    retention: Option<&Retention>,
    now: NaiveDateTime,
) -> Result<SpendLogWrite, RowError> {
    Ok(SpendLogWrite {
        log: entry
            .row()
            .map(|row| record(row, retention, now))
            .transpose()?,
        delivery: delivery(entry),
    })
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDateTime;
    use rstest::rstest;
    use serde_json::{Value, json};

    use litellm_repositories::JsonText;

    use super::{Retention, RowError, spend_log_write};
    use litellm_callbacks_spend_log::SpendLogRow;

    use crate::{
        QueuedSpendLog,
        test_support::{full_row, python_row},
    };

    fn now() -> NaiveDateTime {
        NaiveDateTime::parse_from_str("2026-11-02T12:00:05", "%Y-%m-%dT%H:%M:%S").expect("now")
    }

    fn stored(column: JsonText) -> Value {
        serde_json::from_str(column.get()).expect("column holds JSON")
    }

    fn entry(fields: &str) -> QueuedSpendLog {
        QueuedSpendLog::from_python(&python_row("r", fields)).expect("row is queued")
    }

    #[rstest]
    #[case::include(Retention { include: Some(vec!["a".into()]), exclude: None }, json!({"a": 1, "status": "success"}))]
    #[case::exclude(Retention { include: None, exclude: Some(vec!["a".into()]) }, json!({"b": 2, "status": "success"}))]
    fn retains_configured_metadata_fields(#[case] retention: Retention, #[case] expected: Value) {
        let write = spend_log_write(
            &entry(r#", "metadata": "{\"a\": 1, \"b\": 2, \"status\": \"success\"}""#),
            Some(&retention),
            now(),
        )
        .expect("writes");

        assert_eq!(
            write.log.and_then(|log| log.metadata).map(stored),
            Some(expected)
        );
    }

    #[rstest]
    #[case::not_an_object(r#", "metadata": "[1]""#, json!([1]))]
    #[case::nan_and_nul(r#", "metadata": "{\"spend\": NaN, \"a\\u0000\": \"x\\u0000y\"}""#, json!({"spend": null, "a": "xy"}))]
    fn stores_metadata_as_safe_dumps_writes_it(#[case] fields: &str, #[case] expected: Value) {
        let write =
            spend_log_write(&entry(fields), Some(&Retention::default()), now()).expect("writes");

        assert_eq!(
            write.log.and_then(|log| log.metadata).map(stored),
            Some(expected)
        );
    }

    #[rstest]
    fn keeps_every_digit_of_an_integer_past_64_bits() {
        let write = spend_log_write(
            &entry(r#", "metadata": "{\"n\": 18446744073709551616}""#),
            None,
            now(),
        )
        .expect("writes");

        assert_eq!(
            write
                .log
                .and_then(|log| log.metadata)
                .map(|column| column.get().to_owned()),
            Some(r#"{"n":18446744073709551616}"#.to_owned())
        );
    }

    #[rstest]
    fn drops_nul_from_the_text_columns_of_a_rust_built_row() {
        let row = SpendLogRow {
            user: "a\0b".to_owned(),
            session_id: Some("s\0".to_owned()),
            ..full_row("r")
        };
        let queued = QueuedSpendLog::new(row, None, true).expect("queued");

        let log = spend_log_write(&queued, None, now())
            .expect("writes")
            .log
            .expect("logged");

        assert_eq!(
            (log.user.as_deref(), log.session_id.as_deref()),
            (Some("ab"), Some("s"))
        );
    }

    #[rstest]
    #[case::rfc3339_offset("2026-11-02T14:00:00+02:00", "2026-11-02T12:00:00")]
    #[case::naive_iso("2026-11-02T12:00:00.5", "2026-11-02T12:00:00.500")]
    fn reads_iso_timestamps(#[case] text: &str, #[case] expected: &str) {
        let write = spend_log_write(&entry(&format!(r#", "startTime": "{text}""#)), None, now())
            .expect("writes");

        assert_eq!(
            write.log.map(|log| log.start_time),
            Some(NaiveDateTime::parse_from_str(expected, "%Y-%m-%dT%H:%M:%S%.f").expect("parses"))
        );
    }

    #[rstest]
    #[case::invalid_json(r#", "messages": "not json""#)]
    #[case::not_a_time(r#", "startTime": "soon""#)]
    #[case::too_many_tokens(r#", "total_tokens": 9999999999"#)]
    #[case::null_spend(r#", "spend": null"#)]
    fn refuses_a_row_postgres_cannot_hold(#[case] fields: &str) {
        assert!(matches!(
            spend_log_write(&entry(fields), None, now()),
            Err(RowError::InvalidJson(_)
                | RowError::InvalidTimestamp(_)
                | RowError::OutOfRange(_)
                | RowError::NullSpend)
        ));
    }

    #[rstest]
    #[case::ledger_row(r#", "accounting_delivery_id": "d", "accounting_payload": {"spend": 1.0}"#, Some(("d", false)), true)]
    #[case::settle_once_job(r#", "accounting_payload": {"spend": 1.0}"#, Some(("r", true)), true)]
    #[case::uncharged("", None, true)]
    fn writes_the_delivery_the_row_carries(
        #[case] fields: &str,
        #[case] expected: Option<(&str, bool)>,
        #[case] logged: bool,
    ) {
        let write = spend_log_write(&entry(fields), None, now()).expect("writes");

        assert_eq!(
            write
                .delivery
                .as_ref()
                .map(|delivery| (delivery.delivery_id.as_str(), delivery.is_batch)),
            expected
        );
        assert_eq!(write.log.is_some(), logged);
    }
}
