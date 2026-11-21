//! The rows a commit observer hears about once their ledger transaction committed: a bounded
//! summary per accepted delivery, projected from the delivery's accounting payload, never the
//! request content.

use std::{collections::HashSet, sync::Arc};

use litellm_repositories::{SpendLogDelivery, SpendLogRecord, SpendLogWrite};
use serde_json::{Map, Value, json};

/// Called with one JSON object per accepted delivery after each committed statement. It runs on
/// the flushing task, so it must hand the rows off without blocking.
pub type CommitObserver = Arc<dyn Fn(Vec<String>) + Send + Sync>;

/// A string field longer than this makes its delivery unreportable; the observer still hears
/// the delivery as an `unreportable` marker so it can count it.
pub const MAX_FIELD_BYTES: usize = 512;

const TIMESTAMP: &str = "%Y-%m-%dT%H:%M:%S%.6f";

/// The accounting payload fields that decide cost, usage and identity. A field the payload
/// lacks is reported as null, never filled from anywhere else.
const ACCOUNTING_FIELDS: [&str; 16] = [
    "request_id",
    "call_type",
    "api_key",
    "user",
    "team_id",
    "organization_id",
    "end_user",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "startTime",
    "model",
    "model_group",
    "custom_llm_provider",
    "status",
    "request_duration_ms",
];

fn fits(value: &Value) -> bool {
    match value {
        Value::String(text) => text.len() <= MAX_FIELD_BYTES,
        Value::Array(_) | Value::Object(_) => false,
        _ => true,
    }
}

fn fits_text(value: Option<&str>) -> bool {
    value.is_none_or(|text| text.len() <= MAX_FIELD_BYTES)
}

fn supplement_fits(log: &SpendLogRecord) -> bool {
    [
        log.model_id.as_deref(),
        log.api_base.as_deref(),
        log.session_id.as_deref(),
        log.cache_hit.as_deref(),
        log.litellm_call_id.as_deref(),
    ]
    .into_iter()
    .all(fits_text)
}

fn supplement(log: &SpendLogRecord) -> [(&'static str, Option<Value>); 6] {
    [
        (
            "endTime",
            Some(Value::from(log.end_time.format(TIMESTAMP).to_string())),
        ),
        ("model_id", log.model_id.clone().map(Value::from)),
        ("api_base", log.api_base.clone().map(Value::from)),
        ("session_id", log.session_id.clone().map(Value::from)),
        ("cache_hit", log.cache_hit.clone().map(Value::from)),
        (
            "litellm_call_id",
            log.litellm_call_id.clone().map(Value::from),
        ),
    ]
}

fn unreportable(delivery: &SpendLogDelivery) -> Value {
    let request_id = (delivery.request_id.len() <= MAX_FIELD_BYTES)
        .then(|| Value::from(delivery.request_id.clone()))
        .unwrap_or(Value::Null);
    json!({
        "delivery_id": delivery.delivery_id.chars().take(MAX_FIELD_BYTES).collect::<String>(),
        "request_id": request_id,
        "unreportable": true,
    })
}

fn summary(write: &SpendLogWrite) -> Option<String> {
    let delivery = write.delivery.as_ref()?;
    let payload = &delivery.accounting_payload;
    let reportable = delivery.delivery_id.len() <= MAX_FIELD_BYTES
        && delivery.request_id.len() <= MAX_FIELD_BYTES
        && ACCOUNTING_FIELDS
            .iter()
            .all(|name| payload.get(*name).is_none_or(fits))
        && write.log.as_ref().is_none_or(supplement_fits);
    if !reportable {
        return serde_json::to_string(&unreportable(delivery)).ok();
    }
    let mut fields: Map<String, Value> = Map::new();
    for (name, value) in write.log.as_ref().map(supplement).into_iter().flatten() {
        if let Some(value) = value {
            fields.insert(name.into(), value);
        }
    }
    for name in ACCOUNTING_FIELDS {
        fields.insert(
            name.into(),
            payload.get(name).cloned().unwrap_or(Value::Null),
        );
    }
    if fields["request_id"].is_null() {
        fields.insert(
            "request_id".into(),
            Value::from(delivery.request_id.clone()),
        );
    }
    fields.insert(
        "delivery_id".into(),
        Value::from(delivery.delivery_id.clone()),
    );
    serde_json::to_string(&fields).ok()
}

/// One summary per write whose delivery the commit accepted, first write per delivery.
pub(crate) fn committed_rows(writes: &[SpendLogWrite], accepted: &[String]) -> Vec<String> {
    let accepted: HashSet<&str> = accepted.iter().map(String::as_str).collect();
    let mut reported: HashSet<&str> = HashSet::with_capacity(accepted.len());
    writes
        .iter()
        .filter(|write| {
            write.delivery.as_ref().is_some_and(|delivery| {
                let id = delivery.delivery_id.as_str();
                accepted.contains(id) && reported.insert(id)
            })
        })
        .filter_map(summary)
        .collect()
}

#[cfg(test)]
mod tests {
    use litellm_repositories::{SpendLogDelivery, SpendLogWrite};
    use serde_json::{Value, json};

    use super::{MAX_FIELD_BYTES, committed_rows};

    fn charge(delivery_id: &str, payload: Value) -> SpendLogWrite {
        SpendLogWrite {
            log: None,
            delivery: Some(SpendLogDelivery {
                delivery_id: delivery_id.into(),
                request_id: format!("request-{delivery_id}"),
                accounting_payload: payload.as_object().unwrap().clone(),
                is_batch: false,
            }),
        }
    }

    #[test]
    fn summaries_carry_only_the_accepted_accounting_projection() {
        let writes = [
            charge(
                "a",
                json!({
                    "request_id": "request-a", "spend": 0.5, "prompt_tokens": 3,
                    "completion_tokens": 4, "api_key": "hash", "team_id": "team",
                    "metadata": {"accounting_windows": ["large"]}, "messages": "secret",
                }),
            ),
            charge("a", json!({"request_id": "request-a", "spend": 9.0})),
            charge(
                "rejected",
                json!({"request_id": "request-rejected", "spend": 1.0}),
            ),
        ];
        let rows: Vec<Value> = committed_rows(&writes, &["a".into()])
            .iter()
            .map(|text| serde_json::from_str(text).unwrap())
            .collect();
        assert_eq!(rows.len(), 1);
        let row = rows[0].as_object().unwrap();
        assert_eq!(
            (
                row["spend"].clone(),
                row["api_key"].clone(),
                row["delivery_id"].clone()
            ),
            (json!(0.5), json!("hash"), json!("a"))
        );
        assert_eq!(row["end_user"], Value::Null);
        assert_eq!(row["user"], Value::Null);
        assert!(row.get("metadata").is_none() && row.get("messages").is_none());
    }

    #[test]
    fn oversized_deliveries_are_reported_as_unreportable_markers() {
        let writes = [charge(
            "a",
            json!({"request_id": "request-a", "user": "u".repeat(MAX_FIELD_BYTES + 1)}),
        )];
        let rows: Vec<Value> = committed_rows(&writes, &["a".into()])
            .iter()
            .map(|text| serde_json::from_str(text).unwrap())
            .collect();
        assert_eq!(
            rows,
            [json!({"delivery_id": "a", "request_id": "request-a", "unreportable": true})]
        );
    }
}
