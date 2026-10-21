use crate::{Connection, Error, Parameter};
use litellm_http::Client;
use litellm_traces::Tenant;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Receipt {
    received: u32,
}

#[derive(Deserialize)]
struct Rows {
    data: Vec<Receipt>,
}

pub async fn trace_received(
    client: &Client,
    connection: &Connection,
    tenant: &Tenant,
    trace_id: &str,
    span_ids: &[String],
) -> Result<bool, Error> {
    let valid_id =
        |value: &str, length| value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit());
    if !valid_id(trace_id, 32)
        || span_ids.len() > 1000
        || span_ids.iter().any(|id| !valid_id(id, 16))
    {
        return Err(Error::InvalidParameters);
    }
    let spans: BTreeSet<_> = span_ids.iter().map(|id| id.to_ascii_lowercase()).collect();
    let expected = spans.len();
    let parameters = BTreeMap::from([
        (
            "trace_id".into(),
            Parameter::Text(trace_id.to_ascii_lowercase()),
        ),
        (
            "api_key_hash".into(),
            Parameter::Text(tenant.api_key_hash.clone()),
        ),
        (
            "span_ids".into(),
            Parameter::Strings(spans.into_iter().collect()),
        ),
    ]);
    let response = litellm_storage_clickhouse::execute_read(client, connection,
        "SELECT toUInt32(uniqExact(SpanId)) AS received FROM otel_traces WHERE TraceId={trace_id:String} AND ApiKeyHash={api_key_hash:String} AND (empty({span_ids:Array(String)}) OR has({span_ids:Array(String)}, SpanId))", &parameters).await?;
    let rows: Rows = serde_json::from_str(&response).map_err(|_| Error::InvalidResponse)?;
    let row = rows.data.first().ok_or(Error::InvalidResponse)?;
    Ok(if expected == 0 {
        row.received > 0
    } else {
        row.received as usize == expected
    })
}
