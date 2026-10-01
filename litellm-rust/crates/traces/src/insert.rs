use std::collections::BTreeMap;

use litellm_http::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{Connection, Error};

const MAX_INSERT_BYTES: usize = 64 * 1024 * 1024;

pub enum InsertTable {
    OtelTraces,
    SpendLogs,
}

impl InsertTable {
    pub fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "otel_traces" => Ok(Self::OtelTraces),
            "spend_logs" => Ok(Self::SpendLogs),
            _ => Err(Error::InvalidTable),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::OtelTraces => "otel_traces",
            Self::SpendLogs => "spend_logs",
        }
    }
}

pub async fn insert_rows(
    client: &Client,
    connection: &Connection,
    database: &str,
    table: InsertTable,
    rows: Vec<BTreeMap<String, Value>>,
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    let token = format!(
        "{:x}",
        Sha256::digest(encode_rows_with_limit(rows.clone(), MAX_INSERT_BYTES)?.as_bytes())
    );
    let received_ms = OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000;
    let rows = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .filter(|(key, _)| key != "EngineReceivedMs")
                .chain(std::iter::once((
                    "EngineReceivedMs".to_owned(),
                    Value::from(received_ms as u64),
                )))
                .collect()
        })
        .collect();
    let encoded = encode_rows_with_limit(rows, MAX_INSERT_BYTES)?;
    litellm_storage_clickhouse::insert_encoded_rows(
        client,
        connection,
        database,
        table.name(),
        &token,
        &encoded,
    )
    .await
}

pub fn encode_rows(rows: Vec<BTreeMap<String, Value>>) -> Result<String, Error> {
    encode_rows_with_limit(rows, usize::MAX)
}

fn encode_rows_with_limit(
    rows: Vec<BTreeMap<String, Value>>,
    limit: usize,
) -> Result<String, Error> {
    let mut body = Vec::new();
    for row in rows {
        let encoded = row
            .into_iter()
            .map(|(name, value)| insert_value(&name, value).map(|value| (name, value)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let record = serde_json::to_vec(&encoded).map_err(|_| Error::InvalidRow)?;
        let size = body
            .len()
            .checked_add(record.len())
            .and_then(|size| size.checked_add(usize::from(!body.is_empty())))
            .ok_or(Error::InsertTooLarge)?;
        if size > limit {
            return Err(Error::InsertTooLarge);
        }
        if !body.is_empty() {
            body.push(b'\n');
        }
        body.extend_from_slice(&record);
    }
    String::from_utf8(body).map_err(|_| Error::InvalidRow)
}

fn insert_value(name: &str, value: Value) -> Result<Value, Error> {
    let multiplier = match name {
        "Timestamp" => 1,
        "start_time" | "end_time" | "completion_start_time" => 1_000_000,
        _ => return Ok(value),
    };
    if name == "completion_start_time" && value.is_null() {
        return Ok(value);
    }
    let timestamp = value.as_i64().ok_or(Error::InvalidRow)?;
    let datetime = OffsetDateTime::from_unix_timestamp_nanos(i128::from(timestamp) * multiplier)
        .map_err(|_| Error::InvalidRow)?;
    datetime
        .format(&Rfc3339)
        .map(Value::String)
        .map_err(|_| Error::InvalidRow)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::json;

    use super::encode_rows_with_limit;
    use crate::Error;

    #[rstest]
    fn encoded_limit_counts_utf8_bytes_across_rows() {
        let rows = vec![
            BTreeMap::from([("Input".to_owned(), json!("雪"))]),
            BTreeMap::from([("Input".to_owned(), json!("雪"))]),
        ];
        let encoded = encode_rows_with_limit(rows.clone(), usize::MAX).expect("valid rows");

        assert!(encode_rows_with_limit(rows.clone(), encoded.len()).is_ok());
        assert!(matches!(
            encode_rows_with_limit(rows, encoded.len() - 1),
            Err(Error::InsertTooLarge)
        ));
    }
}
