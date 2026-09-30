use std::{collections::BTreeMap, io::Write, time::Duration};

use flate2::{Compression, write::GzEncoder};
use litellm_http::Client;
use serde_json::Value;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{Connection, Error};

const MAX_INSERT_BYTES: usize = 64 * 1024 * 1024;
const INSERT_TIMEOUT: Duration = Duration::from_secs(30);

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
    let encoded = encode_rows_with_limit(rows, MAX_INSERT_BYTES)?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(encoded.as_bytes())
        .map_err(|_| Error::InvalidRow)?;
    let body = encoder.finish().map_err(|_| Error::InvalidRow)?;
    let mut url = connection.url().clone();
    url.query_pairs_mut()
        .append_pair(
            "query",
            &format!(
                "INSERT INTO `{database}`.{} FORMAT JSONEachRow",
                table.name()
            ),
        )
        .append_pair("async_insert", "1")
        .append_pair("wait_for_async_insert", "1")
        .append_pair("date_time_input_format", "best_effort");
    let response = client
        .post(url)
        .timeout(INSERT_TIMEOUT)
        .header("Content-Encoding", "gzip")
        .body(body)
        .send()
        .await
        .map_err(|_| Error::Transport)?;
    if !response.status().is_success() {
        return Err(Error::InsertFailed(response.status().as_u16()));
    }
    Ok(())
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
