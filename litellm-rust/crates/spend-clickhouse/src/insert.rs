use std::{
    borrow::Cow,
    collections::BTreeMap,
    io::{BufWriter, Write},
};

use serde::{Serialize, Serializer, ser::SerializeMap};

use flate2::{Compression, write::GzEncoder};
use litellm_http::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::Error;
use litellm_storage_clickhouse::Connection;

fn max_insert_bytes() -> Result<usize, Error> {
    let name = "CLICKHOUSE_TRACE_MAX_INSERT_BYTES";
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or(Error::InvalidLimit(name)),
        Err(std::env::VarError::NotPresent) => Ok(64 * 1024 * 1024),
        Err(_) => Err(Error::InvalidLimit(name)),
    }
}

type InsertRow = BTreeMap<String, Value>;

pub async fn insert_rows(
    client: &Client,
    connection: &Connection,
    database: &str,
    rows: Vec<InsertRow>,
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    let received_ms = (OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as u64;
    let (token, body) = prepare_insert(&rows, received_ms, max_insert_bytes()?)?;
    litellm_storage_clickhouse::insert_compressed_rows(
        client,
        connection,
        database,
        "spend_logs",
        &token,
        body,
    )
    .await
    .map_err(Error::from)
}

pub fn encode_rows(rows: Vec<InsertRow>) -> Result<String, Error> {
    let body = write_rows(&rows, None, Vec::new(), usize::MAX)?;
    String::from_utf8(body).map_err(|_| Error::InvalidRow)
}

fn prepare_insert(
    rows: &[InsertRow],
    received_ms: u64,
    limit: usize,
) -> Result<(String, Vec<u8>), Error> {
    let hash = write_rows(rows, None, HashWriter(Sha256::new()), limit)?;
    let token = format!("{:x}", hash.0.finalize());
    let encoder = write_rows(
        rows,
        Some(received_ms),
        BufWriter::new(GzEncoder::new(Vec::new(), Compression::default())),
        limit,
    )?;
    let body = encoder
        .into_inner()
        .map_err(|_| Error::InvalidRow)?
        .finish()
        .map_err(|_| Error::InvalidRow)?;
    Ok((token, body))
}

struct HashWriter(Sha256);

impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct LimitedWriter<W> {
    inner: W,
    remaining: usize,
    exceeded: bool,
}

impl<W: Write> Write for LimitedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(std::io::Error::other(Error::InsertTooLarge));
        }
        let written = self.inner.write(bytes)?;
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn write_rows<W: Write>(
    rows: &[InsertRow],
    received_ms: Option<u64>,
    writer: W,
    limit: usize,
) -> Result<W, Error> {
    let mut writer = LimitedWriter {
        inner: writer,
        remaining: limit,
        exceeded: false,
    };
    for (index, row) in rows.iter().enumerate() {
        let result = (|| {
            if index != 0 {
                writer.write_all(b"\n").map_err(serde_json::Error::io)?;
            }
            serde_json::to_writer(&mut writer, &EncodedRow { row, received_ms })
        })();
        if result.is_err() {
            return Err(if writer.exceeded {
                Error::InsertTooLarge
            } else {
                Error::InvalidRow
            });
        }
    }
    Ok(writer.inner)
}

struct EncodedRow<'a> {
    row: &'a InsertRow,
    received_ms: Option<u64>,
}

impl Serialize for EncodedRow<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        let mut received_ms = self.received_ms;
        for (name, value) in self.row {
            if name.as_str() >= "EngineReceivedMs"
                && let Some(timestamp) = received_ms.take()
            {
                map.serialize_entry("EngineReceivedMs", &timestamp)?;
            }
            if name == "EngineReceivedMs" && self.received_ms.is_some() {
                continue;
            }
            let value = insert_value(name, value).map_err(serde::ser::Error::custom)?;
            map.serialize_entry(name, &value)?;
        }
        if let Some(timestamp) = received_ms {
            map.serialize_entry("EngineReceivedMs", &timestamp)?;
        }
        map.end()
    }
}

fn insert_value<'a>(name: &str, value: &'a Value) -> Result<Cow<'a, Value>, Error> {
    let multiplier = match name {
        "start_time" | "end_time" | "completion_start_time" => 1_000_000,
        _ => return Ok(Cow::Borrowed(value)),
    };
    if name == "completion_start_time" && value.is_null() {
        return Ok(Cow::Borrowed(value));
    }
    let timestamp = value.as_i64().ok_or(Error::InvalidRow)?;
    let datetime = OffsetDateTime::from_unix_timestamp_nanos(i128::from(timestamp) * multiplier)
        .map_err(|_| Error::InvalidRow)?;
    datetime
        .format(&Rfc3339)
        .map(|value| Cow::Owned(Value::String(value)))
        .map_err(|_| Error::InvalidRow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use rstest::rstest;
    use serde_json::json;
    use std::io::Read;

    #[rstest]
    fn encoded_limit_counts_utf8_bytes_across_rows() {
        let rows = vec![
            BTreeMap::from([("metadata".into(), json!("雪"))]),
            BTreeMap::from([("metadata".into(), json!("雪"))]),
        ];
        let encoded = write_rows(&rows, None, Vec::new(), usize::MAX).unwrap();
        assert!(write_rows(&rows, None, Vec::new(), encoded.len()).is_ok());
        assert!(matches!(
            write_rows(&rows, None, Vec::new(), encoded.len() - 1),
            Err(Error::InsertTooLarge)
        ));
    }

    #[rstest]
    #[case::absent(None)]
    #[case::submitted(Some(123))]
    fn retry_token_excludes_receive_time_and_input_is_unchanged(#[case] submitted: Option<u64>) {
        let row = BTreeMap::from([
            ("metadata".into(), json!({"message": "雪\n\""})),
            ("start_time".into(), json!(1_234)),
        ]);
        let row = match submitted {
            Some(value) => row
                .into_iter()
                .chain([("EngineReceivedMs".into(), json!(value))])
                .collect(),
            None => row,
        };
        let original = row.clone();
        let legacy = match submitted {
            Some(_) => {
                "{\"EngineReceivedMs\":123,\"metadata\":{\"message\":\"雪\\n\\\"\"},\"start_time\":\"1970-01-01T00:00:01.234Z\"}"
            }
            None => {
                "{\"metadata\":{\"message\":\"雪\\n\\\"\"},\"start_time\":\"1970-01-01T00:00:01.234Z\"}"
            }
        };
        let rows = vec![row.clone(), row];
        let (token, body) = prepare_insert(&rows, 456, 4096).unwrap();
        let (retry_token, _) = prepare_insert(&rows, 789, 4096).unwrap();
        assert_eq!(token, retry_token);
        assert_eq!(
            token,
            format!("{:x}", Sha256::digest(format!("{legacy}\n{legacy}")))
        );
        let mut decoded = String::new();
        GzDecoder::new(body.as_slice())
            .read_to_string(&mut decoded)
            .unwrap();
        let expected = json!({
            "EngineReceivedMs":456, "metadata":{"message":"雪\n\""},
            "start_time":"1970-01-01T00:00:01.234Z"
        });
        assert_eq!(
            decoded
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>(),
            vec![expected.clone(), expected]
        );
        assert_eq!(rows[0], original);
    }

    #[rstest]
    fn stamped_insert_enforces_the_encoded_limit() {
        let rows = vec![BTreeMap::new()];
        assert!(prepare_insert(&rows, 1, 22).is_ok());
        assert!(matches!(
            prepare_insert(&rows, 1, 21),
            Err(Error::InsertTooLarge)
        ));
    }

    #[rstest]
    #[case::milliseconds("start_time", json!(1_234), json!("1970-01-01T00:00:01.234Z"))]
    #[case::negative("end_time", json!(-1), json!("1969-12-31T23:59:59.999Z"))]
    #[case::nullable("completion_start_time", Value::Null, Value::Null)]
    fn spend_dates_preserve_precision_and_null(
        #[case] field: &str,
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(insert_value(field, &input).unwrap().as_ref(), &expected);
    }

    #[rstest]
    #[case::fractional(json!(1.5))]
    #[case::string(json!("1234"))]
    #[case::null(Value::Null)]
    #[case::out_of_range(json!(i64::MAX))]
    fn invalid_spend_dates_fail_before_transport(#[case] input: Value) {
        assert!(matches!(
            prepare_insert(&[BTreeMap::from([("start_time".into(), input)])], 456, 4096),
            Err(Error::InvalidRow)
        ));
    }
}
