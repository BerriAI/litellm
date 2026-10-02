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

use crate::{Connection, Error, Shared};

const MAX_INSERT_BYTES: usize = 64 * 1024 * 1024;

pub type InsertRow = BTreeMap<String, Shared<Value>>;

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
    insert_shared_rows(client, connection, database, table, shared_rows(rows)).await
}

pub async fn insert_shared_rows(
    client: &Client,
    connection: &Connection,
    database: &str,
    table: InsertTable,
    rows: Vec<InsertRow>,
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    let received_ms = (OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as u64;
    let (token, body) = prepare_insert(&rows, received_ms, MAX_INSERT_BYTES)?;
    litellm_storage_clickhouse::insert_compressed_rows(
        client,
        connection,
        database,
        table.name(),
        &token,
        body,
    )
    .await
}

fn shared_rows(rows: Vec<BTreeMap<String, Value>>) -> Vec<InsertRow> {
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(key, value)| (key, Shared::new(value)))
                .collect()
        })
        .collect()
}

pub fn encode_rows(rows: Vec<BTreeMap<String, Value>>) -> Result<String, Error> {
    let body = write_rows(&shared_rows(rows), None, Vec::new(), usize::MAX)?;
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
        "Timestamp" => 1,
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
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::json;

    use super::{shared_rows, write_rows};
    use crate::Error;

    #[rstest]
    fn encoded_limit_counts_utf8_bytes_across_rows() {
        let rows = shared_rows(vec![
            BTreeMap::from([("Input".to_owned(), json!("雪"))]),
            BTreeMap::from([("Input".to_owned(), json!("雪"))]),
        ]);
        let encoded = write_rows(&rows, None, Vec::new(), usize::MAX).expect("valid rows");

        assert!(write_rows(&rows, None, Vec::new(), encoded.len()).is_ok());
        assert!(matches!(
            write_rows(&rows, None, Vec::new(), encoded.len() - 1),
            Err(Error::InsertTooLarge)
        ));
    }

    #[rstest]
    #[case::absent(None)]
    #[case::submitted(Some(123))]
    fn streamed_insert_preserves_token_and_stamps_receive_time(#[case] submitted: Option<u64>) {
        use flate2::read::GzDecoder;
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut row = BTreeMap::from([
            ("ApiKeyHash".into(), json!("key")),
            ("ResourceAttributes".into(), json!({"message": "雪\n\""})),
            ("Timestamp".into(), json!(1_234_567_890)),
        ]);
        if let Some(value) = submitted {
            row.insert("EngineReceivedMs".into(), json!(value));
        }
        let legacy = match submitted {
            Some(_) => {
                "{\"ApiKeyHash\":\"key\",\"EngineReceivedMs\":123,\"ResourceAttributes\":{\"message\":\"雪\\n\\\"\"},\"Timestamp\":\"1970-01-01T00:00:01.23456789Z\"}"
            }
            None => {
                "{\"ApiKeyHash\":\"key\",\"ResourceAttributes\":{\"message\":\"雪\\n\\\"\"},\"Timestamp\":\"1970-01-01T00:00:01.23456789Z\"}"
            }
        };
        let rows = shared_rows(vec![row.clone(), row]);
        let (token, body) = super::prepare_insert(&rows, 456, 4096).unwrap();
        assert_eq!(
            token,
            format!("{:x}", Sha256::digest(format!("{legacy}\n{legacy}")))
        );
        let mut decoded = String::new();
        GzDecoder::new(body.as_slice())
            .read_to_string(&mut decoded)
            .unwrap();
        let expected = json!({
            "ApiKeyHash": "key", "EngineReceivedMs": 456,
            "ResourceAttributes": {"message": "雪\n\""},
            "Timestamp": "1970-01-01T00:00:01.23456789Z",
        });
        assert_eq!(
            decoded
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .collect::<Vec<_>>(),
            vec![expected.clone(), expected]
        );
        assert_eq!(
            rows[0]
                .get("EngineReceivedMs")
                .map(|value| value.as_u64().unwrap()),
            submitted
        );
    }

    #[rstest]
    fn stamped_insert_enforces_the_encoded_limit() {
        let rows = shared_rows(vec![BTreeMap::new()]);
        assert!(super::prepare_insert(&rows, 1, 22).is_ok());
        assert!(matches!(
            super::prepare_insert(&rows, 1, 21),
            Err(Error::InsertTooLarge)
        ));
    }
}
