use std::{collections::BTreeMap, io::Write, time::Duration};

use flate2::write::GzDecoder;
use litellm_http::Client;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_ENCODING};
use serde::{
    Deserialize, Serialize,
    de::{DeserializeOwned, IgnoredAny},
};

use crate::{Connection, Error};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadLimits {
    pub result_rows: u64,
    pub response_bytes: usize,
    pub execution_seconds: u64,
}

pub const READ_LIMITS: ReadLimits = ReadLimits {
    result_rows: 1000,
    response_bytes: 4 * 1024 * 1024,
    execution_seconds: 10,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Parameter {
    Text(String),
    Integer(i64),
    Unsigned(u64),
    Float(f64),
    Strings(Vec<String>),
}

impl Parameter {
    fn encoded(&self) -> String {
        match self {
            Self::Text(value) => escaped(value),
            Self::Integer(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
            Self::Strings(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(|value| format!("'{}'", escaped(value).replace('\'', "\\'")))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

fn escaped(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\0', "\\0")
}

pub async fn execute_read(
    client: &Client,
    connection: &Connection,
    sql: &str,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error> {
    let body = execute_read_with_limits(client, connection, sql, parameters, READ_LIMITS).await?;
    decode_rows::<IgnoredAny>(&body)?;
    String::from_utf8(body).map_err(|_| Error::InvalidResponse)
}

async fn execute_read_with_limits(
    client: &Client,
    connection: &Connection,
    sql: &str,
    parameters: &BTreeMap<String, Parameter>,
    limits: ReadLimits,
) -> Result<Vec<u8>, Error> {
    if sql.trim().is_empty() {
        return Err(Error::EmptySql);
    }

    let mut url = connection.url().clone();

    let existing_pairs: Vec<(String, String)> = url
        .query_pairs()
        .filter(|(key, _)| {
            !key.starts_with("param_")
                && !matches!(
                    key.as_ref(),
                    "query"
                        | "readonly"
                        | "default_format"
                        | "max_result_rows"
                        | "result_overflow_mode"
                        | "max_execution_time"
                        | "wait_end_of_query"
                        | "enable_http_compression"
                )
        })
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.query_pairs_mut()
        .clear()
        .extend_pairs(existing_pairs)
        .append_pair("readonly", "1")
        .append_pair("max_result_rows", &limits.result_rows.to_string())
        .append_pair("result_overflow_mode", "throw")
        .append_pair("max_execution_time", &limits.execution_seconds.to_string())
        .append_pair("wait_end_of_query", "1")
        .append_pair("enable_http_compression", "1")
        .append_pair("default_format", "JSON");

    url.query_pairs_mut().extend_pairs(
        parameters
            .iter()
            .map(|(name, value)| (format!("param_{name}"), value.encoded())),
    );

    let request = client
        .post(url)
        .timeout(Duration::from_secs(15))
        .header(ACCEPT_ENCODING, "gzip")
        .body(sql.to_owned());
    let mut response = request.send().await.map_err(|_| Error::Transport)?;
    if !response.status().is_success() {
        if response
            .headers()
            .get("x-clickhouse-exception-code")
            .is_some_and(|code| code == "396")
        {
            return Err(Error::ResponseTooLarge);
        }
        return Err(Error::QueryFailed(response.status().as_u16()));
    }

    let capped = Capped {
        bytes: Vec::new(),
        limit: limits.response_bytes,
    };
    let mut sink = match response.headers().get(CONTENT_ENCODING) {
        None => Sink::Identity(capped),
        Some(encoding) if encoding == "gzip" => Sink::Gzip(GzDecoder::new(capped)),
        Some(_) => return Err(Error::InvalidResponse),
    };
    while let Some(chunk) = response.chunk().await.map_err(|_| Error::Transport)? {
        sink.write_all(&chunk).map_err(body_error)?;
    }
    sink.finish().map_err(body_error)
}

struct Capped {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for Capped {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.bytes.len() + buf.len() > self.limit {
            return Err(std::io::Error::from(std::io::ErrorKind::FileTooLarge));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

enum Sink {
    Identity(Capped),
    Gzip(GzDecoder<Capped>),
}

impl Sink {
    fn write_all(&mut self, chunk: &[u8]) -> std::io::Result<()> {
        match self {
            Self::Identity(capped) => capped.write_all(chunk),
            Self::Gzip(decoder) => decoder.write_all(chunk),
        }
    }

    fn finish(self) -> std::io::Result<Vec<u8>> {
        match self {
            Self::Identity(capped) => Ok(capped.bytes),
            Self::Gzip(decoder) => decoder.finish().map(|capped| capped.bytes),
        }
    }
}

fn body_error(error: std::io::Error) -> Error {
    match error.kind() {
        std::io::ErrorKind::FileTooLarge => Error::ResponseTooLarge,
        _ => Error::InvalidResponse,
    }
}

pub trait Query {
    type Params: Serialize;
    type Row: DeserializeOwned;

    const READ_LIMITS: ReadLimits = crate::read::READ_LIMITS;
    const SQL: &'static str;
}

#[derive(Deserialize)]
struct Rows<T> {
    data: Vec<T>,
    exception: Option<IgnoredAny>,
}

fn parameters<T: Serialize>(params: &T) -> Result<BTreeMap<String, Parameter>, Error> {
    let value = serde_json::to_value(params).map_err(|_| Error::InvalidParameters)?;
    serde_json::from_value(value).map_err(|_| Error::InvalidParameters)
}

pub async fn fetch<Q: Query>(
    client: &Client,
    connection: &Connection,
    params: &Q::Params,
) -> Result<Vec<Q::Row>, Error> {
    let body = execute_read_with_limits(
        client,
        connection,
        Q::SQL,
        &parameters(params)?,
        Q::READ_LIMITS,
    )
    .await?;
    decode_rows::<Q::Row>(&body)
}

pub async fn fetch_json<Q: Query>(
    client: &Client,
    connection: &Connection,
    params: &Q::Params,
) -> Result<String, Error> {
    let body = execute_read_with_limits(
        client,
        connection,
        Q::SQL,
        &parameters(params)?,
        Q::READ_LIMITS,
    )
    .await?;
    decode_rows::<Q::Row>(&body)?;
    String::from_utf8(body).map_err(|_| Error::InvalidResponse)
}

fn decode_rows<T: DeserializeOwned>(body: &[u8]) -> Result<Vec<T>, Error> {
    match serde_json::from_slice::<Rows<T>>(body) {
        Ok(Rows {
            data,
            exception: None,
        }) => Ok(data),
        _ => Err(Error::InvalidResponse),
    }
}
