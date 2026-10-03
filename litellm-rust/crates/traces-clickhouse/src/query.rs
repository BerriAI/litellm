use std::collections::{BTreeMap, BTreeSet};

use crate::TraceTable;
use futures_util::{
    StreamExt,
    stream::{self, TryStreamExt},
};
use litellm_http::Client;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use strum::IntoEnumIterator;

use super::{
    Connection, Error, NORMALIZED_FIELD_DEFINITIONS, NormalizedFieldDefinition, Parameter,
    query_access::READER_LIMITS,
};

mod guide;
pub mod lens;
pub mod named;
mod number;

const SAMPLE_ROWS: usize = 200;
const MAX_FIELDS: usize = 200;
const MAX_DEPTH: usize = 16;
const METADATA_SQL: &str = "SELECT metadata FROM spend_logs FINAL \
    WHERE start_time >= now() - INTERVAL 7 DAY AND length(metadata) <= 8192 \
    LIMIT 201";
const METADATA_SCOPE: &str = "Up to 200 unordered rows from the last 7 days, excluding metadata larger than 8192 bytes; up to 200 paths and 16 levels. Missing paths may exist outside this sample. Array indexes are 1-based and describe sampled positions, not a fixed schema";
const ATTRIBUTE_SCOPE: &str = "Distinct keys from up to 200 unordered spans in the last 7 days; up to 200 keys per map. Missing keys may exist outside this sample";

#[derive(Deserialize)]
struct Rows<T> {
    data: Vec<T>,
}

#[derive(Deserialize)]
struct MetadataRow {
    metadata: String,
}

#[derive(Deserialize)]
struct AttributeRow {
    key: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(untagged)]
enum PathPart {
    Key(String),
    Index(usize),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, strum::Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
enum JsonKind {
    Array,
    Boolean,
    Integer,
    Null,
    Number,
    Object,
    String,
}

impl JsonKind {
    fn of(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(_) => Self::Boolean,
            Value::Number(number) if number.is_i64() || number.is_u64() => Self::Integer,
            Value::Number(_) => Self::Number,
            Value::String(_) => Self::String,
            Value::Array(_) => Self::Array,
            Value::Object(_) => Self::Object,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, strum::Display)]
enum MapValueType {
    String,
}

#[derive(Serialize)]
struct MetadataField {
    path: Vec<PathPart>,
    types: BTreeSet<JsonKind>,
    expression: String,
}

#[derive(Deserialize, Serialize)]
struct ColumnSchema {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(flatten)]
    details: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct TableSchema {
    name: TraceTable,
    columns: Vec<ColumnSchema>,
}

trait Unobserved {
    fn unobserved() -> Self;
}

enum Discovery<T> {
    Observed(T),
    Unavailable(String),
}

impl<T: Serialize + Unobserved> Serialize for Discovery<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Unavailable<'a, T> {
            #[serde(flatten)]
            sample: T,
            error: &'a str,
        }
        match self {
            Self::Observed(sample) => sample.serialize(serializer),
            Self::Unavailable(error) => Unavailable {
                sample: T::unobserved(),
                error,
            }
            .serialize(serializer),
        }
    }
}

#[derive(Serialize)]
struct MetadataSample {
    fields: Vec<MetadataField>,
    sampled_rows: usize,
    invalid_json_rows: usize,
    truncated: bool,
}

impl Unobserved for MetadataSample {
    fn unobserved() -> Self {
        Self {
            fields: Vec::new(),
            sampled_rows: 0,
            invalid_json_rows: 0,
            truncated: true,
        }
    }
}

#[derive(Serialize)]
struct MetadataCatalog {
    table: TraceTable,
    column: &'static str,
    #[serde(flatten)]
    discovery: Discovery<MetadataSample>,
    sample_sql: &'static str,
    scope: &'static str,
}

#[derive(Serialize)]
struct AttributeField {
    key: String,
    #[serde(rename = "type")]
    kind: MapValueType,
    expression: String,
}

#[derive(Serialize)]
struct AttributeSample {
    fields: Vec<AttributeField>,
    truncated: bool,
}

impl Unobserved for AttributeSample {
    fn unobserved() -> Self {
        Self {
            fields: Vec::new(),
            truncated: true,
        }
    }
}

#[derive(Serialize)]
struct AttributeCatalog {
    table: TraceTable,
    column: &'static str,
    #[serde(flatten)]
    discovery: Discovery<AttributeSample>,
    discovery_sql: String,
    scope: &'static str,
}

#[derive(Serialize)]
struct NormalizedField {
    table: TraceTable,
    name: &'static str,
    column: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
    meaning: &'static str,
}

impl From<&NormalizedFieldDefinition> for NormalizedField {
    fn from(field: &NormalizedFieldDefinition) -> Self {
        Self {
            table: TraceTable::OtelTraces,
            name: field.name,
            column: field.clickhouse_column,
            kind: field.clickhouse_type,
            meaning: field.meaning,
        }
    }
}

#[derive(Serialize)]
struct Relationship {
    left: &'static str,
    right: &'static str,
    additional_predicates: &'static str,
    meaning: &'static str,
}

const RELATIONSHIPS: [Relationship; 1] = [Relationship {
    left: "otel_traces.LiteLLMRequestId",
    right: "spend_logs.response_id",
    additional_predicates: "otel_traces.TeamId = spend_logs.team_id AND (otel_traces.TeamId != '' OR (otel_traces.UserId != '' AND otel_traces.UserId = spend_logs.user) OR (otel_traces.ApiKeyHash != '' AND otel_traces.ApiKeyHash = spend_logs.api_key))",
    meaning: "The normalized ID is the response ID, not request_id. Cached requests can share response_id; joins may return multiple spend rows",
}];

#[derive(Serialize)]
pub struct QueryHelp {
    dialect: &'static str,
    access: &'static str,
    response: &'static str,
    tables: Vec<TableSchema>,
    normalized_fields: Vec<NormalizedField>,
    metadata: MetadataCatalog,
    attributes: Vec<AttributeCatalog>,
    relationships: &'static [Relationship],
    examples: [guide::Example; 5],
    gotchas: [String; 11],
    guide: String,
}

pub async fn execute_read(
    client: &Client,
    connection: &Connection,
    sql: &str,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error> {
    litellm_storage_clickhouse::execute_read(client, connection, sql, parameters)
        .await
        .map_err(Error::from)
}

pub async fn query_sql(
    client: &Client,
    connection: &Connection,
    sql: &str,
) -> Result<String, Error> {
    execute_read(client, connection, sql, &BTreeMap::new()).await
}

async fn rows<T: serde::de::DeserializeOwned>(
    client: &Client,
    connection: &Connection,
    sql: &str,
) -> Result<Vec<T>, Error> {
    let body = query_sql(client, connection, sql).await?;
    serde_json::from_str::<Rows<T>>(&body)
        .map(|result| result.data)
        .map_err(|_| Error::InvalidResponse)
}

fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn metadata_expression(path: &[PathPart]) -> String {
    let arguments = path
        .iter()
        .map(|part| match part {
            PathPart::Key(key) => literal(key),
            PathPart::Index(index) => index.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("JSONExtractRaw(metadata, {arguments})")
}

fn discover(
    value: &Value,
    path: Vec<PathPart>,
    fields: &mut BTreeMap<Vec<PathPart>, BTreeSet<JsonKind>>,
) -> bool {
    if path.len() > MAX_DEPTH || (fields.len() >= MAX_FIELDS && !fields.contains_key(&path)) {
        return true;
    }
    if !path.is_empty() {
        fields
            .entry(path.clone())
            .or_default()
            .insert(JsonKind::of(value));
    }
    match value {
        Value::Object(object) => object.iter().fold(false, |limited, (key, value)| {
            let child = path
                .iter()
                .cloned()
                .chain([PathPart::Key(key.clone())])
                .collect();
            discover(value, child, fields) | limited
        }),
        Value::Array(array) => array
            .iter()
            .enumerate()
            .fold(false, |limited, (index, value)| {
                let child = path
                    .iter()
                    .cloned()
                    .chain([PathPart::Index(index + 1)])
                    .collect();
                discover(value, child, fields) | limited
            }),
        _ => false,
    }
}

fn metadata_sample(sample: &[MetadataRow]) -> MetadataSample {
    let (fields, limited, invalid_rows) = sample.iter().take(SAMPLE_ROWS).fold(
        (BTreeMap::new(), sample.len() > SAMPLE_ROWS, 0),
        |(fields, limited, invalid_rows), row| match serde_json::from_str::<Value>(&row.metadata) {
            Ok(value) => {
                let mut fields = fields;
                let limited = limited | discover(&value, Vec::new(), &mut fields);
                (fields, limited, invalid_rows)
            }
            Err(_) => (fields, limited, invalid_rows + 1),
        },
    );
    let fields: Vec<_> = fields
        .into_iter()
        .map(|(path, types)| MetadataField {
            expression: metadata_expression(&path),
            path,
            types,
        })
        .collect();
    MetadataSample {
        fields,
        sampled_rows: sample.len().min(SAMPLE_ROWS),
        invalid_json_rows: invalid_rows,
        truncated: limited,
    }
}

pub async fn query_help(client: &Client, connection: &Connection) -> Result<QueryHelp, Error> {
    let tables = stream::iter(TraceTable::iter())
        .then(|table| async move {
            Ok::<_, Error>(TableSchema {
                name: table,
                columns: rows::<ColumnSchema>(
                    client,
                    connection,
                    &format!("DESCRIBE TABLE {table}"),
                )
                .await?,
            })
        })
        .try_collect::<Vec<_>>()
        .await?;
    let metadata = MetadataCatalog {
        table: TraceTable::SpendLogs,
        column: "metadata",
        discovery: match rows::<MetadataRow>(client, connection, METADATA_SQL).await {
            Ok(sample) => Discovery::Observed(metadata_sample(&sample)),
            Err(error) => Discovery::Unavailable(error.to_string()),
        },
        sample_sql: METADATA_SQL,
        scope: METADATA_SCOPE,
    };
    let attributes = stream::iter(["SpanAttributes", "ResourceAttributes"])
        .then(|column| async move {
            let sql = format!(
                "SELECT DISTINCT arrayJoin(mapKeys({column})) AS key FROM \
             (SELECT {column} FROM otel_traces WHERE Timestamp >= now() - INTERVAL 7 DAY \
             LIMIT 200) ORDER BY key LIMIT 201"
            );
            let discovery = match rows::<AttributeRow>(client, connection, &sql).await {
                Ok(keys) => Discovery::Observed(AttributeSample {
                    truncated: keys.len() > MAX_FIELDS,
                    fields: keys
                        .into_iter()
                        .take(MAX_FIELDS)
                        .map(|row| AttributeField {
                            expression: format!("{column}[{}]", literal(&row.key)),
                            key: row.key,
                            kind: MapValueType::String,
                        })
                        .collect(),
                }),
                Err(error) => Discovery::Unavailable(error.to_string()),
            };
            AttributeCatalog {
                table: TraceTable::OtelTraces,
                column,
                discovery,
                discovery_sql: sql,
                scope: ATTRIBUTE_SCOPE,
            }
        })
        .collect::<Vec<_>>()
        .await;
    let guide = guide::QueryGuide {
        tables: &tables,
        normalized_fields: &NORMALIZED_FIELD_DEFINITIONS,
        metadata: &metadata,
        attributes: &attributes,
        limits: &READER_LIMITS,
    };
    Ok(QueryHelp {
        dialect: "ClickHouse SQL",
        access: "Request-log visibility enforced by ClickHouse row policies; proxy admins see all rows, users see their own rows and permitted teams",
        response: "ClickHouse JSON envelope: meta, data, rows, statistics; 64-bit integers may be strings",
        examples: guide.examples()?,
        gotchas: guide.gotchas()?,
        guide: guide::render(&guide)?,
        normalized_fields: NORMALIZED_FIELD_DEFINITIONS
            .iter()
            .map(NormalizedField::from)
            .collect(),
        relationships: &RELATIONSHIPS,
        tables,
        metadata,
        attributes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    fn metadata_discovery_preserves_mixed_types_and_reports_invalid_rows() {
        let sample = [
            MetadataRow {
                metadata: r#"{"x": 1}"#.into(),
            },
            MetadataRow {
                metadata: r#"{"x": "one"}"#.into(),
            },
            MetadataRow {
                metadata: "invalid".into(),
            },
        ];
        let catalog = json!(metadata_sample(&sample));
        assert_eq!(
            catalog["fields"],
            json!([{
                "path": ["x"], "types": ["integer", "string"], "expression": "JSONExtractRaw(metadata, 'x')"
            }])
        );
        assert_eq!(catalog["invalid_json_rows"], 1);
        assert_eq!(catalog["sampled_rows"], sample.len());
    }

    #[rstest]
    #[case::rows(SAMPLE_ROWS + 1, 1)]
    #[case::paths(1, MAX_FIELDS + 1)]
    fn metadata_discovery_reports_truncation(#[case] row_count: usize, #[case] field_count: usize) {
        let metadata: BTreeMap<_, _> = (0..field_count)
            .map(|index| (format!("field{index}"), index))
            .collect();
        let sample: Vec<_> = (0..row_count)
            .map(|_| MetadataRow {
                metadata: json!(metadata).to_string(),
            })
            .collect();
        let catalog = json!(metadata_sample(&sample));
        assert_eq!(catalog["truncated"], true);
        assert_eq!(catalog["sampled_rows"], row_count.min(SAMPLE_ROWS));
        assert_eq!(
            catalog["fields"].as_array().unwrap().len(),
            field_count.min(MAX_FIELDS)
        );
    }
}
