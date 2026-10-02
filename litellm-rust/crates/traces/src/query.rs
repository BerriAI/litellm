use std::collections::{BTreeMap, BTreeSet};

use futures_util::{
    StreamExt,
    stream::{self, TryStreamExt},
};
use litellm_http::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Connection, Error, NORMALIZED_FIELD_DEFINITIONS, execute_read};

mod guide;

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

#[derive(Serialize)]
struct MetadataField {
    path: Vec<PathPart>,
    types: BTreeSet<&'static str>,
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
    name: &'static str,
    columns: Vec<ColumnSchema>,
}

#[derive(Serialize)]
struct MetadataCatalog {
    table: &'static str,
    column: &'static str,
    fields: Vec<MetadataField>,
    sampled_rows: usize,
    invalid_json_rows: usize,
    truncated: bool,
    sample_sql: &'static str,
    scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct AttributeField {
    key: String,
    #[serde(rename = "type")]
    kind: &'static str,
    expression: String,
}

#[derive(Serialize)]
struct AttributeCatalog {
    table: &'static str,
    column: &'static str,
    fields: Vec<AttributeField>,
    truncated: bool,
    discovery_sql: String,
    scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
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
    fields: &mut BTreeMap<Vec<PathPart>, BTreeSet<&'static str>>,
) -> bool {
    if path.len() > MAX_DEPTH || (fields.len() >= MAX_FIELDS && !fields.contains_key(&path)) {
        return true;
    }
    if !path.is_empty() {
        let kind = match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        fields.entry(path.clone()).or_default().insert(kind);
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

fn metadata_catalog(sample: &[MetadataRow]) -> MetadataCatalog {
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
    MetadataCatalog {
        table: "spend_logs",
        column: "metadata",
        fields,
        sampled_rows: sample.len().min(SAMPLE_ROWS),
        invalid_json_rows: invalid_rows,
        truncated: limited,
        sample_sql: METADATA_SQL,
        error: None,
        scope: METADATA_SCOPE,
    }
}

pub async fn query_help(client: &Client, connection: &Connection) -> Result<String, Error> {
    let tables = stream::iter(["otel_traces", "agent_traces_by_key", "spend_logs"])
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
    let metadata = match rows::<MetadataRow>(client, connection, METADATA_SQL).await {
        Ok(sample) => metadata_catalog(&sample),
        Err(error) => MetadataCatalog {
            error: Some(error.to_string()),
            truncated: true,
            ..metadata_catalog(&[])
        },
    };
    let attributes = stream::iter(["SpanAttributes", "ResourceAttributes"])
        .then(|column| async move {
            let sql = format!(
                "SELECT DISTINCT arrayJoin(mapKeys({column})) AS key FROM \
             (SELECT {column} FROM otel_traces WHERE Timestamp >= now() - INTERVAL 7 DAY \
             LIMIT 200) ORDER BY key LIMIT 201"
            );
            let (keys, error) = match rows::<AttributeRow>(client, connection, &sql).await {
                Ok(keys) => (keys, None),
                Err(error) => (Vec::new(), Some(error.to_string())),
            };
            let fields = keys
                .iter()
                .take(MAX_FIELDS)
                .map(|row| AttributeField {
                    key: row.key.clone(),
                    kind: "String",
                    expression: format!("{column}[{}]", literal(&row.key)),
                })
                .collect();
            AttributeCatalog {
                table: "otel_traces",
                column,
                fields,
                truncated: error.is_some() || keys.len() > MAX_FIELDS,
                discovery_sql: sql,
                scope: ATTRIBUTE_SCOPE,
                error,
            }
        })
        .collect::<Vec<_>>()
        .await;
    let guide = guide::QueryGuide {
        tables: &tables,
        normalized_fields: &NORMALIZED_FIELD_DEFINITIONS,
        metadata: &metadata,
        attributes: &attributes,
    };
    Ok(json!({
        "dialect": "ClickHouse SQL",
        "access": "Authenticated team scope enforced by ClickHouse row policies; proxy admins can read all teams, while project-bound and teamless keys can read only their own rows",
        "response": "ClickHouse JSON envelope: meta, data, rows, statistics; 64-bit integers may be strings",
        "tables": tables,
        "normalized_fields": NORMALIZED_FIELD_DEFINITIONS.iter().map(|field| json!({
            "table": "otel_traces", "name": field.name, "column": field.clickhouse_column,
            "type": field.clickhouse_type, "meaning": field.meaning
        })).collect::<Vec<_>>(),
        "metadata": metadata,
        "attributes": attributes,
        "relationships": [{
            "left": "otel_traces.LiteLLMRequestId", "right": "spend_logs.response_id",
            "additional_predicates": "otel_traces.TeamId = spend_logs.team_id AND otel_traces.ApiKeyHash = spend_logs.api_key",
            "meaning": "The normalized ID is the response ID, not request_id. Cached requests can share response_id; joins may return multiple spend rows"
        }],
        "examples": guide.examples()?,
        "gotchas": guide.gotchas()?,
        "guide": guide::render(&guide)?,
    }).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

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
        let catalog = json!(metadata_catalog(&sample));
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
        let catalog = json!(metadata_catalog(&sample));
        assert_eq!(catalog["truncated"], true);
        assert_eq!(catalog["sampled_rows"], row_count.min(SAMPLE_ROWS));
        assert_eq!(
            catalog["fields"].as_array().unwrap().len(),
            field_count.min(MAX_FIELDS)
        );
    }
}
