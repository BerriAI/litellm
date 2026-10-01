use std::collections::{BTreeMap, BTreeSet};

use litellm_http::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Connection, Error, NORMALIZED_FIELD_DEFINITIONS, execute_read};

const SAMPLE_ROWS: usize = 200;
const MAX_FIELDS: usize = 200;
const MAX_DEPTH: usize = 16;
const METADATA_SQL: &str = "SELECT metadata FROM spend_logs \
    WHERE start_time >= now() - INTERVAL 7 DAY AND length(metadata) <= 8192 \
    ORDER BY start_time DESC LIMIT 201";

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

fn metadata_catalog(sample: &[MetadataRow]) -> Value {
    let mut fields = BTreeMap::new();
    let mut limited = sample.len() > SAMPLE_ROWS;
    let mut invalid_rows = 0;
    for row in sample.iter().take(SAMPLE_ROWS) {
        match serde_json::from_str::<Value>(&row.metadata) {
            Ok(value) => limited |= discover(&value, Vec::new(), &mut fields),
            Err(_) => invalid_rows += 1,
        }
    }
    let fields: Vec<_> = fields
        .into_iter()
        .map(|(path, types)| MetadataField {
            expression: metadata_expression(&path),
            path,
            types,
        })
        .collect();
    json!({
        "table": "spend_logs", "column": "metadata", "fields": fields,
        "sampled_rows": sample.len().min(SAMPLE_ROWS), "invalid_json_rows": invalid_rows,
        "truncated": limited, "sample_sql": METADATA_SQL,
        "scope": "Up to 200 recent rows from the last 7 days, excluding metadata larger than 8192 bytes; up to 200 paths and 16 levels. Missing paths may exist outside this sample. Array indexes are 1-based and describe sampled positions, not a fixed schema"
    })
}

pub async fn query_help(client: &Client, connection: &Connection) -> Result<String, Error> {
    let mut tables = Vec::new();
    for table in ["otel_traces", "agent_traces_by_key", "spend_logs"] {
        let columns = rows::<Value>(client, connection, &format!("DESCRIBE TABLE {table}")).await?;
        tables.push(json!({"name": table, "columns": columns}));
    }
    let sample = rows::<MetadataRow>(client, connection, METADATA_SQL).await?;
    let mut attributes = Vec::new();
    for column in ["SpanAttributes", "ResourceAttributes"] {
        let sql = format!(
            "SELECT DISTINCT arrayJoin(mapKeys({column})) AS key FROM \
             (SELECT {column} FROM otel_traces WHERE Timestamp >= now() - INTERVAL 7 DAY \
             ORDER BY Timestamp DESC LIMIT 200) ORDER BY key LIMIT 201"
        );
        let keys = rows::<AttributeRow>(client, connection, &sql).await?;
        let fields: Vec<_> = keys.iter().take(MAX_FIELDS).map(|row| json!({
            "key": row.key, "type": "String", "expression": format!("{column}[{}]", literal(&row.key))
        })).collect();
        attributes.push(json!({
            "table": "otel_traces", "column": column, "fields": fields,
            "truncated": keys.len() > MAX_FIELDS, "discovery_sql": sql,
            "scope": "Distinct keys from up to 200 recent spans in the last 7 days; up to 200 keys per map. Missing keys may exist outside this sample"
        }));
    }
    Ok(json!({
        "dialect": "ClickHouse SQL",
        "access": "Proxy admin only; reads all teams using CLICKHOUSE_READER_URL with SELECT-only grants",
        "response": "ClickHouse JSON envelope: meta, data, rows, statistics; 64-bit integers may be strings",
        "tables": tables,
        "normalized_fields": NORMALIZED_FIELD_DEFINITIONS.iter().map(|field| json!({
            "table": "otel_traces", "name": field.name, "column": field.clickhouse_column,
            "type": field.clickhouse_type, "meaning": field.meaning
        })).collect::<Vec<_>>(),
        "metadata": metadata_catalog(&sample),
        "attributes": attributes,
        "relationships": [{
            "left": "otel_traces.LiteLLMRequestId", "right": "spend_logs.response_id",
            "additional_predicates": "otel_traces.TeamId = spend_logs.team_id AND otel_traces.ApiKeyHash = spend_logs.api_key",
            "meaning": "The normalized ID is the response ID, not request_id. Cached requests can share response_id; joins may return multiple spend rows"
        }],
        "examples": [
            {"name": "Recent normalized LLM spans", "sql": "SELECT TraceId, SpanId, Model, InputTokens, OutputTokens, Duration / 1000000 AS duration_ms FROM otel_traces WHERE Timestamp >= now() - INTERVAL 1 DAY AND ObservationType = 'llm' ORDER BY Timestamp DESC LIMIT 100"},
            {"name": "Find calls by custom metadata", "sql": "SELECT request_id, response_id, model, spend, JSONExtractString(metadata, 'project') AS project FROM spend_logs FINAL WHERE start_time >= now() - INTERVAL 1 DAY AND JSONHas(metadata, 'project') AND JSONExtractString(metadata, 'project') = 'example' ORDER BY start_time DESC LIMIT 100"},
            {"name": "Nested metadata with unknown types", "sql": "SELECT request_id, JSONType(metadata, 'labels', 'priority') AS type, JSONExtractRaw(metadata, 'labels', 'priority') AS value FROM spend_logs FINAL WHERE start_time >= now() - INTERVAL 1 DAY AND JSONHas(metadata, 'labels', 'priority') LIMIT 100"},
            {"name": "Traces correlated with LLM call metadata", "sql": "SELECT t.TraceId, t.SpanId, s.request_id, s.spend, s.metadata FROM otel_traces AS t INNER JOIN (SELECT * FROM spend_logs FINAL WHERE start_time >= now() - INTERVAL 1 DAY) AS s ON t.LiteLLMRequestId = s.response_id AND t.TeamId = s.team_id AND t.ApiKeyHash = s.api_key WHERE t.Timestamp >= now() - INTERVAL 1 DAY AND t.LiteLLMRequestId != '' AND JSONExtractString(s.metadata, 'project') = 'example' LIMIT 100"},
            {"name": "Discover metadata keys over a different window", "sql": "SELECT DISTINCT arrayJoin(JSONExtractKeys(metadata)) AS key FROM spend_logs WHERE start_time >= now() - INTERVAL 30 DAY ORDER BY key LIMIT 200"}
        ],
        "gotchas": [
            "Always bound Timestamp or start_time and use LIMIT; add TeamId/ApiKeyHash or team_id/api_key filters when investigating one tenant",
            "The reader enforces 1000 result rows, 4 MiB response bytes and a 10 second query limit; exceeding limits fails instead of returning partial results",
            "Use the SELECT-only reader profile in crates/traces/config/reader.xml; never configure administrator credentials as the reader",
            "Do not add FORMAT clauses; the endpoint requires ClickHouse JSON output",
            "metadata is a JSON-encoded String; use JSONHas before typed extraction to distinguish missing values from empty strings, zero and false",
            "SpanAttributes and ResourceAttributes are Map(String, String); missing map keys return an empty string, so use mapContains for existence checks",
            "Use the discovered path components as separate JSONExtract arguments; a dot inside a key is literal, not a path separator",
            "Duration is nanoseconds; Timestamp has nanosecond precision, spend start_time has millisecond precision",
            "Use spend_logs FINAL to collapse replacement rows before totals. Shared response IDs and multiple spans can multiply costs in joins; aggregate spend separately",
            "agent_traces_by_key uses SimpleAggregateFunction columns; group by TeamId, ApiKeyHash and TraceId, using min(StartTs), max(EndTs), sum(SpanCount) and groupUniqArrayArray(Models). Do not use Merge combinators",
            "Discovery is sampled, contains no metadata values, and is not an exhaustive schema. Edit the supplied discovery SQL for older data or nested JSONExtractKeys(metadata, 'parent')"
        ]
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
        let catalog = metadata_catalog(&sample);
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
        let catalog = metadata_catalog(&sample);
        assert_eq!(catalog["truncated"], true);
        assert_eq!(catalog["sampled_rows"], row_count.min(SAMPLE_ROWS));
        assert_eq!(
            catalog["fields"].as_array().unwrap().len(),
            field_count.min(MAX_FIELDS)
        );
    }
}
