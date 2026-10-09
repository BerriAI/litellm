use std::collections::{BTreeMap, BTreeSet};

use crate::TraceTable;
use futures_util::{
    StreamExt,
    stream::{self, TryStreamExt},
};
use litellm_http::Client;
use litellm_traces::query::guide::{Example, QueryGuide, Section};
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

#[macro_rules_attribute::apply(crate::request_type)]
struct AttributeRow {
    key: String,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[serde(untagged)]
enum PathPart {
    Key(String),
    Index(usize),
}

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, strum::Display)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "schema", schemars(rename = "MetadataValueType"))]
enum JsonKind {
    #[strum(serialize = "array")]
    Array,
    #[strum(serialize = "boolean")]
    Boolean,
    #[strum(serialize = "integer")]
    Integer,
    #[strum(serialize = "null")]
    Null,
    #[strum(serialize = "number")]
    Number,
    #[strum(serialize = "object")]
    Object,
    #[strum(serialize = "string")]
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

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Clone, Copy, Debug, strum::Display)]
enum MapValueType {
    String,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryMetadataField"))]
struct MetadataField {
    path: Vec<PathPart>,
    types: BTreeSet<JsonKind>,
    expression: String,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryColumn"))]
struct ColumnSchema {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(flatten)]
    details: BTreeMap<String, Value>,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryTable"))]
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

#[cfg(feature = "schema")]
impl<T: schemars::JsonSchema> schemars::JsonSchema for Discovery<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("Discovery{}", T::schema_name()).into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut schema = T::json_schema(generator);
        schema
            .as_object_mut()
            .unwrap()
            .get_mut("properties")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "error".into(),
                serde_json::json!({"type": ["string", "null"], "default": null}),
            );
        schema
    }
}

#[cfg(feature = "schema")]
pub(crate) fn help_schema() -> schemars::Schema {
    schemars::generate::SchemaSettings::draft2020_12()
        .for_serialize()
        .with_transform(litellm_traces::schema::integer_bounds)
        .into_generator()
        .into_root_schema_for::<QueryHelp>()
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

#[macro_rules_attribute::apply(crate::response_type)]
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

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryMetadata"))]
struct MetadataCatalog {
    table: TraceTable,
    column: &'static str,
    #[serde(flatten)]
    discovery: Discovery<MetadataSample>,
    sample_sql: &'static str,
    scope: &'static str,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryAttributeField"))]
struct AttributeField {
    key: String,
    #[serde(rename = "type")]
    kind: MapValueType,
    expression: String,
}

#[macro_rules_attribute::apply(crate::response_type)]
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

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryAttributes"))]
struct AttributeCatalog {
    table: TraceTable,
    column: &'static str,
    #[serde(flatten)]
    discovery: Discovery<AttributeSample>,
    discovery_sql: String,
    scope: &'static str,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryNormalizedField"))]
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

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryRelationship"))]
struct Relationship {
    left: &'static str,
    right: &'static str,
    additional_predicates: &'static str,
    meaning: &'static str,
}

const RELATIONSHIPS: [Relationship; 1] = [Relationship {
    left: "otel_traces.LiteLLMRequestId",
    right: "spend_logs.response_id",
    additional_predicates: "otel_traces.TeamId = spend_logs.team_id AND ((otel_traces.UserId != '' AND otel_traces.UserId = spend_logs.user) OR (otel_traces.ApiKeyHash != '' AND otel_traces.ApiKeyHash = spend_logs.api_key))",
    meaning: "LiteLLMRequestId contains the first normalized request or provider response ID. This relationship matches response IDs only; CallKeys retains all typed identifiers. Cached requests can share response_id; joins may return multiple spend rows",
}];

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryHelp"))]
pub struct QueryHelp {
    dialect: &'static str,
    access: &'static str,
    response: &'static str,
    tables: Vec<TableSchema>,
    normalized_fields: Vec<NormalizedField>,
    metadata: MetadataCatalog,
    attributes: Vec<AttributeCatalog>,
    relationships: &'static [Relationship],
    #[cfg_attr(feature = "schema", schemars(with = "Vec<Example>"))]
    examples: [Example; 12],
    #[cfg_attr(feature = "schema", schemars(with = "Vec<String>"))]
    gotchas: [String; 13],
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
    let bodies = guide.sections()?;
    let sections = [
        "Live ClickHouse schema",
        "Normalized span fields",
        "Observed LLM call metadata",
        "Observed span and resource attributes",
    ]
    .into_iter()
    .zip(&bodies)
    .map(|(title, body)| Section { title, body })
    .collect::<Vec<_>>();
    let examples = guide.examples()?;
    let gotchas = guide.gotchas()?;
    let rendered = QueryGuide {
        sections: &sections,
        examples: &examples,
        gotchas: &gotchas,
    }
    .render()
    .map_err(|_| Error::InvalidResponse)?;
    Ok(QueryHelp {
        dialect: "ClickHouse SQL",
        access: "Request-log visibility enforced by ClickHouse row policies; proxy admins see all rows, users see their own rows and permitted teams",
        response: "JSON object {\"data\": [rows]}; each row maps selected columns to values; 64-bit integers may be strings",
        examples,
        gotchas,
        guide: rendered,
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

    #[cfg(feature = "schema")]
    #[rstest]
    #[case::observed(false)]
    #[case::unavailable(true)]
    fn discovery_serialization_matches_its_schema(#[case] unavailable: bool) {
        let discovery = if unavailable {
            Discovery::Unavailable("discovery failed".into())
        } else {
            Discovery::Observed(MetadataSample::unobserved())
        };
        let catalog = MetadataCatalog {
            table: TraceTable::SpendLogs,
            column: "metadata",
            discovery,
            sample_sql: METADATA_SQL,
            scope: METADATA_SCOPE,
        };
        let schema = schemars::generate::SchemaSettings::draft2020_12()
            .for_serialize()
            .into_generator()
            .into_root_schema_for::<MetadataCatalog>();
        let serialized = serde_json::to_value(&catalog).unwrap();
        assert!(jsonschema::is_valid(schema.as_value(), &serialized));
        assert_eq!(serialized.get("error").is_some(), unavailable);
        assert!(serialized["fields"].is_array());
    }

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
