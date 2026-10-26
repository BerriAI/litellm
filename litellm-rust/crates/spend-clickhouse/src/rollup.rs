use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_storage_clickhouse::{Connection, Parameter, READ_LIMITS, execute_read};
use serde::{Deserialize, Deserializer, Serialize};

use crate::Error;

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    strum::EnumString,
    strum::Display,
    strum::IntoStaticStr,
    strum::VariantArray,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
pub enum Dimension {
    #[strum(serialize = "tag")]
    Tag,
    #[strum(serialize = "api_key")]
    ApiKey,
    #[strum(serialize = "team_id")]
    TeamId,
    #[strum(serialize = "organization_id")]
    OrganizationId,
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "end_user")]
    EndUser,
    #[strum(serialize = "model")]
    Model,
    #[strum(serialize = "model_group")]
    ModelGroup,
    #[strum(serialize = "custom_llm_provider")]
    Provider,
    #[strum(serialize = "status")]
    Status,
    #[strum(serialize = "call_type")]
    CallType,
}

impl Dimension {
    fn column(self) -> &'static str {
        self.into()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::EnumString,
    strum::Display,
    strum::IntoStaticStr,
    strum::VariantArray,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
pub enum Bucket {
    #[strum(serialize = "1m")]
    Minute,
    #[strum(serialize = "5m")]
    FiveMinutes,
    #[strum(serialize = "15m")]
    FifteenMinutes,
    #[strum(serialize = "1h")]
    Hour,
    #[strum(serialize = "1d")]
    Day,
}

impl Bucket {
    fn expression(self) -> &'static str {
        match self {
            Self::Minute => "toStartOfMinute(start_time, 'UTC')",
            Self::FiveMinutes => "toStartOfInterval(start_time, INTERVAL 5 MINUTE, 'UTC')",
            Self::FifteenMinutes => "toStartOfInterval(start_time, INTERVAL 15 MINUTE, 'UTC')",
            Self::Hour => "toStartOfHour(start_time, 'UTC')",
            Self::Day => "toStartOfDay(start_time, 'UTC')",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RollupRequest {
    pub start_ms: i64,
    pub end_ms: i64,
    #[serde(default)]
    pub bucket: Option<Bucket>,
    #[serde(default)]
    pub group_by: Vec<Dimension>,
    #[serde(default)]
    pub filters: BTreeMap<Dimension, Vec<String>>,
    #[serde(default = "default_limit")]
    pub limit: u64,
}

fn default_limit() -> u64 {
    READ_LIMITS.result_rows
}

fn u64_from_json<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Number(u64),
        Text(String),
    }
    match Raw::deserialize(deserializer)? {
        Raw::Number(value) => Ok(value),
        Raw::Text(value) => value.parse().map_err(serde::de::Error::custom),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RollupRow {
    #[serde(default)]
    pub bucket: Option<String>,
    #[serde(flatten)]
    pub dimensions: BTreeMap<Dimension, String>,
    pub spend: f64,
    #[serde(deserialize_with = "u64_from_json")]
    pub prompt_tokens: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub completion_tokens: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub cache_read_input_tokens: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub cache_creation_input_tokens: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub api_requests: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub successful_requests: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub failed_requests: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub total_response_time_ms: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub timed_requests: u64,
    #[serde(deserialize_with = "u64_from_json")]
    pub distinct_api_keys: u64,
}

const METRICS: &str = "\
sum(coalesce(spend, 0)) AS spend, \
toUInt64(sum(prompt_tokens)) AS prompt_tokens, \
toUInt64(sum(completion_tokens)) AS completion_tokens, \
toUInt64(sum(cache_read_tokens)) AS cache_read_input_tokens, \
toUInt64(sum(cache_write_tokens)) AS cache_creation_input_tokens, \
count() AS api_requests, \
countIf(status = 'success') AS successful_requests, \
countIf(status != 'success') AS failed_requests, \
toUInt64(sum(greatest(dateDiff('millisecond', start_time, end_time), 0))) AS total_response_time_ms, \
countIf(end_time > start_time) AS timed_requests, \
uniqExact(api_key) AS distinct_api_keys";

#[derive(Debug)]
pub struct RollupQuery {
    sql: String,
    parameters: BTreeMap<String, Parameter>,
}

impl RollupQuery {
    pub fn build(request: &RollupRequest) -> Result<Self, Error> {
        if request.end_ms <= request.start_ms {
            return Err(Error::InvalidRollup("end_ms must be after start_ms"));
        }
        if !(1..=READ_LIMITS.result_rows).contains(&request.limit) {
            return Err(Error::InvalidRollup(
                "limit must be between 1 and the read row limit",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        if request
            .group_by
            .iter()
            .any(|dimension| !seen.insert(dimension))
        {
            return Err(Error::InvalidRollup("group_by must not repeat a dimension"));
        }
        if request.filters.values().any(Vec::is_empty) {
            return Err(Error::InvalidRollup("filters must list at least one value"));
        }

        let mut selected: Vec<String> = Vec::new();
        let mut keys: Vec<&'static str> = Vec::new();
        if let Some(bucket) = request.bucket {
            selected.push(format!(
                "formatDateTime({}, '%Y-%m-%dT%H:%i:%S', 'UTC') AS bucket",
                bucket.expression()
            ));
            keys.push("bucket");
        }
        for dimension in &request.group_by {
            let column = dimension.column();
            selected.push(column.to_owned());
            keys.push(column);
        }
        selected.push(METRICS.to_owned());

        let needs_tags = request.group_by.contains(&Dimension::Tag)
            || request.filters.contains_key(&Dimension::Tag);
        let source = if needs_tags {
            "spend_logs ARRAY JOIN request_tags AS tag"
        } else {
            "spend_logs"
        };

        let mut parameters = BTreeMap::from([
            ("start_ms".to_owned(), Parameter::Integer(request.start_ms)),
            ("end_ms".to_owned(), Parameter::Integer(request.end_ms)),
            ("limit".to_owned(), Parameter::Unsigned(request.limit)),
        ]);
        let mut conditions = vec![
            "start_time >= fromUnixTimestamp64Milli({start_ms:Int64})".to_owned(),
            "start_time < fromUnixTimestamp64Milli({end_ms:Int64})".to_owned(),
        ];
        for (dimension, values) in &request.filters {
            let column = dimension.column();
            let name = format!("filter_{column}");
            conditions.push(format!("{column} IN {{{name}:Array(String)}}"));
            parameters.insert(name, Parameter::Strings(values.clone()));
        }

        let group_by = if keys.is_empty() {
            String::new()
        } else {
            format!(" GROUP BY {} ORDER BY {}", keys.join(", "), keys.join(", "))
        };
        let sql = format!(
            "SELECT {} FROM {source} WHERE {}{group_by} LIMIT {{limit:UInt64}}",
            selected.join(", "),
            conditions.join(" AND "),
        );
        Ok(Self { sql, parameters })
    }

    pub fn sql(&self) -> &str {
        &self.sql
    }

    pub fn parameters(&self) -> &BTreeMap<String, Parameter> {
        &self.parameters
    }
}

#[derive(Deserialize)]
struct Rows {
    data: Vec<RollupRow>,
}

pub fn decode_rollup_rows(body: &str) -> Result<Vec<RollupRow>, Error> {
    serde_json::from_str::<Rows>(body)
        .map(|rows| rows.data)
        .map_err(|_| Error::Storage(litellm_storage_clickhouse::Error::InvalidResponse))
}

pub async fn fetch_rollup(
    client: &Client,
    connection: &Connection,
    request: &RollupRequest,
) -> Result<Vec<RollupRow>, Error> {
    let query = RollupQuery::build(request)?;
    let body = execute_read(client, connection, query.sql(), query.parameters()).await?;
    decode_rollup_rows(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn request(bucket: Option<Bucket>, group_by: &[Dimension]) -> RollupRequest {
        RollupRequest {
            start_ms: 1_000,
            end_ms: 2_000,
            bucket,
            group_by: group_by.to_vec(),
            filters: BTreeMap::new(),
            limit: 1000,
        }
    }

    #[rstest]
    #[case::grand_total(
        None,
        &[],
        false,
        "SELECT ",
        " FROM spend_logs WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64}) AND start_time < fromUnixTimestamp64Milli({end_ms:Int64}) LIMIT {limit:UInt64}"
    )]
    #[case::hourly_tags(
        Some(Bucket::Hour),
        &[Dimension::Tag, Dimension::Model],
        true,
        "SELECT formatDateTime(toStartOfHour(start_time, 'UTC'), '%Y-%m-%dT%H:%i:%S', 'UTC') AS bucket, tag, model, ",
        " FROM spend_logs ARRAY JOIN request_tags AS tag WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64}) AND start_time < fromUnixTimestamp64Milli({end_ms:Int64}) GROUP BY bucket, tag, model ORDER BY bucket, tag, model LIMIT {limit:UInt64}"
    )]
    #[case::five_minute_keys(
        Some(Bucket::FiveMinutes),
        &[Dimension::ApiKey],
        false,
        "SELECT formatDateTime(toStartOfInterval(start_time, INTERVAL 5 MINUTE, 'UTC'), '%Y-%m-%dT%H:%i:%S', 'UTC') AS bucket, api_key, ",
        " FROM spend_logs WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64}) AND start_time < fromUnixTimestamp64Milli({end_ms:Int64}) GROUP BY bucket, api_key ORDER BY bucket, api_key LIMIT {limit:UInt64}"
    )]
    fn renders_bucket_and_dimensions(
        #[case] bucket: Option<Bucket>,
        #[case] group_by: &[Dimension],
        #[case] joins_tags: bool,
        #[case] prefix: &str,
        #[case] suffix: &str,
    ) {
        let query = RollupQuery::build(&request(bucket, group_by)).unwrap();
        assert_eq!(&query.sql()[..prefix.len()], prefix);
        assert!(query.sql().ends_with(suffix), "{}", query.sql());
        assert_eq!(query.sql().contains("ARRAY JOIN"), joins_tags);
        assert_eq!(query.sql().matches(METRICS).count(), 1);
        assert_eq!(
            query.parameters().keys().cloned().collect::<Vec<_>>(),
            ["end_ms", "limit", "start_ms"]
        );
    }

    #[rstest]
    fn binds_filters_as_array_parameters() {
        let mut request = request(Some(Bucket::Minute), &[Dimension::Model]);
        request
            .filters
            .insert(Dimension::Tag, vec!["prod".into(), "it's".into()]);
        request
            .filters
            .insert(Dimension::ApiKey, vec!["hashed".into()]);
        let query = RollupQuery::build(&request).unwrap();
        assert!(query.sql().contains("ARRAY JOIN request_tags AS tag"));
        assert!(
            query
                .sql()
                .contains("AND api_key IN {filter_api_key:Array(String)}")
        );
        assert!(
            query
                .sql()
                .contains("AND tag IN {filter_tag:Array(String)}")
        );
        assert!(matches!(
            &query.parameters()["filter_tag"],
            Parameter::Strings(values) if values == &["prod".to_owned(), "it's".to_owned()]
        ));
    }

    #[rstest]
    #[case::reversed_window(|r: &mut RollupRequest| r.end_ms = r.start_ms, "end_ms must be after start_ms")]
    #[case::zero_limit(|r: &mut RollupRequest| r.limit = 0, "limit must be between 1 and the read row limit")]
    #[case::limit_over_read_cap(|r: &mut RollupRequest| r.limit = READ_LIMITS.result_rows + 1, "limit must be between 1 and the read row limit")]
    #[case::repeated_dimension(|r: &mut RollupRequest| r.group_by.push(Dimension::Model), "group_by must not repeat a dimension")]
    #[case::empty_filter(|r: &mut RollupRequest| { r.filters.insert(Dimension::TeamId, vec![]); }, "filters must list at least one value")]
    fn rejects_invalid_requests(#[case] mutate: fn(&mut RollupRequest), #[case] message: &str) {
        let mut request = request(Some(Bucket::Hour), &[Dimension::Model]);
        mutate(&mut request);
        let error = RollupQuery::build(&request).unwrap_err();
        assert!(matches!(error, Error::InvalidRollup(reason) if reason == message));
    }

    #[rstest]
    #[case::wire_names(r#"{"start_ms":1,"end_ms":2,"bucket":"15m","group_by":["tag","custom_llm_provider"],"filters":{"team_id":["t1"]},"limit":5}"#, true)]
    #[case::unknown_dimension(r#"{"start_ms":1,"end_ms":2,"group_by":["request_id"]}"#, false)]
    #[case::unknown_bucket(r#"{"start_ms":1,"end_ms":2,"bucket":"2h"}"#, false)]
    #[case::unknown_field(r#"{"start_ms":1,"end_ms":2,"sql":"DROP TABLE spend_logs"}"#, false)]
    fn request_accepts_only_known_dimensions_and_buckets(#[case] json: &str, #[case] valid: bool) {
        let parsed = serde_json::from_str::<RollupRequest>(json);
        assert_eq!(parsed.is_ok(), valid, "{parsed:?}");
        if let Ok(request) = parsed {
            assert_eq!(request.bucket, Some(Bucket::FifteenMinutes));
            assert_eq!(request.group_by, [Dimension::Tag, Dimension::Provider]);
            assert_eq!(request.filters[&Dimension::TeamId], ["t1"]);
            assert_eq!(request.limit, 5);
        }
    }

    #[rstest]
    fn decodes_quoted_64bit_counters_and_dimension_columns() {
        let body = r#"{"meta":[],"data":[{"bucket":"2026-10-24T21:00:00","tag":"prod","model":"gpt-4o","spend":0.5,"prompt_tokens":"12","completion_tokens":3,"cache_read_input_tokens":"0","cache_creation_input_tokens":"0","api_requests":"2","successful_requests":"1","failed_requests":"1","total_response_time_ms":"340","timed_requests":"2","distinct_api_keys":"1"}],"rows":1}"#;
        let rows = decode_rollup_rows(body).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.bucket.as_deref(), Some("2026-10-24T21:00:00"));
        assert_eq!(row.dimensions[&Dimension::Tag], "prod");
        assert_eq!(row.dimensions[&Dimension::Model], "gpt-4o");
        assert_eq!(
            (row.prompt_tokens, row.completion_tokens, row.api_requests),
            (12, 3, 2)
        );
        assert_eq!(row.total_response_time_ms, 340);
        assert_eq!(
            serde_json::to_value(row).unwrap()["tag"],
            serde_json::json!("prod")
        );
        assert!(matches!(
            decode_rollup_rows(r#"{"data":[{"bucket":1}]}"#),
            Err(Error::Storage(
                litellm_storage_clickhouse::Error::InvalidResponse
            ))
        ));
    }
}
