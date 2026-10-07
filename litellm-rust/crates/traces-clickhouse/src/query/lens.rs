use litellm_storage_clickhouse::{Query, ReadLimits};

const SAMPLE_READ_LIMITS: ReadLimits = ReadLimits {
    result_rows: 10_000,
    response_bytes: 16 * 1024 * 1024,
    ..litellm_storage_clickhouse::READ_LIMITS
};

pub const LENS_QUERIES: [litellm_traces::ReadQuery; 5] = [
    litellm_traces::ReadQuery::Availability,
    litellm_traces::ReadQuery::Agents,
    litellm_traces::ReadQuery::Sample,
    litellm_traces::ReadQuery::Content,
    litellm_traces::ReadQuery::Evidence,
];

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionSource {
    Traces,
    Requests,
    Both,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(rename_all = "lowercase")]
pub enum ContentSource {
    Traces,
    Requests,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
pub struct LensAccessParams {
    #[serde(
        deserialize_with = "super::number::boolean",
        serialize_with = "litellm_traces::wire::serialize_flag"
    )]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "litellm_traces::schema::flag")
    )]
    pub all_teams: bool,
    pub team: String,
    pub key_hash: String,
}

pub struct LensAvailability;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensAvailabilityParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "ActivityAvailability"))]
pub struct LensAvailabilityRow {
    #[serde(default, deserialize_with = "super::number::flag")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::boolean_flag")
    )]
    pub traces: u8,
    #[serde(default, deserialize_with = "super::number::flag")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::boolean_flag")
    )]
    pub requests: u8,
}

impl Query for LensAvailability {
    type Params = LensAvailabilityParams;
    type Row = LensAvailabilityRow;

    const SQL: &'static str = include_str!("../../query/lens_availability.sql");
}

pub struct LensAgents;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensAgentsParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "AgentRow"))]
pub struct LensAgentsRow {
    pub agent_name: String,
}

impl Query for LensAgents {
    type Params = LensAgentsParams;
    type Row = LensAgentsRow;

    const SQL: &'static str = include_str!("../../query/lens_agents.sql");
}

pub struct LensSample;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensSampleParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub source: ExecutionSource,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub end: u64,
    pub agent_name: String,
    pub service: String,
    pub filter_keys: Vec<String>,
    pub filter_values: Vec<String>,
    pub selected_team: String,
    pub execution_ids: Vec<String>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub sample_cap: u64,
    #[serde(deserialize_with = "super::number::percent")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0, max = 100)))]
    pub sample_percent: f64,
    #[serde(deserialize_with = "super::number::flag")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "litellm_traces::schema::flag")
    )]
    pub preview: u8,
    pub after: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub limit: u32,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub offset: u64,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "ExecutionRow"))]
pub struct LensSampleRow {
    pub source: ContentSource,
    pub trace_id: String,
    pub team_id: String,
    #[serde(default)]
    pub trace_ref: String,
    pub name: String,
    pub start_time: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub span_count: u64,
    #[serde(deserialize_with = "super::number::flag")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::flag_number")
    )]
    pub root_seen: u8,
    #[serde(default)]
    pub service: String,
    #[serde(default)]
    pub attributes: Vec<(String, String)>,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub eligible: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub position: u64,
    #[serde(default, deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::selected")
    )]
    pub selected: f64,
    #[serde(default)]
    pub selection_key: String,
}

impl Query for LensSample {
    type Params = LensSampleParams;
    type Row = LensSampleRow;

    const READ_LIMITS: ReadLimits = SAMPLE_READ_LIMITS;
    const SQL: &'static str = include_str!("../../query/lens_sample.sql");
}

pub struct LensContent;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensContentParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub source: ContentSource,
    pub id: String,
    pub record_team: String,
    pub start_time: String,
    pub trace_ref: String,
    pub cursor: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub offset: u32,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "PartRow"))]
pub struct LensContentRow {
    pub span_id: String,
    pub parent_span_id: String,
    pub name: String,
    pub kind: String,
    pub start_time: String,
    pub end_time: String,
    pub content: String,
    #[serde(deserialize_with = "super::number::flag")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::flag_number")
    )]
    pub truncated: u8,
}

impl Query for LensContent {
    type Params = LensContentParams;
    type Row = LensContentRow;

    const SQL: &'static str = include_str!("../../query/lens_content.sql");
}

pub struct LensEvidence;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensEvidenceParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub source: ContentSource,
    pub id: String,
    pub record_team: String,
    pub start_time: String,
    pub trace_ref: String,
    pub span: String,
    pub quote: String,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "CountRow"))]
pub struct LensEvidenceRow {
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub count: u64,
}

impl Query for LensEvidence {
    type Params = LensEvidenceParams;
    type Row = LensEvidenceRow;

    const SQL: &'static str = include_str!("../../query/lens_evidence.sql");
}
