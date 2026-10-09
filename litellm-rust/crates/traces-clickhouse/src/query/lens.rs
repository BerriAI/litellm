use litellm_storage_clickhouse::{Query, ReadLimits};

const SAMPLE_READ_LIMITS: ReadLimits = ReadLimits {
    result_rows: 10_000,
    response_bytes: 16 * 1024 * 1024,
    ..litellm_storage_clickhouse::READ_LIMITS
};

pub const LENS_QUERIES: [litellm_traces::ReadQuery; 9] = [
    litellm_traces::ReadQuery::TraceAgents,
    litellm_traces::ReadQuery::Availability,
    litellm_traces::ReadQuery::Agents,
    litellm_traces::ReadQuery::Sample,
    litellm_traces::ReadQuery::Content,
    litellm_traces::ReadQuery::Evidence,
    litellm_traces::ReadQuery::FeedbackTarget,
    litellm_traces::ReadQuery::Feedback,
    litellm_traces::ReadQuery::FeedbackSummary,
];

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionSource {
    Traces,
    Requests,
    Both,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(rename_all = "lowercase")]
pub enum ContentSource {
    Traces,
    Requests,
}

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensAvailabilityParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
}

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensAgentsParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
}

#[macro_rules_attribute::apply(crate::wire_type)]
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

pub struct TraceAgents;

/// Same access shape as `list_traces`: every team, the caller's own traces, or their teams' traces.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
pub struct TraceAgentsParams {
    #[serde(
        deserialize_with = "super::number::boolean",
        serialize_with = "litellm_traces::wire::serialize_flag"
    )]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "litellm_traces::schema::flag")
    )]
    pub all_teams: bool,
    pub user_id: String,
    pub team_ids: Vec<String>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub end_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub limit: u32,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "TraceAgentRow"))]
pub struct TraceAgentsRow {
    pub agent_name: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub runs: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub failed_runs: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub last_seen_ms: u64,
    #[serde(default)]
    pub frameworks: Vec<String>,
}

impl Query for TraceAgents {
    type Params = TraceAgentsParams;
    type Row = TraceAgentsRow;

    const SQL: &'static str = include_str!("../../query/trace_agents.sql");
}

pub struct LensSample;

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

pub struct LensFeedbackTarget;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensFeedbackTargetParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "FeedbackTargetRow"))]
pub struct LensFeedbackTargetRow {
    pub team_id: String,
    pub key_hash: String,
    pub trace_ref: String,
}

impl Query for LensFeedbackTarget {
    type Params = LensFeedbackTargetParams;
    type Row = LensFeedbackTargetRow;

    const SQL: &'static str = include_str!("../../query/lens_feedback_target.sql");
}

pub struct LensFeedback;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensFeedbackParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "FeedbackRow"))]
pub struct LensFeedbackRow {
    pub trace_id: String,
    pub trace_ref: String,
    pub author: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub score: u64,
    pub comment: String,
    pub created_at: String,
    pub updated_at: String,
}

impl Query for LensFeedback {
    type Params = LensFeedbackParams;
    type Row = LensFeedbackRow;

    const SQL: &'static str = include_str!("../../query/lens_feedback.sql");
}

pub struct LensFeedbackSummary;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[serde(deny_unknown_fields)]
pub struct LensFeedbackSummaryParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub trace_ids: Vec<String>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "FeedbackSummaryRow"))]
pub struct LensFeedbackSummaryRow {
    pub trace_id: String,
    pub trace_ref: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub count: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub average: f64,
    #[serde(deserialize_with = "super::number::deserialize")]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "crate::wire_schema::u64_number")
    )]
    pub lowest: u64,
}

impl Query for LensFeedbackSummary {
    type Params = LensFeedbackSummaryParams;
    type Row = LensFeedbackSummaryRow;

    const SQL: &'static str = include_str!("../../query/lens_feedback_summary.sql");
}
