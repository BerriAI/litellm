use litellm_storage_clickhouse::Query;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct LensAccessParams {
    #[serde(deserialize_with = "super::number::deserialize")]
    pub all_teams: u8,
    pub team: String,
    pub key_hash: String,
}

pub struct LensAvailability;

#[derive(Debug, Deserialize, Serialize)]
pub struct LensAvailabilityParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LensAvailabilityRow {
    #[serde(deserialize_with = "super::number::deserialize")]
    pub traces: u8,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub requests: u8,
}

impl Query for LensAvailability {
    type Params = LensAvailabilityParams;
    type Row = LensAvailabilityRow;

    const SQL: &'static str = include_str!("../../query/lens_availability.sql");
}

pub struct LensAgents;

#[derive(Debug, Deserialize, Serialize)]
pub struct LensAgentsParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LensAgentsRow {
    pub agent_name: String,
}

impl Query for LensAgents {
    type Params = LensAgentsParams;
    type Row = LensAgentsRow;

    const SQL: &'static str = include_str!("../../query/lens_agents.sql");
}

pub struct LensSample;

#[derive(Debug, Deserialize, Serialize)]
pub struct LensSampleParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub source: String,
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
    #[serde(deserialize_with = "super::number::deserialize")]
    pub sample_percent: f64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub preview: u8,
    pub after: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub limit: u32,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub offset: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LensSampleRow {
    pub source: String,
    pub trace_id: String,
    pub team_id: String,
    pub trace_ref: String,
    pub name: String,
    pub start_time: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub span_count: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub root_seen: u8,
    pub service: String,
    pub attributes: Vec<(String, String)>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub eligible: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub position: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub selected: f64,
    pub selection_key: String,
}

impl Query for LensSample {
    type Params = LensSampleParams;
    type Row = LensSampleRow;

    const SQL: &'static str = include_str!("../../query/lens_sample.sql");
}

pub struct LensContent;

#[derive(Debug, Deserialize, Serialize)]
pub struct LensContentParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub source: String,
    pub id: String,
    pub record_team: String,
    pub trace_ref: String,
    pub cursor: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub offset: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LensContentRow {
    pub span_id: String,
    pub parent_span_id: String,
    pub name: String,
    pub kind: String,
    pub content: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub truncated: u8,
}

impl Query for LensContent {
    type Params = LensContentParams;
    type Row = LensContentRow;

    const SQL: &'static str = include_str!("../../query/lens_content.sql");
}

pub struct LensEvidence;

#[derive(Debug, Deserialize, Serialize)]
pub struct LensEvidenceParams {
    #[serde(flatten)]
    pub access: LensAccessParams,
    pub source: String,
    pub id: String,
    pub record_team: String,
    pub trace_ref: String,
    pub span: String,
    pub quote: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LensEvidenceRow {
    #[serde(deserialize_with = "super::number::deserialize")]
    pub count: u64,
}

impl Query for LensEvidence {
    type Params = LensEvidenceParams;
    type Row = LensEvidenceRow;

    const SQL: &'static str = include_str!("../../query/lens_evidence.sql");
}
