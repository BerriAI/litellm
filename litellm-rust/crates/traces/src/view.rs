//! Trace read responses, as the LiteLLM UI consumes them.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::ui::UiContent;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SpanStatus {
    Ok,
    Error,
    Unset,
}

impl SpanStatus {
    pub fn from_code(code: &str) -> Self {
        match code {
            "STATUS_CODE_OK" => Self::Ok,
            "STATUS_CODE_ERROR" => Self::Error,
            _ => Self::Unset,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Span {
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub agent: String,
    pub framework: String,
    pub start_offset_ms: f64,
    pub duration_ms: f64,
    pub status: SpanStatus,
    pub error: Option<String>,
    pub error_truncated: bool,
    pub input_preview: String,
    pub model: Option<String>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub litellm_request_id: Option<String>,
    pub spend: Option<f64>,
}

/// One distinct agent in a trace: 200 invocations of `researcher` are one node.
#[derive(Debug, PartialEq, Serialize)]
pub struct AgentNode {
    pub name: String,
    pub parent_agent: Option<String>,
    pub invocations: u64,
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub duration_ms: f64,
    pub spend: Option<f64>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct TraceSummary {
    pub trace_id: String,
    pub trace_ref: String,
    pub name: String,
    pub service: String,
    pub agent_names: Vec<String>,
    pub frameworks: Vec<String>,
    pub input_preview: String,
    pub start_time: String,
    pub duration_ms: f64,
    pub status: SpanStatus,
    pub span_count: u64,
    pub agent_count: u64,
    pub agent_invocations: u64,
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub error_count: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub models: Vec<String>,
    pub spend: Option<f64>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Trace {
    pub summary: TraceSummary,
    pub agents: Vec<AgentNode>,
    pub spans: Vec<Span>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct TracePage {
    pub data: Vec<TraceSummary>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct SpanDetail {
    pub span_id: String,
    pub input_ui: UiContent,
    pub output_ui: UiContent,
    pub input: String,
    pub output: String,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct SpanErrorPage {
    pub span_id: String,
    pub message: String,
    pub total_chars: u64,
    pub next_cursor: Option<String>,
}
