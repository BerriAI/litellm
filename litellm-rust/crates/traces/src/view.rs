//! Trace read responses, as the LiteLLM UI consumes them.

use std::collections::BTreeMap;

use crate::ui::UiContent;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SpanStatus {
    #[serde(alias = "STATUS_CODE_OK")]
    Ok,
    #[serde(alias = "STATUS_CODE_ERROR")]
    Error,
    #[serde(other)]
    Unset,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: crate::ObservationType,
    pub agent: String,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub actor_id: Option<String>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub actor_unassigned: bool,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub parent_actor_id: Option<String>,
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
    pub spend_log_request_id: Option<String>,
    pub spend_match: Option<SpendMatch>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SpendMatch {
    Matched,
    NoCallId,
    NoSpendLog,
    Ambiguous,
    IncompleteEvidence,
}

/// One distinct agent in a trace: 200 invocations of `researcher` are one node.
#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct AgentNode {
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub actor_id: Option<String>,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub parent_actor_id: Option<String>,
    pub name: String,
    pub parent_agent: Option<String>,
    pub invocations: u64,
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub duration_ms: f64,
    pub spend: Option<f64>,
    pub priced_calls: u64,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSummary {
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub resolution_limited: bool,
    pub trace_id: String,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub trace_ref: String,
    pub name: String,
    pub service: String,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub agent_names: Vec<String>,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
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
    pub priced_calls: u64,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub capture: Option<TraceCapture>,
    pub summary: TraceSummary,
    pub agents: Vec<AgentNode>,
    pub spans: Vec<Span>,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-optional" = true)))]
    pub next_cursor: Option<String>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Debug, PartialEq)]
pub struct TracePage {
    pub data: Vec<TraceSummary>,
    pub next_cursor: Option<String>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Debug, PartialEq)]
pub struct SpanDetail {
    pub span_id: String,
    pub input_ui: UiContent,
    pub output_ui: UiContent,
    pub input: String,
    pub output: String,
    pub attributes: BTreeMap<String, String>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Debug, PartialEq)]
pub struct SpanErrorPage {
    pub span_id: String,
    pub message: String,
    pub total_chars: u64,
    pub next_cursor: Option<String>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActorCapture {
    pub actor_id: String,
    pub name: String,
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub reply_events: u64,
    pub model_outputs: u64,
    pub content_events: u64,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceCapture {
    pub actors: Vec<ActorCapture>,
    pub content_events: u64,
    pub unassigned_events: u64,
    pub warning_events: u64,
}
