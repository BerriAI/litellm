pub mod guide;
pub mod named;

#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::EnumString, strum::Display, strum::AsRefStr)]
pub enum ReadQuery {
    #[strum(serialize = "list_traces")]
    ListTraces,
    #[strum(serialize = "trace_agents")]
    TraceAgents,
    #[strum(serialize = "trace_spans")]
    TraceSpans,
    #[strum(serialize = "trace_page_spans")]
    TracePageSpans,
    #[strum(serialize = "trace_identity")]
    TraceIdentity,
    #[strum(serialize = "span_detail")]
    SpanDetail,
    #[strum(serialize = "span_error")]
    SpanError,
    #[strum(serialize = "spend_by_response_ids")]
    SpendByResponseIds,
    #[strum(serialize = "availability")]
    Availability,
    #[strum(serialize = "agents")]
    Agents,
    #[strum(serialize = "sample")]
    Sample,
    #[strum(serialize = "content")]
    Content,
    #[strum(serialize = "evidence")]
    Evidence,
    #[strum(serialize = "feedback_target")]
    FeedbackTarget,
    #[strum(serialize = "feedback")]
    Feedback,
    #[strum(serialize = "feedback_summary")]
    FeedbackSummary,
}

impl ReadQuery {
    pub fn parse(value: &str) -> Result<Self, crate::InvalidQuery> {
        value.parse().map_err(|_| crate::InvalidQuery)
    }
}
