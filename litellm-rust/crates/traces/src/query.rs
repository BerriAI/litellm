pub mod guide;
pub mod named;

#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::EnumString, strum::Display, strum::AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum ReadQuery {
    ListTraces,
    TraceSpans,
    TracePageSpans,
    TraceIdentity,
    SpanDetail,
    SpanError,
    SpendByResponseIds,
    Availability,
    Agents,
    Sample,
    Content,
    Evidence,
}

impl ReadQuery {
    pub fn parse(value: &str) -> Result<Self, crate::InvalidQuery> {
        value.parse().map_err(|_| crate::InvalidQuery)
    }
}
