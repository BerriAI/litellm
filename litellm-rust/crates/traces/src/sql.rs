use std::collections::BTreeMap;

use litellm_http::Client;

use crate::{Connection, Error, Parameter, execute_read};

pub enum ReadQuery {
    ListTraces,
    TraceSpans,
    SpanDetail,
    SpanError,
    SpendByResponseIds,
}

impl ReadQuery {
    pub fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "list_traces" => Ok(Self::ListTraces),
            "trace_spans" => Ok(Self::TraceSpans),
            "span_detail" => Ok(Self::SpanDetail),
            "span_error" => Ok(Self::SpanError),
            "spend_by_response_ids" => Ok(Self::SpendByResponseIds),
            _ => Err(Error::InvalidQuery),
        }
    }

    fn sql(&self) -> &'static str {
        match self {
            Self::ListTraces => include_str!("../query/list_traces.sql"),
            Self::TraceSpans => include_str!("../query/trace_spans.sql"),
            Self::SpanDetail => include_str!("../query/span_detail.sql"),
            Self::SpanError => include_str!("../query/span_error.sql"),
            Self::SpendByResponseIds => include_str!("../query/spend_by_response_ids.sql"),
        }
    }
}

#[derive(Clone, Copy)]
pub enum LensQuery {
    Availability,
    Agents,
    Sample,
    Content,
    Evidence,
}

impl LensQuery {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "availability" => Ok(Self::Availability),
            "agents" => Ok(Self::Agents),
            "sample" => Ok(Self::Sample),
            "content" => Ok(Self::Content),
            "evidence" => Ok(Self::Evidence),
            _ => Err(Error::InvalidQuery),
        }
    }
    pub fn sql(self) -> &'static str {
        match self {
            Self::Availability => include_str!("../query/lens_availability.sql"),
            Self::Agents => include_str!("../query/lens_agents.sql"),
            Self::Sample => include_str!("../query/lens_sample.sql"),
            Self::Content => include_str!("../query/lens_content.sql"),
            Self::Evidence => include_str!("../query/lens_evidence.sql"),
        }
    }
}

pub async fn execute_named_read(
    client: &Client,
    connection: &Connection,
    query: ReadQuery,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error> {
    execute_read(client, connection, query.sql(), parameters).await
}
