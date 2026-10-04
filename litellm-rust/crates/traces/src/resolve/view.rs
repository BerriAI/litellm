use std::collections::{BTreeSet, HashSet};

use indexmap::IndexMap;
use time::OffsetDateTime;

use crate::{
    normalize::ObservationType,
    query::named::{ListTracesRow, SpendByResponseIdsRow as SpendRow, TraceSpansRow},
    view::{AgentNode, Span, SpanStatus, Trace, TraceSummary},
};

use super::{
    resolution::{Resolution, agent_label},
    spend::{Requests, request_cost, total},
};

const NANOS_PER_MS: f64 = 1_000_000.0;

fn optional(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

fn span(resolution: &Resolution<'_>, index: usize, trace_start_ns: i64) -> Span {
    let row = resolution.row(index);
    let requests = resolution.requests(index).complete_requests();
    Span {
        span_id: row.span_id.clone(),
        parent_span_id: optional(&row.parent_span_id),
        name: row.name.clone(),
        kind: resolution.kind(index),
        agent: row.agent.clone(),
        framework: row.framework.clone(),
        start_offset_ms: (i128::from(row.start_ns) - i128::from(trace_start_ns)) as f64
            / NANOS_PER_MS,
        duration_ms: row.duration_ns as f64 / NANOS_PER_MS,
        status: row.status,
        error: optional(&row.status_message),
        error_truncated: row.error_truncated,
        input_preview: row.input_preview.clone(),
        model: optional(&row.model),
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        litellm_request_id: optional(&row.litellm_request_id),
        spend: requests
            .as_ref()
            .and_then(|requests| request_cost(requests)),
    }
}

fn agents(resolution: &Resolution<'_>) -> Vec<AgentNode> {
    let graph = &resolution.graph;
    let mut entries: IndexMap<&str, Vec<usize>> = IndexMap::new();
    for index in (0..graph.rows.len()).filter(|index| resolution.is_agent(*index)) {
        entries
            .entry(agent_label(resolution.row(index)))
            .or_default()
            .push(index);
    }
    let explicit: HashSet<&str> = entries.keys().copied().collect();
    for (index, row) in graph.rows.iter().enumerate() {
        let parent_agent = graph
            .parent(index)
            .map(|parent| graph.rows[parent].agent.as_str());
        if !row.agent.is_empty()
            && !explicit.contains(row.agent.as_str())
            && parent_agent != Some(row.agent.as_str())
        {
            entries.entry(&row.agent).or_default().push(index);
        }
    }
    let calls: Vec<(&str, Option<Requests<'_>>)> = resolution
        .model_calls
        .iter()
        .map(|call| (resolution.owner(*call), resolution.call_requests(*call)))
        .collect();
    let tools = resolution.unique_tools();
    entries
        .into_iter()
        .map(|(name, spans)| {
            let parent_agent = graph.ancestors(spans[0]).into_iter().find_map(|ancestor| {
                let label = agent_label(resolution.row(ancestor));
                (resolution.is_agent(ancestor) && label != name).then(|| label.to_owned())
            });
            let owned_calls: Vec<Option<Requests<'_>>> = calls
                .iter()
                .filter(|(owner, _)| *owner == name)
                .map(|(_, requests)| requests.clone())
                .collect();
            AgentNode {
                name: name.to_owned(),
                parent_agent,
                invocations: spans.len() as u64,
                llm_calls: owned_calls.len() as u64,
                tool_calls: tools
                    .iter()
                    .filter(|tool| resolution.owner(**tool) == name)
                    .count() as u64,
                duration_ms: spans
                    .iter()
                    .map(|span| graph.rows[*span].duration_ns)
                    .sum::<u64>() as f64
                    / NANOS_PER_MS,
                spend: total(&owned_calls),
            }
        })
        .collect()
}

pub fn iso_time(ms: i64) -> String {
    let instant = OffsetDateTime::from_unix_timestamp_nanos(i128::from(ms) * 1_000_000)
        .unwrap_or(OffsetDateTime::UNIX_EPOCH);
    let fraction = match instant.millisecond() {
        0 => String::new(),
        millis => format!(".{millis:03}000"),
    };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{fraction}+00:00",
        instant.year(),
        u8::from(instant.month()),
        instant.day(),
        instant.hour(),
        instant.minute(),
        instant.second(),
    )
}

fn sorted_unique<'a>(values: impl Iterator<Item = &'a str>) -> Vec<String> {
    values
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

pub fn resolve_trace(
    trace_id: &str,
    trace_ref: &str,
    rows: &[TraceSpansRow],
    spend: &[SpendRow],
) -> Option<Trace> {
    let first = rows.first()?;
    let resolution = Resolution::new(rows, spend);
    let trace_start_ns = rows.iter().map(|row| row.start_ns).min()?;
    let trace_end_ns = rows
        .iter()
        .map(|row| i128::from(row.start_ns) + i128::from(row.duration_ns))
        .max()?;
    let spans: Vec<Span> = (0..rows.len())
        .map(|index| span(&resolution, index, trace_start_ns))
        .collect();
    let root = (0..rows.len())
        .find(|index| resolution.graph.is_root(*index))
        .unwrap_or_default();
    let agents = agents(&resolution);
    let calls = &resolution.model_calls;
    let counted: Vec<&TraceSpansRow> = if calls.is_empty() {
        rows.iter().collect()
    } else {
        calls.iter().map(|call| &rows[*call]).collect()
    };
    let first_input = spans
        .iter()
        .zip(rows)
        .enumerate()
        .filter(|(_, (span, _))| {
            !span.input_preview.is_empty()
                && matches!(span.kind, ObservationType::Agent | ObservationType::Llm)
        })
        .min_by_key(|(index, (_, row))| (row.start_ns, *index))
        .map(|(_, (span, _))| span.input_preview.clone())
        .unwrap_or_default();
    let summary = TraceSummary {
        resolution_limited: false,
        trace_id: trace_id.to_owned(),
        trace_ref: trace_ref.to_owned(),
        name: spans[root].name.clone(),
        service: first.service.clone(),
        agent_names: agents
            .iter()
            .map(|agent| agent.name.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        frameworks: sorted_unique(spans.iter().map(|span| span.framework.as_str())),
        input_preview: optional(&spans[root].input_preview).unwrap_or(first_input),
        start_time: iso_time(trace_start_ns.div_euclid(1_000_000)),
        duration_ms: (trace_end_ns - i128::from(trace_start_ns)) as f64 / NANOS_PER_MS,
        status: spans[root].status,
        span_count: spans.len() as u64,
        agent_count: agents.len() as u64,
        agent_invocations: agents.iter().map(|agent| agent.invocations).sum(),
        llm_calls: calls.len() as u64,
        tool_calls: resolution.unique_tools().len() as u64,
        error_count: spans
            .iter()
            .filter(|span| span.status == SpanStatus::Error)
            .count() as u64,
        input_tokens: counted.iter().map(|row| u64::from(row.input_tokens)).sum(),
        output_tokens: counted.iter().map(|row| u64::from(row.output_tokens)).sum(),
        models: sorted_unique(calls.iter().map(|call| rows[*call].model.as_str())),
        spend: total(
            &calls
                .iter()
                .map(|call| resolution.call_requests(*call))
                .collect::<Vec<_>>(),
        ),
    };
    Some(Trace {
        summary,
        agents,
        spans,
        next_cursor: None,
    })
}

pub fn listed_summary(row: &ListTracesRow) -> TraceSummary {
    TraceSummary {
        resolution_limited: true,
        trace_id: row.trace_id.clone(),
        trace_ref: row.trace_ref.clone(),
        name: row.name.clone(),
        service: row.service.clone(),
        agent_names: row.agent_names.clone(),
        frameworks: row.frameworks.clone(),
        input_preview: row.input_preview.clone(),
        start_time: iso_time(row.start_ms),
        duration_ms: row.duration_ms as f64,
        status: row.status,
        span_count: row.span_count,
        agent_count: row.agent_count,
        agent_invocations: if row.agent_invocations == 0 {
            row.agent_count
        } else {
            row.agent_invocations
        },
        llm_calls: row.llm_calls,
        tool_calls: row.tool_calls,
        error_count: row.error_count,
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        models: row.models.clone(),
        spend: None,
    }
}
