//! Trace resolution: the evidence normalization recorded on each span, resolved against the whole
//! trace.
//!
//! Normalization reads one span at a time, so it can only say what a span claims to be. Whether an
//! unnamed root agent only wraps the agents below it, which agent owns a model call, and which spend
//! records a call accounts for depend on the span graph, and parents and children can arrive in
//! separate exports. Trace detail and the trace list both resolve through here.

use std::collections::{BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use time::OffsetDateTime;

use crate::{
    normalize::CallKey,
    query::named::{ListTracesRow, SpendByResponseIdsRow as SpendRow, TraceSpansRow},
    view::{AgentNode, Span, SpanStatus, Trace, TraceSummary},
};

const NANOS_PER_MS: f64 = 1_000_000.0;

fn parse_call_key(encoded: &str) -> Option<CallKey> {
    let (kind, id) = encoded.split_once(':')?;
    match kind {
        "provider_response" if !id.is_empty() => Some(CallKey::ProviderResponse(id.to_owned())),
        "litellm_request" if !id.is_empty() => Some(CallKey::LiteLlmRequest(id.to_owned())),
        "transport" if id.is_empty() => Some(CallKey::Transport),
        _ => None,
    }
}

/// The span's recorded keys; rows stored before call evidence keep one response id.
fn call_keys(row: &TraceSpansRow) -> Vec<Option<CallKey>> {
    if row.call_keys.is_empty() && !row.litellm_request_id.is_empty() {
        return vec![Some(CallKey::ProviderResponse(
            row.litellm_request_id.clone(),
        ))];
    }
    row.call_keys
        .iter()
        .map(|key| parse_call_key(key))
        .collect()
}

fn call_evidence(row: &TraceSpansRow) -> &str {
    match (
        row.call_evidence.as_str(),
        row.litellm_request_id.is_empty(),
    ) {
        ("", false) => "complete",
        ("", true) => "unknown",
        (recorded, _) => recorded,
    }
}

/// The spend records to fetch for a set of spans.
#[derive(Debug, Default, PartialEq)]
pub struct SpendLookup {
    pub response_ids: Vec<String>,
    pub request_ids: Vec<String>,
    /// Traces whose transport spans LiteLLM logged by `traceparent`.
    pub trace_ids: Vec<String>,
}

impl SpendLookup {
    pub fn new(rows: &[TraceSpansRow]) -> Self {
        let mut response_ids = BTreeSet::new();
        let mut request_ids = BTreeSet::new();
        let mut trace_ids = BTreeSet::new();
        for row in rows {
            for key in call_keys(row) {
                match key {
                    Some(CallKey::ProviderResponse(id)) => {
                        response_ids.insert(id.to_owned());
                    }
                    Some(CallKey::LiteLlmRequest(id)) => {
                        request_ids.insert(id.to_owned());
                    }
                    Some(CallKey::Transport) if !row.trace_id.is_empty() => {
                        trace_ids.insert(row.trace_id.clone());
                    }
                    _ => {}
                }
            }
        }
        Self {
            response_ids: response_ids.into_iter().collect(),
            request_ids: request_ids.into_iter().collect(),
            trace_ids: trace_ids.into_iter().collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_ids.is_empty() && self.request_ids.is_empty() && self.trace_ids.is_empty()
    }
}

/// Who a trace's spend records must belong to.
struct Ownership<'a> {
    team_id: &'a str,
    api_key_hash: &'a str,
    user_id: &'a str,
}

impl Ownership<'_> {
    fn owns(&self, spend: &SpendRow) -> bool {
        spend.team_id == self.team_id
            && ((!self.user_id.is_empty() && spend.user == self.user_id)
                || (!self.api_key_hash.is_empty() && spend.api_key == self.api_key_hash))
    }
}

struct Graph<'a> {
    rows: &'a [TraceSpansRow],
    by_id: HashMap<&'a str, usize>,
    children: HashMap<&'a str, Vec<usize>>,
}

impl<'a> Graph<'a> {
    fn new(rows: &'a [TraceSpansRow]) -> Self {
        let by_id: HashMap<&str, usize> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| (row.span_id.as_str(), index))
            .collect();
        let mut children: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, row) in rows.iter().enumerate() {
            if row.parent_span_id != row.span_id && by_id.contains_key(row.parent_span_id.as_str())
            {
                children.entry(&row.parent_span_id).or_default().push(index);
            }
        }
        Self {
            rows,
            by_id,
            children,
        }
    }

    fn id(&self, index: usize) -> &'a str {
        &self.rows[index].span_id
    }

    fn parent(&self, index: usize) -> Option<usize> {
        let row = &self.rows[index];
        if row.parent_span_id == row.span_id {
            return None;
        }
        self.by_id.get(row.parent_span_id.as_str()).copied()
    }

    fn is_root(&self, index: usize) -> bool {
        let parent = &self.rows[index].parent_span_id;
        parent.is_empty() || !self.by_id.contains_key(parent.as_str())
    }

    fn ancestors(&self, index: usize) -> Vec<usize> {
        let mut seen = HashSet::from([self.id(index)]);
        let mut found = Vec::new();
        let mut current = self.parent(index);
        while let Some(ancestor) = current.filter(|ancestor| seen.insert(self.id(*ancestor))) {
            found.push(ancestor);
            current = self.parent(ancestor);
        }
        found
    }

    fn descendants(&self, index: usize) -> Vec<usize> {
        let children = |index: usize| {
            self.children
                .get(self.id(index))
                .into_iter()
                .flatten()
                .copied()
        };
        let mut seen = HashSet::from([self.id(index)]);
        let mut found = Vec::new();
        let mut stack: Vec<usize> = children(index).collect();
        while let Some(descendant) = stack.pop() {
            if seen.insert(self.id(descendant)) {
                found.push(descendant);
                stack.extend(children(descendant));
            }
        }
        found
    }
}

fn agent_label(row: &TraceSpansRow) -> &str {
    if row.agent.is_empty() {
        &row.name
    } else {
        &row.agent
    }
}

type Requests<'a> = Vec<&'a SpendRow>;

enum KeyMatch<'a> {
    Missing,
    Unique(&'a SpendRow),
    Ambiguous(Requests<'a>),
}

impl<'a> KeyMatch<'a> {
    fn new(requests: Requests<'a>) -> Self {
        match requests.as_slice() {
            [] => Self::Missing,
            [request] => Self::Unique(request),
            _ => Self::Ambiguous(requests),
        }
    }

    fn unique(&self) -> Option<&'a SpendRow> {
        match self {
            Self::Unique(request) => Some(request),
            Self::Missing | Self::Ambiguous(_) => None,
        }
    }

    fn agrees_with(&self, selected: &[&SpendRow]) -> bool {
        match self {
            Self::Missing => false,
            Self::Unique(request) => selected
                .iter()
                .any(|row| row.request_id == request.request_id),
            Self::Ambiguous(requests) => {
                requests
                    .iter()
                    .filter(|request| {
                        selected
                            .iter()
                            .any(|row| row.request_id == request.request_id)
                    })
                    .count()
                    == 1
            }
        }
    }
}

enum SpendEvidence<'a> {
    Unknown,
    Partial(Vec<KeyMatch<'a>>),
    Complete(Vec<KeyMatch<'a>>),
}

impl<'a> SpendEvidence<'a> {
    fn complete_requests(&self) -> Option<Requests<'a>> {
        match self {
            Self::Complete(matches) if !matches.is_empty() => {
                let requests: Requests<'a> = matches
                    .iter()
                    .filter_map(KeyMatch::unique)
                    .map(|request| (request.request_id.as_str(), request))
                    .collect::<IndexMap<_, _>>()
                    .into_values()
                    .collect();
                matches
                    .iter()
                    .all(|matched| matched.agrees_with(&requests))
                    .then_some(requests)
            }
            Self::Unknown | Self::Partial(_) | Self::Complete(_) => None,
        }
    }

    fn agrees_with(&self, selected: &[&SpendRow]) -> bool {
        match self {
            Self::Unknown => true,
            Self::Partial(matches) | Self::Complete(matches) => matches
                .iter()
                .all(|evidence| evidence.agrees_with(selected)),
        }
    }
}

/// Roles, ownership and calls of one trace's spans.
struct Resolution<'a> {
    graph: Graph<'a>,
    ownership: Ownership<'a>,
    spend: &'a [SpendRow],
    types: HashMap<&'a str, &'a str>,
    model_calls: Vec<usize>,
}

impl<'a> Resolution<'a> {
    fn new(rows: &'a [TraceSpansRow], spend: &'a [SpendRow]) -> Self {
        let graph = Graph::new(rows);
        let named_agents = rows.iter().any(|row| !row.agent.is_empty());
        let types: HashMap<&str, &str> = (0..rows.len())
            .map(|index| (graph.id(index), resolved_type(&graph, index, named_agents)))
            .collect();
        let model_calls = (0..rows.len())
            .filter(|index| {
                types[graph.id(*index)] == "llm"
                    && !graph
                        .descendants(*index)
                        .into_iter()
                        .any(|descendant| types[graph.id(descendant)] == "llm")
            })
            .collect();
        Self {
            ownership: Ownership {
                team_id: &rows[0].team_id,
                api_key_hash: &rows[0].api_key_hash,
                user_id: &rows[0].user_id,
            },
            graph,
            spend,
            types,
            model_calls,
        }
    }

    fn row(&self, index: usize) -> &'a TraceSpansRow {
        &self.graph.rows[index]
    }

    fn kind(&self, index: usize) -> &'a str {
        self.types[self.graph.id(index)]
    }

    fn is_agent(&self, index: usize) -> bool {
        self.kind(index) == "agent"
    }

    /// The agent a call or tool runs for: its own, else the agent it runs inside.
    fn owner(&self, index: usize) -> &'a str {
        let row = self.row(index);
        if !row.agent.is_empty() {
            return &row.agent;
        }
        self.graph
            .ancestors(index)
            .into_iter()
            .find(|ancestor| self.is_agent(*ancestor))
            .map_or("", |agent| agent_label(self.row(agent)))
    }

    fn matches(
        &self,
        key: &Option<CallKey>,
        row: &TraceSpansRow,
    ) -> IndexMap<&'a str, &'a SpendRow> {
        let matches = |spend: &SpendRow| match key {
            Some(CallKey::ProviderResponse(id)) => {
                !id.is_empty() && (spend.response_id == *id || spend.upstream_response_id == *id)
            }
            Some(CallKey::LiteLlmRequest(id)) => !id.is_empty() && spend.request_id == *id,
            Some(CallKey::Transport) => {
                !row.trace_id.is_empty()
                    && !row.span_id.is_empty()
                    && spend.trace_id == row.trace_id
                    && spend.span_id == row.span_id
            }
            None => false,
        };
        self.spend
            .iter()
            .filter(|spend| self.ownership.owns(spend) && matches(spend))
            .map(|spend| (spend.request_id.as_str(), spend))
            .collect()
    }

    fn requests(&self, index: usize) -> SpendEvidence<'a> {
        let row = self.row(index);
        let matches = call_keys(row)
            .iter()
            .map(|key| KeyMatch::new(self.matches(key, row).into_values().collect()))
            .collect();
        match call_evidence(row) {
            "complete" => SpendEvidence::Complete(matches),
            "partial" => SpendEvidence::Partial(matches),
            _ => SpendEvidence::Unknown,
        }
    }

    fn call_requests(&self, call: usize) -> Option<Requests<'a>> {
        let wrappers = self.graph.ancestors(call).into_iter().filter(|ancestor| {
            self.kind(*ancestor) == "llm"
                && self
                    .graph
                    .descendants(*ancestor)
                    .into_iter()
                    .all(|descendant| {
                        self.graph.id(descendant) == self.graph.id(call)
                            || self.kind(descendant) != "llm"
                    })
        });
        let sources: Vec<_> = std::iter::once(call)
            .chain(wrappers)
            .map(|source| self.requests(source))
            .collect();
        let transports: Vec<_> = self
            .graph
            .descendants(call)
            .into_iter()
            .filter(|descendant| {
                call_keys(self.row(*descendant))
                    .iter()
                    .any(|key| matches!(key, Some(CallKey::Transport)))
            })
            .map(|transport| self.requests(transport))
            .collect();
        let transport_requests: Option<Vec<Requests<'a>>> = (!transports.is_empty())
            .then(|| {
                transports
                    .iter()
                    .map(SpendEvidence::complete_requests)
                    .collect()
            })
            .flatten();
        let selected: Requests<'a> = transport_requests
            .map(|requests| requests.into_iter().flatten().collect())
            .into_iter()
            .chain(sources.iter().filter_map(SpendEvidence::complete_requests))
            .find(|selected| {
                sources
                    .iter()
                    .chain(&transports)
                    .all(|source| source.agrees_with(selected))
            })?;
        Some(
            selected
                .into_iter()
                .map(|request| (request.request_id.as_str(), request))
                .collect::<IndexMap<_, _>>()
                .into_values()
                .collect(),
        )
    }

    /// Tool spans, once per call: overlapping instrumentations record one call under one id.
    fn unique_tools(&self) -> Vec<usize> {
        let mut by_call: IndexMap<&str, usize> = IndexMap::new();
        for index in (0..self.graph.rows.len()).filter(|index| self.kind(*index) == "tool") {
            let row = self.row(index);
            let key = if row.tool_call_id.is_empty() {
                &row.span_id
            } else {
                &row.tool_call_id
            };
            by_call.entry(key).or_insert(index);
        }
        by_call.into_values().collect()
    }
}

fn resolved_type<'a>(graph: &Graph<'a>, index: usize, named_agents: bool) -> &'a str {
    let row = &graph.rows[index];
    if row.wrapper_candidate == 0 || row.kind != "agent" {
        return &row.kind;
    }
    // An unnamed agent wraps the run when the trace names its agents; a named one repeats the agent
    // it runs inside.
    if row.agent.is_empty() {
        return if named_agents { "chain" } else { "agent" };
    }
    let nearest = graph
        .ancestors(index)
        .into_iter()
        .find(|ancestor| graph.rows[*ancestor].kind == "agent");
    match nearest {
        Some(agent) if agent_label(&graph.rows[agent]) == row.agent => "chain",
        _ => "agent",
    }
}

fn request_cost(requests: &[&SpendRow]) -> Option<f64> {
    requests.iter().try_fold(0.0, |total, request| {
        let cost = request.spend.filter(|cost| cost.is_finite())?;
        let sum = total + cost;
        sum.is_finite().then_some(sum)
    })
}

fn total(calls: &[Option<Requests<'_>>]) -> Option<f64> {
    if calls.is_empty() {
        return None;
    }
    let requests: Option<Vec<&SpendRow>> = calls
        .iter()
        .map(|requests| requests.as_ref())
        .collect::<Option<Vec<_>>>()
        .map(|calls| calls.into_iter().flatten().copied().collect());
    let unique: IndexMap<&str, &SpendRow> = requests?
        .into_iter()
        .map(|request| (request.request_id.as_str(), request))
        .collect();
    request_cost(&unique.into_values().collect::<Vec<_>>())
}

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
        kind: resolution.kind(index).to_owned(),
        agent: row.agent.clone(),
        framework: row.framework.clone(),
        start_offset_ms: (i128::from(row.start_ns) - i128::from(trace_start_ns)) as f64
            / NANOS_PER_MS,
        duration_ms: row.duration_ns as f64 / NANOS_PER_MS,
        status: SpanStatus::from_code(&row.status),
        error: optional(&row.status_message),
        error_truncated: row.error_truncated != 0,
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

/// One node per distinct agent: agent spans, and agents named only by their calls and tools.
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

/// `datetime.isoformat()` of a UTC instant: microseconds only when the instant has any.
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
            !span.input_preview.is_empty() && matches!(span.kind.as_str(), "agent" | "llm")
        })
        .min_by_key(|(index, (_, row))| (row.start_ns, *index))
        .map(|(_, (span, _))| span.input_preview.clone())
        .unwrap_or_default();
    let summary = TraceSummary {
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
        // Agent spans repeat their calls' usage, so totals count model calls when there are any.
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
    })
}

/// A listed trace whose spans could not be read: rollup counts only, cost unknown.
pub fn listed_summary(row: &ListTracesRow) -> TraceSummary {
    TraceSummary {
        trace_id: row.trace_id.clone(),
        trace_ref: row.trace_ref.clone(),
        name: row.name.clone(),
        service: row.service.clone(),
        agent_names: row.agent_names.clone(),
        frameworks: row.frameworks.clone(),
        input_preview: row.input_preview.clone(),
        start_time: iso_time(row.start_ms),
        duration_ms: row.duration_ms as f64,
        status: SpanStatus::from_code(&row.status),
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
