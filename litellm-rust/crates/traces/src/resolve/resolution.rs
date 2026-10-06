use std::collections::{BTreeSet, HashMap};

use indexmap::IndexMap;

use crate::{
    normalize::ObservationType,
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};

use super::{
    graph::Graph,
    spend::{self, Requests},
};

pub(super) fn agent_label(row: &TraceSpansRow) -> &str {
    if row.agent.is_empty() {
        &row.name
    } else {
        &row.agent
    }
}

pub(super) struct Resolution<'a> {
    pub(super) graph: Graph<'a>,
    spend: &'a [SpendRow],
    types: HashMap<&'a str, ObservationType>,
    tool_failures: HashMap<&'a str, &'a TraceSpansRow>,
    pub(super) model_calls: Vec<usize>,
}

impl<'a> Resolution<'a> {
    pub(super) fn new(rows: &'a [TraceSpansRow], spend: &'a [SpendRow]) -> Self {
        let graph = Graph::new(rows);
        let named_agents = rows.iter().any(|row| !row.agent.is_empty());
        let types: HashMap<&str, ObservationType> = (0..rows.len())
            .map(|index| (graph.id(index), resolved_type(&graph, index, named_agents)))
            .collect();
        let model_calls = (0..rows.len())
            .filter(|index| {
                types[graph.id(*index)] == ObservationType::Llm
                    && !graph
                        .descendants(*index)
                        .into_iter()
                        .any(|descendant| types[graph.id(descendant)] == ObservationType::Llm)
            })
            .collect();
        Self {
            graph,
            spend,
            types,
            tool_failures: rows
                .iter()
                .filter(|row| {
                    row.framework == "claude-code"
                        && row.name == "claude_code.tool_result"
                        && !row.tool_call_id.is_empty()
                        && row.status == crate::SpanStatus::Error
                })
                .map(|row| (row.tool_call_id.as_str(), row))
                .collect(),
            model_calls,
        }
    }

    pub(super) fn row(&self, index: usize) -> &'a TraceSpansRow {
        &self.graph.rows[index]
    }

    pub(super) fn status_source(&self, index: usize) -> &'a TraceSpansRow {
        let row = self.row(index);
        if row.framework != "claude-code"
            || row.kind != ObservationType::Tool
            || row.status == crate::SpanStatus::Error
        {
            return row;
        }
        self.graph
            .children(index)
            .into_iter()
            .map(|child| self.row(child))
            .find(|child| {
                child.name == "claude_code.tool.execution"
                    && !row.tool_call_id.is_empty()
                    && child.tool_call_id == row.tool_call_id
                    && child.status == crate::SpanStatus::Error
            })
            .or_else(|| self.tool_failures.get(row.tool_call_id.as_str()).copied())
            .unwrap_or(row)
    }

    pub(super) fn kind(&self, index: usize) -> ObservationType {
        self.types[self.graph.id(index)]
    }

    pub(super) fn is_agent(&self, index: usize) -> bool {
        self.kind(index) == ObservationType::Agent
    }

    pub(super) fn owner(&self, index: usize) -> &'a str {
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

    pub(super) fn requests(&self, index: usize) -> Option<Requests<'a>> {
        spend::requests(&spend::response_ids(self.row(index)), self.spend)
    }

    /// A model call is priced by every response id recorded on it, on the LLM wrappers around
    /// only it, and on the spans beneath it.
    pub(super) fn call_requests(&self, call: usize) -> Option<Requests<'a>> {
        let wrappers = self.graph.ancestors(call).into_iter().filter(|ancestor| {
            self.kind(*ancestor) == ObservationType::Llm
                && self
                    .graph
                    .descendants(*ancestor)
                    .into_iter()
                    .all(|descendant| {
                        self.graph.id(descendant) == self.graph.id(call)
                            || self.kind(descendant) != ObservationType::Llm
                    })
        });
        let ids: BTreeSet<String> = std::iter::once(call)
            .chain(wrappers)
            .chain(self.graph.descendants(call))
            .flat_map(|source| spend::response_ids(self.row(source)))
            .collect();
        spend::requests(&ids, self.spend)
    }

    pub(super) fn unique_tools(&self) -> Vec<usize> {
        let mut by_call: IndexMap<&str, usize> = IndexMap::new();
        for index in
            (0..self.graph.rows.len()).filter(|index| self.kind(*index) == ObservationType::Tool)
        {
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

fn resolved_type(graph: &Graph<'_>, index: usize, named_agents: bool) -> ObservationType {
    let row = &graph.rows[index];
    if !row.wrapper_candidate || row.kind != ObservationType::Agent {
        return row.kind;
    }
    if row.agent.is_empty() {
        return if named_agents {
            ObservationType::Chain
        } else {
            ObservationType::Agent
        };
    }
    let nearest = graph
        .ancestors(index)
        .into_iter()
        .find(|ancestor| graph.rows[*ancestor].kind == ObservationType::Agent);
    match nearest {
        Some(agent) if agent_label(&graph.rows[agent]) == row.agent => ObservationType::Chain,
        _ => ObservationType::Agent,
    }
}
