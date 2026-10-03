use std::collections::HashMap;

use indexmap::IndexMap;

use crate::{
    normalize::{CallKey, ObservationType},
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};

use super::{
    graph::Graph,
    spend::{self, Ownership, Requests, SpendEvidence},
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
    ownership: Ownership<'a>,
    spend: &'a [SpendRow],
    types: HashMap<&'a str, ObservationType>,
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

    pub(super) fn row(&self, index: usize) -> &'a TraceSpansRow {
        &self.graph.rows[index]
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

    pub(super) fn requests(&self, index: usize) -> SpendEvidence<'a> {
        spend::requests(self.row(index), &self.ownership, self.spend)
    }

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
        let sources: Vec<_> = std::iter::once(call)
            .chain(wrappers)
            .map(|source| self.requests(source))
            .collect();
        let transports: Vec<_> = self
            .graph
            .descendants(call)
            .into_iter()
            .filter(|descendant| {
                self.row(*descendant)
                    .call_keys
                    .contains(&CallKey::Transport)
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
