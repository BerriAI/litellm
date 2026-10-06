use std::collections::{BTreeMap, BTreeSet, HashMap};

use sha2::{Digest, Sha256};

use crate::query::named::TraceSpansRow;

use super::graph::Graph;

pub(super) fn native(row: &TraceSpansRow) -> bool {
    (row.framework == "claude-code"
        || (row.framework == "claude-agent-sdk"
            && (row.name.starts_with("claude_code.")
                || !row.native_agent_id.is_empty()
                || !row.query_source.is_empty())))
        && (!row.session_id.is_empty() || !row.native_agent_id.is_empty())
}

fn identity(row: &TraceSpansRow, agent: &str) -> String {
    let session = if row.session_id.is_empty() {
        &row.trace_id
    } else {
        &row.session_id
    };
    let mut digest = Sha256::new();
    digest.update((session.len() as u64).to_be_bytes());
    digest.update(session);
    digest.update(agent);
    format!("claude-code:{:x}", digest.finalize())
}

fn label(value: &str) -> String {
    value.chars().take(256).collect()
}

fn root(row: &TraceSpansRow) -> String {
    identity(row, "root")
}

fn explicit(row: &TraceSpansRow) -> Option<String> {
    if !native(row) {
        return None;
    }
    if !row.native_agent_id.is_empty() {
        return Some(identity(row, &format!("agent:{}", row.native_agent_id)));
    }
    (row.name == "claude_code.interaction"
        || matches!(
            row.query_source.as_str(),
            "repl_main_thread"
                | "sdk"
                | "sdk_main_thread"
                | "generate_session_title"
                | "prompt_suggestion"
        ))
    .then(|| root(row))
}

fn child_source(row: &TraceSpansRow) -> bool {
    row.query_source == "agent"
        || row.query_source.starts_with("agent:")
        || row.query_source.starts_with("agent.")
}

pub(super) struct Actor {
    pub(super) id: String,
    pub(super) parent: Option<String>,
    pub(super) name: String,
    pub(super) spans: Vec<usize>,
}

pub(super) struct Actors {
    owners: Vec<Option<String>>,
    pub(super) entries: BTreeMap<String, Actor>,
}

impl Actors {
    pub(super) fn new(graph: &Graph<'_>) -> Self {
        let rows = graph.rows;
        let direct: Vec<_> = rows.iter().map(explicit).collect();
        let mut boundaries: HashMap<usize, BTreeSet<&str>> = HashMap::new();
        let mut tools: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for (index, owner) in direct.iter().enumerate() {
            let Some(owner) = owner.as_deref() else {
                continue;
            };
            let row = &rows[index];
            if let Some(parent) = graph.parent(index)
                && rows[parent].name == "claude_code.tool.execution"
            {
                boundaries.entry(parent).or_default().insert(owner);
            }
            if !row.tool_call_id.is_empty() {
                tools.entry(&row.tool_call_id).or_default().insert(owner);
            }
        }
        let unique = |set: Option<&BTreeSet<&str>>| {
            set.filter(|set| set.len() == 1)
                .and_then(|set| set.first().copied())
                .map(str::to_owned)
        };
        let candidates: Vec<_> = (0..rows.len())
            .map(|index| unique(boundaries.get(&index)).or_else(|| direct[index].clone()))
            .collect();
        let inherited =
            graph.nearest_ancestors(&candidates.iter().map(Option::is_some).collect::<Vec<_>>());
        let inherited_child = graph.nearest_ancestors(
            &candidates
                .iter()
                .enumerate()
                .map(|(index, candidate)| {
                    candidate
                        .as_ref()
                        .is_some_and(|candidate| candidate != &root(&rows[index]))
                })
                .collect::<Vec<_>>(),
        );
        let owners: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                if !native(row) {
                    return None;
                }
                direct[index]
                    .clone()
                    .or_else(|| unique(boundaries.get(&index)))
                    .or_else(|| unique(tools.get(row.tool_call_id.as_str())))
                    .or_else(|| {
                        let ancestor = if child_source(row) {
                            inherited_child[index]
                        } else {
                            inherited[index]
                        }?;
                        candidates[ancestor].clone()
                    })
            })
            .collect();
        let mut entries: BTreeMap<String, Actor> = BTreeMap::new();
        for (index, owner) in owners.iter().enumerate() {
            let Some(id) = owner else { continue };
            let row = &rows[index];
            let actor = entries.entry(id.clone()).or_insert_with(|| Actor {
                id: id.clone(),
                parent: (id != &root(row)).then(|| root(row)),
                name: if id == &root(row) {
                    label(&row.agent)
                } else {
                    "Subagent".to_owned()
                },
                spans: Vec::new(),
            });
            actor.spans.push(index);
            if !row.native_parent_agent_id.is_empty() {
                actor.parent = Some(identity(
                    row,
                    &format!("agent:{}", row.native_parent_agent_id),
                ));
            }
            if child_source(row) {
                let name = row
                    .query_source
                    .strip_prefix("agent:")
                    .and_then(|source| source.split_once(':'))
                    .or_else(|| {
                        row.query_source
                            .strip_prefix("agent.")
                            .and_then(|source| source.split_once('.'))
                    });
                if let Some((_, name)) = name
                    && !name.is_empty()
                {
                    actor.name = label(name);
                }
            }
        }
        Self { owners, entries }
    }

    pub(super) fn owner(&self, index: usize) -> Option<&Actor> {
        self.owners[index]
            .as_ref()
            .and_then(|id| self.entries.get(id))
    }
}
