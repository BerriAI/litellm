use std::collections::{HashMap, HashSet};

use crate::query::named::TraceSpansRow;

pub(super) struct Graph<'a> {
    pub(super) rows: &'a [TraceSpansRow],
    by_id: HashMap<&'a str, usize>,
    children: HashMap<&'a str, Vec<usize>>,
}

impl<'a> Graph<'a> {
    pub(super) fn new(rows: &'a [TraceSpansRow]) -> Self {
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

    pub(super) fn id(&self, index: usize) -> &'a str {
        &self.rows[index].span_id
    }

    pub(super) fn parent(&self, index: usize) -> Option<usize> {
        let row = &self.rows[index];
        if row.parent_span_id == row.span_id {
            return None;
        }
        self.by_id.get(row.parent_span_id.as_str()).copied()
    }

    pub(super) fn is_root(&self, index: usize) -> bool {
        let parent = &self.rows[index].parent_span_id;
        parent.is_empty() || !self.by_id.contains_key(parent.as_str())
    }

    pub(super) fn children(&self, index: usize) -> Vec<usize> {
        self.children
            .get(self.id(index))
            .map(|children| children.to_vec())
            .unwrap_or_default()
    }

    pub(super) fn ancestors(&self, index: usize) -> Vec<usize> {
        let mut seen = HashSet::from([self.id(index)]);
        let mut found = Vec::new();
        let mut current = self.parent(index);
        while let Some(ancestor) = current.filter(|ancestor| seen.insert(self.id(*ancestor))) {
            found.push(ancestor);
            current = self.parent(ancestor);
        }
        found
    }

    pub(super) fn nearest_session_ancestors(&self, matches: &[bool]) -> Vec<Option<usize>> {
        let parent = |index: usize| {
            self.parent(index)
                .filter(|parent| self.rows[*parent].session_id == self.rows[index].session_id)
        };
        let mut nearest: Vec<_> = matches
            .iter()
            .enumerate()
            .map(|(index, found)| found.then_some(index))
            .collect();
        let mut resolved = matches.to_vec();
        let mut visiting = vec![false; self.rows.len()];
        for index in 0..self.rows.len() {
            let mut path = Vec::new();
            let mut current = Some(index);
            let found = loop {
                let Some(node) = current else { break None };
                if resolved[node] {
                    break nearest[node];
                }
                if visiting[node] {
                    break None;
                }
                visiting[node] = true;
                path.push(node);
                current = parent(node);
            };
            for node in path {
                nearest[node] = found;
                resolved[node] = true;
            }
        }
        (0..self.rows.len())
            .map(|index| {
                parent(index)
                    .and_then(|parent| nearest[parent])
                    .filter(|found| *found != index)
            })
            .collect()
    }

    pub(super) fn descendants(&self, index: usize) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::Graph;
    use crate::query::named::TraceSpansRow;

    fn row(id: &str, parent: &str) -> TraceSpansRow {
        serde_json::from_value(serde_json::json!({
            "span_id": id,
            "parent_span_id": parent,
            "name": id,
            "type": "chain",
            "agent": "",
            "status": "STATUS_CODE_OK",
            "status_message": "",
            "error_truncated": 0,
            "start_ns": 0,
            "duration_ns": 0,
            "service": "",
            "input_preview": "",
            "model": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "litellm_request_id": "",
            "team_id": "",
            "api_key_hash": "",
            "user_id": ""
        }))
        .unwrap()
    }

    #[fixture]
    fn unordered_rows() -> Vec<TraceSpansRow> {
        vec![
            row("leaf", "middle"),
            row("sibling", "root"),
            row("middle", "root"),
            row("root", ""),
        ]
    }

    #[rstest]
    fn traversal_follows_links_instead_of_export_order(unordered_rows: Vec<TraceSpansRow>) {
        let graph = Graph::new(&unordered_rows);
        assert_eq!(graph.ancestors(0), [2, 3]);
        let descendants: std::collections::BTreeSet<&str> = graph
            .descendants(3)
            .into_iter()
            .map(|index| graph.id(index))
            .collect();
        assert_eq!(descendants, ["leaf", "middle", "sibling"].into());
        assert!(graph.is_root(3));
        assert!(!graph.is_root(0));
    }

    #[rstest]
    #[case::root(vec![false, false, false, true], vec![Some(3), Some(3), Some(3), None])]
    #[case::nearer(vec![false, false, true, true], vec![Some(2), Some(3), Some(3), None])]
    #[case::unassigned(vec![false; 4], vec![None; 4])]
    fn nearest_candidates_follow_unordered_links(
        unordered_rows: Vec<TraceSpansRow>,
        #[case] candidates: Vec<bool>,
        #[case] expected: Vec<Option<usize>>,
    ) {
        assert_eq!(
            Graph::new(&unordered_rows).nearest_session_ancestors(&candidates),
            expected
        );
    }

    #[rstest]
    #[case::unassigned(vec![false, false], vec![None, None])]
    #[case::one_candidate(vec![true, false], vec![None, Some(0)])]
    #[case::both_candidates(vec![true, true], vec![Some(1), Some(0)])]
    fn nearest_candidates_do_not_revisit_cycles(
        #[case] candidates: Vec<bool>,
        #[case] expected: Vec<Option<usize>>,
    ) {
        let rows = [row("first", "second"), row("second", "first")];
        assert_eq!(
            Graph::new(&rows).nearest_session_ancestors(&candidates),
            expected
        );
    }

    #[rstest]
    #[case::missing_parent("missing", &[], &[1])]
    #[case::self_link("first", &[], &[1])]
    #[case::cycle("second", &[1], &[1])]
    fn traversal_stops_at_missing_parents_and_cycles(
        #[case] parent: &str,
        #[case] ancestors: &[usize],
        #[case] descendants: &[usize],
    ) {
        let rows = [row("first", parent), row("second", "first")];
        let graph = Graph::new(&rows);
        assert_eq!(graph.ancestors(0), ancestors);
        assert_eq!(graph.descendants(0), descendants);
        assert_eq!(graph.parent(0), ancestors.first().copied());
    }
}
