use std::collections::HashMap;

use super::Rows;

/// Keys only, never rows, so a value write never makes it stale. A later key wins when two keys fold to one name
#[derive(Debug, Default)]
pub(super) struct NameIndex {
    by_folded: HashMap<String, String>,
    by_group: HashMap<String, Vec<String>>,
}

impl NameIndex {
    pub(super) fn scan<R: Rows>(rows: &R) -> Self {
        let mut index = Self::default();
        rows.scan(&mut |key, group| {
            index.by_folded.insert(R::fold(key), key.to_owned());
            if let Some(group) = group {
                index
                    .by_group
                    .entry(group.to_owned())
                    .or_default()
                    .push(key.to_owned());
            }
        });
        index
    }

    pub(super) fn key_for(&self, folded: &str) -> Option<&str> {
        self.by_folded.get(folded).map(String::as_str)
    }

    pub(super) fn keys_in(&self, group: &str) -> &[String] {
        self.by_group.get(group).map_or(&[], Vec::as_slice)
    }
}
