use std::collections::BTreeMap;

use crate::{KeyColumns, KeyView};

/// The flat JSON object of the row of a key `/key/generate` made with no parameters.
pub const PLAIN_KEY_JSON: &str = include_str!("plain_key.json");

/// The row of a key `/key/generate` made with no parameters, stored under `hash`.
pub fn plain_key(hash: &str) -> KeyView {
    let row: KeyView = serde_json::from_str(PLAIN_KEY_JSON).expect("the fixture is a key row");
    KeyView {
        key: KeyColumns {
            token: hash.to_owned(),
            ..row.key
        },
        ..row
    }
}

/// Key rows held in memory, each under the hash its own `token` column holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeldKeys(BTreeMap<String, KeyView>);

impl HeldKeys {
    pub fn find(&self, hash: &str) -> Option<KeyView> {
        self.0.get(hash).cloned()
    }
}

impl FromIterator<KeyView> for HeldKeys {
    fn from_iter<I: IntoIterator<Item = KeyView>>(rows: I) -> Self {
        Self(
            rows.into_iter()
                .map(|row| (row.key.token.clone(), row))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{HeldKeys, plain_key};
    use crate::{KeyColumns, KeyView};

    #[test]
    fn a_held_key_is_found_under_its_own_hash_alone() {
        let blocked = KeyView {
            key: KeyColumns {
                blocked: Some(true),
                ..plain_key("hash-b").key
            },
            ..plain_key("hash-b")
        };
        let held: HeldKeys = [plain_key("hash-a"), blocked.clone()].into_iter().collect();

        assert_eq!(
            (
                held.find("hash-a"),
                held.find("hash-b"),
                held.find("hash-c")
            ),
            (Some(plain_key("hash-a")), Some(blocked), None)
        );
    }
}
