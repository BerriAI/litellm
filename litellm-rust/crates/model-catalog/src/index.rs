use crate::{AliasIssue, Error};
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct ModelIndex<T> {
    pub entries: IndexMap<String, T>,
    pub aliases: IndexMap<String, String>,
    pub alias_issues: Vec<AliasIssue>,
    lowercase_keys: HashMap<String, String>,
}

pub(crate) struct IndexedMatch<'a, T> {
    pub matched_key: &'a str,
    pub canonical_key: &'a str,
    pub entry: &'a T,
}

pub(crate) struct ParsedIndex<T> {
    pub index: ModelIndex<T>,
    pub sample_spec: Option<Value>,
    pub fallback_generalizations: Option<Value>,
}

pub(crate) fn parse_index<T>(
    body: &[u8],
    entry: impl Fn(Map<String, Value>) -> Result<T, Error>,
) -> Result<ParsedIndex<T>, Error> {
    let root: IndexMap<String, Value> = serde_json::from_slice(body)?;
    if root.is_empty() {
        return Err(Error::Empty);
    }
    let mut entries = IndexMap::with_capacity(root.len());
    let mut alias_lists = Vec::new();
    let mut alias_issues = Vec::new();
    let mut sample_spec = None;
    let mut fallback_generalizations = None;
    for (name, value) in root {
        match name.as_str() {
            "sample_spec" => {
                sample_spec = Some(value);
                continue;
            }
            "fallback_generalizations" => {
                fallback_generalizations = Some(value);
                continue;
            }
            _ => {}
        }
        let Value::Object(mut fields) = value else {
            return Err(Error::EntryNotObject { model: name });
        };
        if let Some(aliases) = fields.remove("aliases") {
            match aliases {
                Value::Array(names) => alias_lists.push((name.clone(), names)),
                _ => alias_issues.push(AliasIssue::InvalidList {
                    model: name.clone(),
                }),
            }
        }
        entries.insert(name, entry(fields)?);
    }
    let mut aliases = IndexMap::new();
    for (model, names) in alias_lists {
        for name in names {
            let Value::String(alias) = name else {
                alias_issues.push(AliasIssue::InvalidName {
                    model: model.clone(),
                });
                continue;
            };
            if entries.contains_key(&alias) {
                alias_issues.push(AliasIssue::CanonicalCollision {
                    model: model.clone(),
                    alias,
                });
            } else if aliases.contains_key(&alias) {
                alias_issues.push(AliasIssue::AliasCollision {
                    model: model.clone(),
                    alias,
                });
            } else {
                aliases.insert(alias, model.clone());
            }
        }
    }
    let lowercase_keys = entries
        .keys()
        .chain(aliases.keys())
        .map(|key| (key.to_lowercase(), key.clone()))
        .collect();
    Ok(ParsedIndex {
        index: ModelIndex {
            entries,
            aliases,
            alias_issues,
            lowercase_keys,
        },
        sample_spec,
        fallback_generalizations,
    })
}

impl<T> ModelIndex<T> {
    pub fn lookup(&self, key: &str) -> Option<IndexedMatch<'_, T>> {
        let matched_key = if self.entries.contains_key(key) || self.aliases.contains_key(key) {
            key
        } else {
            self.lowercase_keys.get(&key.to_lowercase())?.as_str()
        };
        let canonical_key = self
            .aliases
            .get(matched_key)
            .map(String::as_str)
            .unwrap_or(matched_key);
        let (canonical_key, entry) = self.entries.get_key_value(canonical_key)?;
        let matched_key = self
            .entries
            .get_key_value(matched_key)
            .map(|(key, _)| key.as_str())
            .or_else(|| {
                self.aliases
                    .get_key_value(matched_key)
                    .map(|(key, _)| key.as_str())
            })?;
        Some(IndexedMatch {
            matched_key,
            canonical_key,
            entry,
        })
    }
}
