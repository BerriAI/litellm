//! Fallback lists as Python spells them, and how a failed group's chain is found in one.
//! Mirrors `get_fallback_model_group`, `get_fallback_model_group_for_lookup_groups`,
//! `check_non_standard_fallback_format` and `fallback_attempt_key` in
//! `router_utils/fallback_event_handlers.py`.
//!
//! A list holds keyed entries (`{"gpt-4": ["gpt-3.5", {"model": "x", "temperature": 0}]}`),
//! bare names (`"claude"`, a chain for every group) and, in a client-side list, dict targets
//! (`{"model": "x", "messages": [...]}`), which override request params for their attempt.

use serde::Deserialize;
use serde_json::{Map, Value};

use std::collections::BTreeMap;

/// One item of a fallback list, kept as Python holds it.
#[derive(Clone, Debug, PartialEq)]
enum Entry {
    Name(String),
    Dict(Map<String, Value>),
    /// Anything else: Python skips it when it looks a chain up.
    Other,
}

/// A fallback list, in order. `"*"` keys the default chain.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(from = "Vec<Value>")]
pub struct Fallbacks(Vec<Entry>);

impl From<Vec<Value>> for Fallbacks {
    fn from(entries: Vec<Value>) -> Self {
        Self(
            entries
                .into_iter()
                .map(|entry| match entry {
                    Value::String(name) => Entry::Name(name),
                    Value::Object(dict) => Entry::Dict(dict),
                    _ => Entry::Other,
                })
                .collect(),
        )
    }
}

/// One target of a chain: a group name, or a dict target that also overrides request params.
#[derive(Clone, Debug, PartialEq)]
pub enum FallbackTarget {
    Name(String),
    Params(Map<String, Value>),
}

impl FallbackTarget {
    fn of(value: &Value) -> Option<Self> {
        match value {
            Value::String(name) => Some(Self::Name(name.clone())),
            Value::Object(dict) => Some(Self::Params(dict.clone())),
            _ => None,
        }
    }

    /// `_get_fallback_target_model_group`: the group the target names, if any.
    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Name(name) => Some(name),
            Self::Params(dict) => dict.get("model").and_then(Value::as_str),
        }
    }

    /// `fallback_attempt_key`: a bare name and `{"model": name}` are the same attempt; a dict
    /// carrying anything else is its own.
    pub fn identity(&self) -> FallbackTarget {
        match self {
            Self::Params(dict) if dict.len() == 1 => match dict.get("model") {
                Some(Value::String(name)) => Self::Name(name.clone()),
                _ => self.clone(),
            },
            _ => self.clone(),
        }
    }

    /// The params a dict target merges into the request (`kwargs.update(target)`).
    pub fn params(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Params(dict) => Some(dict),
            Self::Name(_) => None,
        }
    }
}

/// What fallback keys match besides the exact group name: Python's `provider_list`, for
/// `"openai/gpt-4o"` to use a `"gpt-4o"` entry, the provider-prefixed spelling of each bare
/// group name (`_provider_prefixed_model_group`), for `"gpt-4o"` to use an `"openai/gpt-4o"`
/// entry, and the `LiteLLMParamsTypedDict` keys that mark a client-side dict target.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct FallbackKeys {
    pub providers: Vec<String>,
    pub provider_prefixed: BTreeMap<String, String>,
    pub param_keys: Vec<String>,
}

impl FallbackKeys {
    /// `_check_stripped_model_group`.
    fn is_stripped(&self, group: &str, key: &str) -> bool {
        self.providers.iter().any(|provider| {
            let prefix = format!("{provider}/");
            group.starts_with(&prefix) && group.replace(&prefix, "") == key
        })
    }
}

/// A chain found for a group, and whether it is the `"*"` chain itself.
struct Found {
    targets: Vec<FallbackTarget>,
    generic: bool,
}

/// The targets of a keyed entry's value: a list of names and dicts. A bare string is walked
/// character by character, as Python iterates it.
fn chain_of(value: &Value) -> Vec<FallbackTarget> {
    match value {
        Value::Array(items) => items.iter().filter_map(FallbackTarget::of).collect(),
        Value::String(text) => text
            .chars()
            .map(|c| FallbackTarget::Name(c.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

impl Fallbacks {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `check_non_standard_fallback_format`: a client-side list of names, or of dicts that set
    /// a litellm param, is one chain for every group.
    pub fn is_client_side(&self, keys: &FallbackKeys) -> bool {
        if self.0.is_empty() {
            return false;
        }
        if self.0.iter().all(|entry| matches!(entry, Entry::Name(_))) {
            return true;
        }
        let dicts: Vec<_> = self
            .0
            .iter()
            .filter_map(|entry| match entry {
                Entry::Dict(dict) => Some(dict),
                _ => None,
            })
            .collect();
        dicts.len() == self.0.len()
            && dicts.iter().any(|dict| {
                keys.param_keys
                    .iter()
                    .any(|key| dict.get(key).is_some_and(|value| !value.is_array()))
            })
    }

    /// The whole list as one chain (a client-side list).
    pub fn as_chain(&self) -> Vec<FallbackTarget> {
        self.0
            .iter()
            .filter_map(|entry| match entry {
                Entry::Name(name) => Some(FallbackTarget::Name(name.clone())),
                Entry::Dict(dict) => Some(FallbackTarget::Params(dict.clone())),
                Entry::Other => None,
            })
            .collect()
    }

    /// `_get_first_default_fallback`: the first target of the first `"*"` entry.
    pub fn default_target(&self) -> Option<&str> {
        self.0.iter().find_map(|entry| match entry {
            Entry::Dict(dict) => match dict.get("*") {
                Some(Value::Array(targets)) => targets.first().and_then(Value::as_str),
                _ => None,
            },
            _ => None,
        })
    }

    /// `_has_default_fallbacks`: some entry keys `"*"`.
    pub fn has_star(&self) -> bool {
        self.0
            .iter()
            .any(|entry| matches!(entry, Entry::Dict(dict) if dict.contains_key("*")))
    }

    /// `get_fallback_model_group`.
    fn lookup(&self, group: &str, keys: &FallbackKeys) -> Option<Found> {
        let prefixed = keys.provider_prefixed.get(group).map(String::as_str);
        let mut generic = None;
        let mut stripped = None;
        let mut found = None;
        for entry in &self.0 {
            match entry {
                Entry::Dict(dict) => {
                    let Some((key, value)) = dict.iter().next() else {
                        continue;
                    };
                    if key == group {
                        found = Some(chain_of(value));
                        break;
                    } else if Some(key.as_str()) == prefixed || keys.is_stripped(group, key) {
                        stripped = Some(chain_of(value));
                    } else if key == "*" {
                        generic = Some(value);
                    }
                }
                Entry::Name(name) => found = Some(vec![FallbackTarget::Name(name.clone())]),
                Entry::Other => {}
            }
        }
        found
            .or(stripped)
            .map(|targets| Found {
                targets,
                generic: false,
            })
            .or_else(|| {
                generic.map(|value| Found {
                    targets: chain_of(value),
                    generic: true,
                })
            })
    }

    /// `get_fallback_model_group_for_lookup_groups`: the first lookup group with its own chain
    /// wins; `"*"` applies only after every group missed.
    pub fn lookup_groups(
        &self,
        groups: &[&str],
        keys: &FallbackKeys,
    ) -> Option<Vec<FallbackTarget>> {
        let results: Vec<Found> = groups
            .iter()
            .filter_map(|group| self.lookup(group, keys))
            .collect();
        let index = results
            .iter()
            .position(|found| !found.generic)
            .or_else(|| (!results.is_empty()).then_some(0))?;
        results.into_iter().nth(index).map(|found| found.targets)
    }

    /// `_has_fallback_group`'s view: only keyed entries whose targets are names or dicts count.
    pub fn has_keyed_chain(&self, group: &str, keys: &FallbackKeys) -> bool {
        let keyed = Fallbacks(
            self.0
                .iter()
                .filter(|entry| matches!(entry, Entry::Dict(_)))
                .cloned()
                .collect(),
        );
        keyed
            .lookup(group, keys)
            .is_some_and(|found| !found.targets.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn names(targets: Option<Vec<FallbackTarget>>) -> Option<Vec<String>> {
        targets.map(|targets| {
            targets
                .iter()
                .map(|t| t.model().unwrap_or_default().to_owned())
                .collect()
        })
    }

    #[rstest]
    #[case::own_entry(json!([{"a": ["b"]}, {"*": ["d"]}]), "a", Some(vec!["b"]))]
    #[case::star_entry(json!([{"a": ["b"]}, {"*": ["d"]}]), "x", Some(vec!["d"]))]
    #[case::string_is_every_groups_chain(json!(["b", {"x": ["c"]}]), "a", Some(vec!["b"]))]
    #[case::keyed_entry_wins(json!(["c", {"a": ["b"]}]), "a", Some(vec!["b"]))]
    #[case::string_value_is_iterated(json!([{"a": "bc"}]), "a", Some(vec!["b", "c"]))]
    #[case::none(json!([{"x": ["c"]}]), "a", None)]
    fn chains_follow_python(
        #[case] list: Value,
        #[case] group: &str,
        #[case] expected: Option<Vec<&str>>,
    ) {
        let fallbacks: Fallbacks = serde_json::from_value(list).unwrap();
        assert_eq!(
            names(fallbacks.lookup_groups(&[group], &FallbackKeys::default())),
            expected.map(|e| e.into_iter().map(str::to_owned).collect())
        );
    }

    #[rstest]
    #[case::names(json!(["a", "b"]), true)]
    #[case::param_dicts(json!([{"model": "a", "temperature": 1}]), true)]
    #[case::keyed(json!([{"a": ["b"]}]), false)]
    #[case::mixed(json!(["a", {"a": ["b"]}]), false)]
    fn client_side_lists_follow_python(#[case] list: Value, #[case] expected: bool) {
        let fallbacks: Fallbacks = serde_json::from_value(list).unwrap();
        let keys = FallbackKeys {
            param_keys: vec!["model".to_owned(), "temperature".to_owned()],
            ..FallbackKeys::default()
        };
        assert_eq!(fallbacks.is_client_side(&keys), expected);
    }
}
