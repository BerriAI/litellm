//! [`HostIndexes`]: the registry indexes the Python router hands out, kept the way it keeps them.
//!
//! The Python router keeps its indexes by history rather than deriving them from the deployment
//! list: `model_names` is the set of names `set_model_list` was given (invalid entries included),
//! `deployment_names` only ever grows, assigning `model_list` reindexes nothing, a removal shifts
//! the positions every index holds (a caller-written one included), and the wildcard pattern
//! routers keep the copy of a deployment they were given. The registry keeps one of these per
//! version and applies those rules on its own writes; the host reads them back as the live
//! objects `registry().index()`, `names()` and `patterns()` return, and hands a caller's write to
//! any of them back as a changeset ([`IndexWrite`]).
//!
//! Values are JSON so whatever a caller wrote reads back as written, junk included.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One wildcard pattern router (`PatternMatchRouter`): regex to the deployments it routes to,
/// most specific pattern first.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PatternRouter {
    pub patterns: Vec<(String, Vec<Value>)>,
}

/// The indexes, as the host reads and writes them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HostIndexes {
    /// `model_id_to_deployment_index_map`: id to position, in insertion order.
    pub id_map: Map<String, Value>,
    /// `model_name_to_deployment_indices`: name to its positions.
    pub model_name: Map<String, Value>,
    /// `team_model_to_deployment_indices`: `[team_id, public name, positions]`.
    pub team_model: Vec<(String, String, Value)>,
    /// `model_names`: a set, its members in the order they joined.
    pub model_names: Vec<Value>,
    pub deployment_names: Vec<Value>,
    pub team_public_model_names: Vec<String>,
    pub provider_default_ids: Vec<String>,
    /// `pattern_router`.
    pub patterns: PatternRouter,
    /// `team_pattern_routers`: team id to its router.
    pub team_patterns: Vec<(String, PatternRouter)>,
}

/// A host write to the indexes: each field it names replaces the one held.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct IndexWrite {
    pub id_map: Option<Map<String, Value>>,
    pub model_name: Option<Map<String, Value>>,
    pub team_model: Option<Vec<(String, String, Value)>>,
    pub model_names: Option<Vec<Value>>,
    pub deployment_names: Option<Vec<Value>>,
    pub team_public_model_names: Option<Vec<String>>,
    pub provider_default_ids: Option<Vec<String>>,
    pub patterns: Option<PatternRouter>,
    pub team_patterns: Option<Vec<(String, PatternRouter)>>,
}

/// A removal the Python router could not finish: an index held a position it cannot compare
/// (`_without_indices` raised `TypeError`). The indexes keep what it shifted before that.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("'>' not supported between instances of '{found}' and 'int'")]
pub struct RemovalError {
    pub found: &'static str,
}

impl HostIndexes {
    /// Apply a host write.
    pub fn write(&mut self, write: IndexWrite) {
        let IndexWrite {
            id_map,
            model_name,
            team_model,
            model_names,
            deployment_names,
            team_public_model_names,
            provider_default_ids,
            patterns,
            team_patterns,
        } = write;
        if let Some(value) = id_map {
            self.id_map = value;
        }
        if let Some(value) = model_name {
            self.model_name = value;
        }
        if let Some(value) = team_model {
            self.team_model = value;
        }
        if let Some(value) = model_names {
            self.model_names = unique(value);
        }
        if let Some(value) = deployment_names {
            self.deployment_names = value;
        }
        if let Some(value) = team_public_model_names {
            self.team_public_model_names = value;
        }
        if let Some(value) = provider_default_ids {
            self.provider_default_ids = value;
        }
        if let Some(value) = patterns {
            self.patterns = value;
        }
        if let Some(value) = team_patterns {
            self.team_patterns = value;
        }
    }

    /// `set_model_list` starting over: the id, name and team indexes and the team routers go;
    /// the deployment names, the wildcard router and the provider default ids stay.
    pub fn reset(&mut self) {
        self.id_map.clear();
        self.model_name.clear();
        self.team_model.clear();
        self.team_patterns.clear();
        self.team_public_model_names.clear();
    }

    /// `set_model_list`'s `model_names`: the names it was given.
    pub fn set_model_names(&mut self, names: Vec<String>) {
        self.model_names = unique(names.into_iter().map(Value::String).collect());
    }

    /// A deployment joined the list at `position` (`_add_deployment` then
    /// `_add_model_to_list_and_index_map`); `add` is `add_deployment`, which also names its group.
    pub fn added(&mut self, entry: &Map<String, Value>, position: usize, add: bool) {
        let params = entry.get("litellm_params").and_then(Value::as_object);
        let info = entry.get("model_info").and_then(Value::as_object);
        let model_name = entry.get("model_name").and_then(Value::as_str);
        let id = info
            .and_then(|info| info.get("id"))
            .filter(|id| !id.is_null());
        self.deployment_names.push(
            params
                .and_then(|params| params.get("model"))
                .cloned()
                .unwrap_or(Value::Null),
        );
        if let Some(name) = model_name.filter(|name| name.contains('*')) {
            self.patterns.add(name, Value::Object(entry.clone()));
            if let Some(id) = id.and_then(Value::as_str).filter(|id| !id.is_empty()) {
                self.provider_default_ids.push(id.to_owned());
            }
        }
        let team_id = info
            .and_then(|info| info.get("team_id"))
            .filter(|v| !v.is_null());
        let public = info
            .and_then(|info| info.get("team_public_model_name"))
            .filter(|v| !v.is_null());
        if let (Some(team_id), Some(Value::String(public))) = (team_id, public)
            && public.contains('*')
        {
            let team = python_str(team_id);
            let router = match self.team_patterns.iter_mut().find(|(t, _)| *t == team) {
                Some((_, router)) => router,
                None => {
                    self.team_patterns.push((team, PatternRouter::default()));
                    &mut self.team_patterns.last_mut().expect("just pushed").1
                }
            };
            router.add(public, Value::Object(entry.clone()));
        }
        let position = Value::from(position);
        if let Some(id) = id {
            self.id_map.insert(python_str(id), position.clone());
        }
        if let Some(name) = model_name.filter(|name| !name.is_empty()) {
            push_position(
                self.model_name
                    .entry(name.to_owned())
                    .or_insert_with(|| Value::Array(Vec::new())),
                &position,
            );
        }
        if add {
            let name = Value::String(model_name.unwrap_or_default().to_owned());
            if !self.model_names.contains(&name) {
                self.model_names.push(name);
            }
        }
        // `_update_team_model_index`: a truthy team id and public name.
        if let (Some(team_id), Some(public)) = (team_id, public)
            && truthy(team_id)
            && truthy(public)
        {
            let (team, public) = (python_str(team_id), python_str(public));
            if !self.team_public_model_names.contains(&public) {
                self.team_public_model_names.push(public.clone());
            }
            let held = match self
                .team_model
                .iter_mut()
                .find(|(t, p, _)| *t == team && *p == public)
            {
                Some((_, _, held)) => held,
                None => {
                    self.team_model
                        .push((team, public, Value::Array(Vec::new())));
                    &mut self.team_model.last_mut().expect("just pushed").2
                }
            };
            if let Value::Array(positions) = held
                && !positions.contains(&position)
            {
                positions.push(position);
            }
        }
    }

    /// The deployment `id` left the list from `position` (`_update_deployment_indices_after_removal`).
    pub fn removed(&mut self, id: &str, position: usize) -> Result<(), RemovalError> {
        let mut id_map = Map::new();
        for (key, held) in std::mem::take(&mut self.id_map) {
            if key == id {
                continue;
            }
            let shifted = shift(&held, position)?.unwrap_or(held);
            id_map.insert(key, shifted);
        }
        self.id_map = id_map;
        let names: Vec<String> = self.model_name.keys().cloned().collect();
        for name in names {
            let kept = without(&self.model_name[&name], position)?;
            if kept.is_empty() {
                self.model_name.remove(&name);
                self.model_names
                    .retain(|held| held.as_str() != Some(name.as_str()));
            } else {
                self.model_name.insert(name, Value::Array(kept));
            }
        }
        let mut team_model = Vec::new();
        let held = std::mem::take(&mut self.team_model);
        let mut rest = held.into_iter();
        for (team, public, positions) in rest.by_ref() {
            match without(&positions, position) {
                Ok(kept) if kept.is_empty() => {}
                Ok(kept) => team_model.push((team, public, Value::Array(kept))),
                Err(error) => {
                    team_model.push((team, public, positions));
                    team_model.extend(rest);
                    self.team_model = team_model;
                    return Err(error);
                }
            }
        }
        self.team_model = team_model;
        let mut publics: Vec<String> = Vec::new();
        for (_, public, _) in &self.team_model {
            if !publics.contains(public) {
                publics.push(public.clone());
            }
        }
        self.team_public_model_names = publics;
        self.patterns.remove(id);
        for (_, router) in &mut self.team_patterns {
            router.remove(id);
        }
        self.team_patterns
            .retain(|(_, router)| !router.patterns.is_empty());
        Ok(())
    }

    /// Whether the id map holds `id` (`_has_model_id`).
    pub fn has_id(&self, id: &str) -> bool {
        self.id_map.contains_key(id)
    }

    /// What the id map holds for `id`, as written (`model_id_to_deployment_index_map.get(id)`).
    pub fn id_entry(&self, id: &str) -> Option<&Value> {
        self.id_map.get(id)
    }

    /// What the name index holds for `name`, as written.
    pub fn named(&self, name: &str) -> Option<&Value> {
        self.model_name.get(name)
    }

    /// What the team index holds for `(team, public)`, as written.
    pub fn team_named(&self, team: &str, public: &str) -> Option<&Value> {
        self.team_model
            .iter()
            .find(|(t, p, _)| t == team && p == public)
            .map(|(_, _, positions)| positions)
    }

    /// `team_public_model_names_of`: the public names the team index holds for `team`.
    pub fn team_public_names_of(&self, team: &str) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for (t, public, _) in &self.team_model {
            if t == team && !names.contains(public) {
                names.push(public.clone());
            }
        }
        names
    }

    /// The position the id map holds for `id`, when it is a position.
    pub fn position_of(&self, id: &str) -> Option<usize> {
        self.id_map
            .get(id)
            .and_then(Value::as_u64)
            .and_then(|position| usize::try_from(position).ok())
    }
}

impl PatternRouter {
    /// `add_pattern`: a new regex joins in specificity order (a stable sort, most specific
    /// first); a known one gains the deployment.
    pub fn add(&mut self, pattern: &str, deployment: Value) {
        let regex = pattern_to_regex(pattern);
        if let Some((_, deployments)) = self.patterns.iter_mut().find(|(held, _)| *held == regex) {
            deployments.push(deployment);
            return;
        }
        self.patterns.push((regex, vec![deployment]));
        self.patterns
            .sort_by_key(|(regex, _)| std::cmp::Reverse(regex_specificity(regex)));
    }

    /// `remove_deployment`: every deployment with this id goes, and so does a pattern left empty.
    pub fn remove(&mut self, id: &str) {
        for (_, deployments) in &mut self.patterns {
            deployments.retain(|deployment| {
                deployment
                    .get("model_info")
                    .and_then(|info| info.get("id"))
                    .and_then(Value::as_str)
                    != Some(id)
            });
        }
        self.patterns
            .retain(|(_, deployments)| !deployments.is_empty());
    }

    /// `route`: the deployments of the first pattern `request` matches, each a copy whose
    /// `litellm_params.model` takes the parts the wildcards matched; `filtered` keeps to the
    /// patterns of those names.
    pub fn route(&self, request: &str, filtered: Option<&[String]>) -> Option<Vec<Value>> {
        let filtered: Option<Vec<String>> =
            filtered.map(|names| names.iter().map(|name| pattern_to_regex(name)).collect());
        for (regex, deployments) in &self.patterns {
            if filtered
                .as_ref()
                .is_some_and(|names| !names.contains(regex))
            {
                continue;
            }
            let Some(groups) = regex_match(regex, request) else {
                continue;
            };
            let mut routed = Vec::with_capacity(deployments.len());
            for deployment in deployments {
                let mut copy = deployment.clone();
                let model = deployment
                    .get("litellm_params")
                    .and_then(|params| params.get("model"));
                let Some(Value::String(model)) = model else {
                    // `deployment["litellm_params"]["model"]` raising ends the route with None.
                    return None;
                };
                let named = deployment_model_name(&groups, request, model);
                if let Some(params) = copy
                    .get_mut("litellm_params")
                    .and_then(Value::as_object_mut)
                {
                    params.insert("model".to_owned(), Value::String(named));
                }
                routed.push(copy);
            }
            return Some(routed);
        }
        None
    }

    /// `get_models_by_pattern`'s pick: the first (request, pattern) pair that matches, trying
    /// `model` against every pattern, then `provider/model`.
    pub fn models_by_pattern(&self, requests: &[String]) -> Option<Vec<String>> {
        for request in requests {
            for (regex, deployments) in &self.patterns {
                if let Some(groups) = regex_match(regex, request) {
                    return Some(
                        deployments
                            .iter()
                            .filter_map(|deployment| {
                                deployment
                                    .get("litellm_params")
                                    .and_then(|params| params.get("model"))
                                    .and_then(Value::as_str)
                                    .map(|model| deployment_model_name(&groups, request, model))
                            })
                            .collect(),
                    );
                }
            }
        }
        None
    }
}

/// `re.escape(pattern).replace(r"\*", "(.*)")`.
pub fn pattern_to_regex(pattern: &str) -> String {
    let mut regex = String::new();
    for c in pattern.chars() {
        match c {
            '*' => regex.push_str("(.*)"),
            '(' | ')' | '[' | ']' | '{' | '}' | '?' | '+' | '-' | '|' | '^' | '$' | '\\' | '.'
            | '&' | '~' | '#' | ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }
    regex
}

/// `PatternUtils.calculate_pattern_specificity`: the regex's length, then how much regex syntax
/// it has.
pub fn regex_specificity(regex: &str) -> (usize, usize) {
    let complexity = regex
        .chars()
        .filter(|c| matches!(c, '*' | '+' | '?' | '\\' | '^' | '$' | '|' | '(' | ')'))
        .count();
    (regex.chars().count(), complexity)
}

/// A pattern regex read back as its literal runs between wildcards, or None when it holds regex
/// syntax `pattern_to_regex` never writes (a caller wrote it).
fn literal_runs(regex: &str) -> Option<Vec<String>> {
    let mut runs = vec![String::new()];
    let mut chars = regex.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => runs.last_mut()?.push(chars.next()?),
            '(' => {
                let rest: String = chars.by_ref().take(3).collect();
                if rest != ".*)" {
                    return None;
                }
                runs.push(String::new());
            }
            '.' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | ')' => return None,
            other => runs.last_mut()?.push(other),
        }
    }
    Some(runs)
}

/// `re.match(regex, request)` for a pattern regex: anchored at the start only, each `(.*)`
/// greedy. The groups it captured, or None when it does not match.
pub fn regex_match(regex: &str, request: &str) -> Option<Vec<String>> {
    let runs = literal_runs(regex)?;
    let rest = request.strip_prefix(runs[0].as_str())?;
    let mut groups = Vec::new();
    greedy(&runs[1..], rest, &mut groups).then_some(groups)
}

/// Match `runs` (each a literal that follows a group) against `rest`, longest group first.
fn greedy(runs: &[String], rest: &str, groups: &mut Vec<String>) -> bool {
    let Some((literal, later)) = runs.split_first() else {
        return true;
    };
    let mut ends: Vec<usize> = rest.char_indices().map(|(at, _)| at).collect();
    ends.push(rest.len());
    for &end in ends.iter().rev() {
        let (group, after) = rest.split_at(end);
        let Some(after) = after.strip_prefix(literal.as_str()) else {
            continue;
        };
        groups.push(group.to_owned());
        if greedy(later, after, groups) {
            return true;
        }
        groups.pop();
    }
    false
}

/// `set_deployment_model_name`: the deployment's model with each wildcard replaced by the group
/// matched for it, the request itself when more groups matched than it has wildcards.
pub fn deployment_model_name(groups: &[String], request: &str, model: &str) -> String {
    if !model.contains('*') {
        return model.to_owned();
    }
    if groups.len() > model.matches('*').count() {
        return request.to_owned();
    }
    let mut named = model.to_owned();
    for group in groups {
        named = named.replacen('*', group, 1);
    }
    named
}

fn unique(values: Vec<Value>) -> Vec<Value> {
    let mut kept: Vec<Value> = Vec::with_capacity(values.len());
    for value in values {
        if !kept.contains(&value) {
            kept.push(value);
        }
    }
    kept
}

fn push_position(held: &mut Value, position: &Value) {
    if let Value::Array(positions) = held {
        positions.push(position.clone());
    }
}

/// Python's truthiness for a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
    }
}

/// Python's `str()` for an id or team id.
pub(crate) fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Null => "None".to_owned(),
        other => other.to_string(),
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `idx - 1 if idx > removal_idx else idx`: None keeps `held` as is.
fn shift(held: &Value, position: usize) -> Result<Option<Value>, RemovalError> {
    let removal = position as f64;
    let number = match held {
        Value::Bool(flag) => f64::from(u8::from(*flag)),
        Value::Number(number) => number.as_f64().unwrap_or_default(),
        other => {
            return Err(RemovalError {
                found: type_name(other),
            });
        }
    };
    if number <= removal {
        return Ok(None);
    }
    Ok(Some(match held {
        Value::Number(n) if n.is_i64() || n.is_u64() => Value::from(n.as_i64().unwrap_or(0) - 1),
        _ => Value::from(number - 1.0),
    }))
}

/// `_without_indices`: the positions without `position`, the later ones shifted down.
fn without(held: &Value, position: usize) -> Result<Vec<Value>, RemovalError> {
    let Value::Array(positions) = held else {
        return Ok(Vec::new());
    };
    let removal = Value::from(position);
    let mut kept = Vec::with_capacity(positions.len());
    for index in positions {
        if *index == removal {
            continue;
        }
        kept.push(shift(index, position)?.unwrap_or_else(|| index.clone()));
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn entry(name: &str, id: &str, model: &str) -> Map<String, Value> {
        serde_json::from_value(json!({
            "model_name": name,
            "litellm_params": {"model": model},
            "model_info": {"id": id},
        }))
        .unwrap()
    }

    #[test]
    fn removal_shifts_every_index_and_drops_emptied_names() {
        let mut indexes = HostIndexes::default();
        indexes.added(&entry("gpt", "a", "openai/a"), 0, false);
        indexes.added(&entry("gpt", "b", "openai/b"), 1, false);
        indexes.added(&entry("claude", "c", "anthropic/c"), 2, false);
        indexes.set_model_names(vec!["gpt".into(), "claude".into(), "broken".into()]);
        indexes.removed("a", 0).unwrap();
        assert_eq!(
            indexes.id_map,
            json!({"b": 0, "c": 1}).as_object().unwrap().clone()
        );
        assert_eq!(indexes.model_name["gpt"], json!([0]));
        indexes.removed("c", 1).unwrap();
        assert!(!indexes.model_name.contains_key("claude"));
        assert_eq!(indexes.model_names, vec![json!("gpt"), json!("broken")]);
        assert_eq!(indexes.deployment_names.len(), 3);
    }

    #[test]
    fn a_pattern_routes_with_the_matched_parts() {
        let mut router = PatternRouter::default();
        router.add(
            "llmengine/*",
            json!({"litellm_params": {"model": "openai/*"}, "model_info": {"id": "w"}}),
        );
        router.add(
            "*meta.llama3*",
            json!({"litellm_params": {"model": "bedrock/meta.llama3*"}, "model_info": {"id": "m"}}),
        );
        let routed = router.route("llmengine/foo", None).unwrap();
        assert_eq!(routed[0]["litellm_params"]["model"], "openai/foo");
        let routed = router.route("hello-world-meta.llama3-70b", None).unwrap();
        assert_eq!(
            routed[0]["litellm_params"]["model"],
            "hello-world-meta.llama3-70b"
        );
        assert_eq!(
            regex_match(r"a(.*)b(.*)", "axbyb").unwrap(),
            vec!["xby", ""]
        );
        router.remove("w");
        assert!(router.route("llmengine/foo", None).is_none());
    }
}
