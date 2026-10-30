//! The configured deployments, indexed by id, model group, alias, wildcard pattern, routing
//! group and team. A [`Registry`] is immutable; [`Registry::apply`] builds the next version.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write as _,
    sync::Arc,
};

use litellm_router_types::{LitellmParams, Spelled};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::ConfigError;

/// A registry's version; every change produces a higher one.
pub type RegistryVersion = u64;

/// A deployment as a config or an `apply` spells it.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct DeploymentSpec {
    pub model_name: String,
    pub litellm_params: Map<String, Value>,
    #[serde(default)]
    pub model_info: Map<String, Value>,
}

/// `routing_groups`: a named subset of model groups sharing a routing strategy.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RoutingGroupSpec {
    pub group_name: String,
    pub models: Vec<String>,
    pub routing_strategy: crate::config::RoutingStrategy,
}

/// The `model_info` fields the router reads; every other key stays in [`Self::raw`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelInfo {
    pub team_id: Option<String>,
    pub team_public_model_name: Option<String>,
    pub access_groups: Vec<String>,
    pub blocked: bool,
    pub cooldown_time: Option<f64>,
    pub allowed_fails: Option<u32>,
    pub raw: Map<String, Value>,
}

impl ModelInfo {
    fn from_raw(raw: Map<String, Value>) -> Self {
        let text = |key: &str| raw.get(key).and_then(Value::as_str).map(str::to_owned);
        Self {
            team_id: text("team_id"),
            team_public_model_name: text("team_public_model_name"),
            access_groups: raw
                .get("access_groups")
                .and_then(Value::as_array)
                .map(|groups| {
                    groups
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            blocked: raw.get("blocked").and_then(Value::as_bool) == Some(true),
            cooldown_time: raw.get("cooldown_time").and_then(Value::as_f64),
            allowed_fails: raw
                .get("allowed_fails")
                .and_then(Value::as_u64)
                .and_then(|fails| u32::try_from(fails).ok()),
            raw,
        }
    }
}

/// One configured deployment.
#[derive(Clone, Debug, PartialEq)]
pub struct Deployment {
    pub id: String,
    pub model_name: String,
    pub params: LitellmParams,
    /// `litellm_params` as written, in order: what the id hashes and the host receives.
    pub raw_params: Map<String, Value>,
    pub model_info: ModelInfo,
}

impl Deployment {
    pub fn from_spec(spec: DeploymentSpec) -> Result<Self, ConfigError> {
        let params: LitellmParams = serde_json::from_value(Value::Object(
            spec.litellm_params.clone(),
        ))
        .map_err(|source| ConfigError::InvalidDeployment {
            model_name: spec.model_name.clone(),
            source,
        })?;
        let id = match spec.model_info.get("id") {
            Some(Value::String(id)) => id.clone(),
            Some(Value::Number(id)) => id.to_string(),
            _ => deployment_id(&spec.model_name, &spec.litellm_params),
        };
        Ok(Self {
            id,
            model_name: spec.model_name,
            params,
            raw_params: spec.litellm_params,
            model_info: ModelInfo::from_raw(spec.model_info),
        })
    }

    /// The provider: `custom_llm_provider`, else the `model` prefix before `/`.
    pub fn provider(&self) -> &str {
        self.params
            .custom_llm_provider
            .as_deref()
            .or_else(|| {
                self.params
                    .model
                    .split_once('/')
                    .map(|(provider, _)| provider)
            })
            .unwrap_or("openai")
    }

    /// A numeric `litellm_params` value, as `simple_shuffle._metric_weight` reads it.
    pub fn metric(&self, key: &str) -> Option<f64> {
        let spelled = |value: &Option<Spelled<f64>>| match value {
            Some(Spelled::Value(value)) => Some(*value),
            Some(Spelled::Text(_)) | None => None,
        };
        match key {
            "rpm" => spelled(&self.params.rpm),
            "tpm" => spelled(&self.params.tpm),
            _ => self.raw_params.get(key).and_then(Value::as_f64),
        }
    }

    /// `litellm_params.order` (`get_deployment_order`).
    pub fn order(&self) -> Option<i64> {
        self.raw_params.get("order").and_then(Value::as_i64)
    }

    /// Whether `team_id` may use this deployment (`deployment_usable_by_team`).
    pub fn usable_by_team(&self, team_id: Option<&str>) -> bool {
        self.model_info
            .team_id
            .as_deref()
            .is_none_or(|owner| Some(owner) == team_id)
    }

    /// Whether the router counts per-minute usage for this deployment
    /// (`_deployment_tracks_usage`, without the model-catalog limits).
    pub fn tracks_usage(&self) -> bool {
        self.params.tpm.is_some()
            || self.params.rpm.is_some()
            || self.params.itpm.is_some()
            || self.params.otpm.is_some()
    }
}

/// The connection params of one deployment (`credentials(id)`).
#[derive(Clone, Debug, PartialEq)]
pub struct Credentials {
    pub provider: String,
    pub api_base: Option<String>,
    pub api_version: Option<String>,
    pub credential_name: Option<String>,
    pub params: LitellmParams,
}

/// What a name resolved through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveKind {
    SpecificDeployment,
    Id,
    Alias,
    RoutingGroup,
    Group,
    Wildcard,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub kind: ResolveKind,
    /// The model group the deployments are routed as.
    pub group: String,
    pub deployments: Vec<Arc<Deployment>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResolveScope {
    /// The caller named a deployment's `litellm_params.model` (`specific_deployment=True`).
    pub specific_deployment: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DeploymentFilter<'a> {
    pub model_name: Option<&'a str>,
    pub team_id: Option<&'a str>,
    pub provider: Option<&'a str>,
}

#[derive(Clone, Debug, Default)]
pub struct NameFilter {
    pub include_aliases: bool,
}

/// One atomic change: upserts, deletes and, optionally, a new dynamic config.
#[derive(Clone, Debug, Default)]
pub struct ChangeSet {
    pub upserts: Vec<DeploymentSpec>,
    pub deletes: Vec<String>,
    pub settings: Option<crate::config::RouterDynamicConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct Registry {
    version: RegistryVersion,
    deployments: Vec<Arc<Deployment>>,
    by_id: HashMap<String, usize>,
    by_group: BTreeMap<String, Vec<usize>>,
    by_team: HashMap<String, Vec<usize>>,
    wildcards: Vec<(String, Vec<usize>)>,
    aliases: BTreeMap<String, String>,
    routing_groups: Vec<RoutingGroupSpec>,
}

impl Registry {
    pub fn build(
        specs: Vec<DeploymentSpec>,
        aliases: BTreeMap<String, String>,
        routing_groups: Vec<RoutingGroupSpec>,
    ) -> Result<Self, ConfigError> {
        let deployments = specs
            .into_iter()
            .map(|spec| Deployment::from_spec(spec).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = BTreeSet::new();
        if let Some(duplicate) = deployments.iter().find(|d| !seen.insert(d.id.as_str())) {
            return Err(ConfigError::DuplicateId {
                id: duplicate.id.clone(),
            });
        }
        Ok(Self::index(0, deployments, aliases, routing_groups))
    }

    fn index(
        version: RegistryVersion,
        deployments: Vec<Arc<Deployment>>,
        aliases: BTreeMap<String, String>,
        routing_groups: Vec<RoutingGroupSpec>,
    ) -> Self {
        let mut by_id = HashMap::new();
        let mut by_group: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut by_team: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, deployment) in deployments.iter().enumerate() {
            by_id.insert(deployment.id.clone(), index);
            by_group
                .entry(deployment.model_name.clone())
                .or_default()
                .push(index);
            if let Some(team) = &deployment.model_info.team_id {
                by_team.entry(team.clone()).or_default().push(index);
            }
        }
        // In the order each pattern first appears, then most specific first, as
        // `PatternUtils.sorted_patterns` keeps them: a stable sort, longest regex first,
        // then the most regex syntax.
        let mut wildcards: Vec<(String, Vec<usize>)> = Vec::new();
        for deployment in &deployments {
            let name = &deployment.model_name;
            if name.contains('*') && !wildcards.iter().any(|(pattern, _)| pattern == name) {
                wildcards.push((
                    name.clone(),
                    by_group.get(name).cloned().unwrap_or_default(),
                ));
            }
        }
        wildcards.sort_by_key(|(pattern, _)| std::cmp::Reverse(pattern_specificity(pattern)));
        Self {
            version,
            deployments,
            by_id,
            by_group,
            by_team,
            wildcards,
            aliases,
            routing_groups,
        }
    }

    pub fn version(&self) -> RegistryVersion {
        self.version
    }

    /// The next registry: deletes, then upserts. An upserted deployment whose id exists is
    /// removed and appended at the end, as the Python router does on an edit.
    pub fn apply(
        &self,
        changes: &ChangeSet,
        aliases: Option<BTreeMap<String, String>>,
    ) -> Result<Self, ConfigError> {
        let mut deployments = self.deployments.clone();
        deployments.retain(|deployment| !changes.deletes.contains(&deployment.id));
        for spec in &changes.upserts {
            let deployment = Deployment::from_spec(spec.clone())?;
            deployments.retain(|existing| existing.id != deployment.id);
            deployments.push(Arc::new(deployment));
        }
        Ok(Self::index(
            self.version + 1,
            deployments,
            aliases.unwrap_or_else(|| self.aliases.clone()),
            self.routing_groups.clone(),
        ))
    }

    fn at(&self, indices: &[usize]) -> Vec<Arc<Deployment>> {
        indices
            .iter()
            .map(|index| self.deployments[*index].clone())
            .collect()
    }

    pub fn deployment(&self, id: &str) -> Option<Arc<Deployment>> {
        self.by_id
            .get(id)
            .map(|index| self.deployments[*index].clone())
    }

    pub fn deployments(&self, filter: &DeploymentFilter<'_>) -> Vec<Arc<Deployment>> {
        let candidates = match (filter.model_name, filter.team_id) {
            (Some(name), _) => self
                .by_group
                .get(name)
                .map(|i| self.at(i))
                .unwrap_or_default(),
            (None, Some(team)) => self
                .by_team
                .get(team)
                .map(|i| self.at(i))
                .unwrap_or_default(),
            (None, None) => self.deployments.clone(),
        };
        candidates
            .into_iter()
            .filter(|d| {
                filter
                    .team_id
                    .is_none_or(|team| d.model_info.team_id.as_deref() == Some(team))
            })
            .filter(|d| {
                filter
                    .provider
                    .is_none_or(|provider| d.provider() == provider)
            })
            .collect()
    }

    /// The model group names, in name order, with aliases when asked.
    pub fn names(&self, filter: &NameFilter) -> Vec<String> {
        let mut names: Vec<String> = self.by_group.keys().cloned().collect();
        if filter.include_aliases {
            names.extend(self.aliases.keys().cloned());
        }
        names
    }

    pub fn group(&self, name: &str) -> Option<Vec<Arc<Deployment>>> {
        self.by_group.get(name).map(|indices| self.at(indices))
    }

    /// The routing group a model group belongs to, if any.
    pub fn routing_group_of(&self, model_name: &str) -> Option<&RoutingGroupSpec> {
        self.routing_groups
            .iter()
            .find(|group| group.models.iter().any(|model| model == model_name))
    }

    /// The strategy a routing group named `group_name` declares.
    pub fn resolve_routing_group_strategy(
        &self,
        group_name: &str,
    ) -> Option<crate::config::RoutingStrategy> {
        self.routing_groups
            .iter()
            .find(|group| group.group_name == group_name)
            .map(|group| group.routing_strategy.clone())
    }

    pub(crate) fn routing_groups(&self) -> &[RoutingGroupSpec] {
        &self.routing_groups
    }

    /// Resolve a requested name, in the order of `_common_checks_available_deployment`:
    /// a specific deployment, an id, an alias, a routing group, a model group, a wildcard.
    pub fn resolve(&self, name: &str, scope: ResolveScope) -> Option<Resolution> {
        let resolution = |kind, group: &str, deployments: Vec<Arc<Deployment>>| {
            (!deployments.is_empty()).then(|| Resolution {
                kind,
                group: group.to_owned(),
                deployments,
            })
        };
        if scope.specific_deployment {
            let matching = self
                .deployments
                .iter()
                .filter(|d| d.params.model == name)
                .cloned()
                .collect();
            return resolution(ResolveKind::SpecificDeployment, name, matching);
        }
        if !self.by_group.contains_key(name)
            && let Some(deployment) = self.deployment(name)
        {
            let group = deployment.model_name.clone();
            return resolution(ResolveKind::Id, &group, vec![deployment]);
        }
        if let Some(target) = self.aliases.get(name) {
            return self
                .group(target)
                .and_then(|deployments| resolution(ResolveKind::Alias, target, deployments));
        }
        if let Some(routing_group) = self.routing_groups.iter().find(|g| g.group_name == name) {
            let deployments = routing_group
                .models
                .iter()
                .filter_map(|model| self.by_group.get(model))
                .flat_map(|indices| self.at(indices))
                .collect();
            return resolution(ResolveKind::RoutingGroup, name, deployments);
        }
        if let Some(indices) = self.by_group.get(name) {
            return resolution(ResolveKind::Group, name, self.at(indices));
        }
        self.wildcards
            .iter()
            .find(|(pattern, _)| wildcard_matches(pattern, name))
            .and_then(|(pattern, indices)| {
                resolution(ResolveKind::Wildcard, pattern, self.at(indices))
            })
    }

    /// Access group name to the model groups in it.
    pub fn access_groups(&self) -> BTreeMap<String, BTreeSet<String>> {
        let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for deployment in &self.deployments {
            for group in &deployment.model_info.access_groups {
                groups
                    .entry(group.clone())
                    .or_default()
                    .insert(deployment.model_name.clone());
            }
        }
        groups
    }

    pub fn credentials(&self, id: &str) -> Option<Credentials> {
        self.deployment(id).map(|deployment| Credentials {
            provider: deployment.provider().to_owned(),
            api_base: deployment.params.api_base.clone(),
            api_version: deployment.params.api_version.clone(),
            credential_name: deployment.params.litellm_credential_name.clone(),
            params: deployment.params.clone(),
        })
    }
}

/// `PatternUtils.calculate_pattern_specificity` over the pattern's regex
/// (`re.escape(pattern).replace(r"\*", "(.*)")`): its length, then how much regex syntax it has.
fn pattern_specificity(pattern: &str) -> (usize, usize) {
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
    let complexity = regex
        .chars()
        .filter(|c| matches!(c, '*' | '+' | '?' | '\\' | '^' | '$' | '|' | '(' | ')'))
        .count();
    (regex.chars().count(), complexity)
}

/// `*` matches any run of characters, as the pattern router's `(.*)` does under `re.match`:
/// anchored at the start only, so the name may run on past the pattern's last literal.
fn wildcard_matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    for part in parts {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

/// `deployment_id` in `litellm/router_api.py`: sha256 over the model group and every
/// `litellm_params` key and value in order. A string is taken as is, a dict as `json.dumps`, and
/// anything else as Python's `str()`.
pub fn deployment_id(model_group: &str, litellm_params: &Map<String, Value>) -> String {
    let mut text = String::from(model_group);
    for (key, value) in litellm_params {
        text.push_str(key);
        match value {
            Value::String(value) => text.push_str(value),
            Value::Object(_) => json_dumps(value, &mut text),
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => {
                python_str(value, &mut text);
            }
        }
    }
    let digest = Sha256::digest(text.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for exponents in
/// `-4..16`, scientific otherwise.
fn python_float(value: f64, out: &mut String) {
    if value.is_nan() {
        out.push_str("nan");
        return;
    }
    if value.is_infinite() {
        out.push_str(if value > 0.0 { "inf" } else { "-inf" });
        return;
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    out.push_str(sign);
    if (-4..16).contains(&exponent) {
        if exponent < 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
            out.push_str(&digits);
        } else {
            let point = exponent as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', point - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
}

fn number(value: &serde_json::Number, out: &mut String) {
    match value.as_f64() {
        Some(float) if !value.is_i64() && !value.is_u64() => python_float(float, out),
        Some(_) | None => out.push_str(&value.to_string()),
    }
}

/// `json.dumps` with its defaults: `", "` and `": "` separators and `ensure_ascii`.
fn json_dumps(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(value) => number(value, out),
        Value::String(text) => json_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                json_dumps(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                json_string(key, out);
                out.push_str(": ");
                json_dumps(item, out);
            }
            out.push('}');
        }
    }
}

fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ch if ch.is_ascii() && !ch.is_ascii_control() => out.push(ch),
            ch => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// Python's `str()` of a JSON value: `repr` for the items of a list or dict.
fn python_str(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(flag) => out.push_str(if *flag { "True" } else { "False" }),
        Value::Number(value) => number(value, out),
        Value::String(text) => python_repr_str(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_str(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr_str(key, out);
                out.push_str(": ");
                python_str(item, out);
            }
            out.push('}');
        }
    }
}

fn python_repr_str(text: &str, out: &mut String) {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch if (ch as u32) < 0x20 || ch as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push(quote);
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn spec(model_name: &str, params: Value, info: Value) -> DeploymentSpec {
        serde_json::from_value(json!({
            "model_name": model_name, "litellm_params": params, "model_info": info,
        }))
        .unwrap()
    }

    /// Expected ids computed with `litellm.router_api.deployment_id`.
    #[rstest]
    #[case::strings(
        "g",
        json!({"model": "openai/gpt-4o", "api_key": "sk-1"}),
        "30256ba697cc98ce2aff32c3b73c4b90514cb199876aef429ffcd91ae826d025"
    )]
    #[case::scalars(
        "claude",
        json!({"model": "anthropic/claude-3", "rpm": 10, "timeout": 1.5, "stream": true, "x": null}),
        "645fd1504089c1084316573ccfc5dd1dd9622623b169026657484fb64a2c89a2"
    )]
    #[case::list_and_dict(
        "m",
        json!({"model": "azure/x", "tags": ["a", "b'c"], "extra_body": {"k": [1, 2.0, "é"], "z": {"q": false}}}),
        "64f11e16dabed321268bd92132de2020290ef193c0cf554b2c74faee8badbb96"
    )]
    fn deployment_id_matches_python(
        #[case] group: &str,
        #[case] params: Value,
        #[case] expected: &str,
    ) {
        let Value::Object(params) = params else {
            unreachable!()
        };
        assert_eq!(deployment_id(group, &params), expected);
    }

    #[rstest]
    #[case::integral(2.0, "2.0")]
    #[case::small(0.00001, "1e-05")]
    #[case::large(1e16, "1e+16")]
    #[case::fraction(0.1, "0.1")]
    fn floats_print_as_python_repr(#[case] value: f64, #[case] expected: &str) {
        let mut out = String::new();
        python_float(value, &mut out);
        assert_eq!(out, expected);
    }

    #[rstest]
    #[case::group("gpt-4", ResolveKind::Group, "gpt-4")]
    #[case::alias("smart", ResolveKind::Alias, "gpt-4")]
    #[case::id("dep-b", ResolveKind::Id, "gpt-4")]
    #[case::wildcard("anthropic/claude-x", ResolveKind::Wildcard, "anthropic/*")]
    fn resolve_follows_python_order(
        #[case] name: &str,
        #[case] kind: ResolveKind,
        #[case] group: &str,
    ) {
        let registry = Registry::build(
            vec![
                spec(
                    "gpt-4",
                    json!({"model": "openai/gpt-4"}),
                    json!({"id": "dep-a"}),
                ),
                spec(
                    "gpt-4",
                    json!({"model": "azure/gpt-4"}),
                    json!({"id": "dep-b"}),
                ),
                spec("anthropic/*", json!({"model": "anthropic/*"}), json!({})),
            ],
            BTreeMap::from([("smart".to_owned(), "gpt-4".to_owned())]),
            Vec::new(),
        )
        .unwrap();
        let resolution = registry.resolve(name, ResolveScope::default()).unwrap();
        assert_eq!((resolution.kind, resolution.group.as_str()), (kind, group));
    }

    #[rstest]
    fn apply_moves_an_edited_deployment_to_the_end() {
        let registry = Registry::build(
            vec![
                spec("a", json!({"model": "openai/a"}), json!({"id": "1"})),
                spec("a", json!({"model": "openai/b"}), json!({"id": "2"})),
            ],
            BTreeMap::new(),
            Vec::new(),
        )
        .unwrap();
        let next = registry
            .apply(
                &ChangeSet {
                    upserts: vec![spec("a", json!({"model": "openai/c"}), json!({"id": "1"}))],
                    ..ChangeSet::default()
                },
                None,
            )
            .unwrap();
        let ids: Vec<String> = next
            .group("a")
            .unwrap()
            .iter()
            .map(|d| d.id.clone())
            .collect();
        assert_eq!(
            (next.version(), ids),
            (1, vec!["2".to_owned(), "1".to_owned()])
        );
    }
}
