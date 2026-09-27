use std::collections::HashMap;

use fancy_regex::{Regex, RegexBuilder};
use serde_json::{Map, Value};

const PROVIDER_KEY: &str = "litellm_provider";

#[derive(Clone, Debug)]
struct RoutingRule {
    pattern: Regex,
    provider: String,
}

#[derive(Clone, Debug)]
struct CapabilityRule {
    pattern: Regex,
    model_info: Map<String, Value>,
}

#[derive(Clone, Debug, Default)]
pub struct FallbackGeneralizations {
    routing_rules: Vec<RoutingRule>,
    capability_rules: Vec<CapabilityRule>,
}

fn resolve_legacy_extends(rules: &[Value]) -> Vec<Value> {
    let base_by_name: HashMap<&str, &Map<String, Value>> = rules
        .iter()
        .filter_map(|rule| {
            Some((
                rule.get("name")?.as_str()?,
                rule.get("model_info")?.as_object()?,
            ))
        })
        .collect();
    rules
        .iter()
        .map(|rule| {
            let parent = rule
                .get("extends")
                .and_then(Value::as_str)
                .and_then(|name| base_by_name.get(name));
            match (parent, rule.get("model_info").and_then(Value::as_object)) {
                (Some(parent), Some(own)) => {
                    let mut merged = (*parent).clone();
                    merged.extend(own.clone());
                    let mut resolved = rule.as_object().cloned().unwrap_or_default();
                    resolved.insert("model_info".to_owned(), Value::Object(merged));
                    Value::Object(resolved)
                }
                _ => rule.clone(),
            }
        })
        .collect()
}

fn fill_missing_is_malformed(rule: &Value) -> bool {
    rule.get("fill_missing_for_providers")
        .is_some_and(|providers| {
            !providers
                .as_array()
                .is_some_and(|providers| providers.iter().all(Value::is_string))
        })
}

fn compile_rule(rule: &Value) -> (Option<RoutingRule>, Option<CapabilityRule>) {
    let (Some(pattern), Some(model_info)) = (
        rule.get("pattern").and_then(Value::as_str),
        rule.get("model_info").and_then(Value::as_object),
    ) else {
        return (None, None);
    };
    let Ok(pattern) = RegexBuilder::new(pattern).case_insensitive(true).build() else {
        return (None, None);
    };
    if fill_missing_is_malformed(rule) {
        return (None, None);
    }
    let capability = || CapabilityRule {
        pattern: pattern.clone(),
        model_info: model_info.clone(),
    };
    match model_info.get(PROVIDER_KEY) {
        None => (None, Some(capability())),
        Some(Value::String(provider)) => {
            let routing = RoutingRule {
                pattern: pattern.clone(),
                provider: provider.clone(),
            };
            let capability = (model_info.len() > 1).then(capability);
            (Some(routing), capability)
        }
        Some(_) => (None, None),
    }
}

fn is_match(pattern: &Regex, model: &str) -> bool {
    pattern.is_match(model).unwrap_or(false)
}

impl FallbackGeneralizations {
    pub fn from_block(block: Option<&Value>) -> Self {
        let rules = block
            .and_then(|block| block.get("rules"))
            .and_then(Value::as_array)
            .map(|rules| resolve_legacy_extends(rules))
            .unwrap_or_default();
        let (routing_rules, capability_rules): (Vec<_>, Vec<_>) =
            rules.iter().map(compile_rule).unzip();
        Self {
            routing_rules: routing_rules.into_iter().flatten().collect(),
            capability_rules: capability_rules.into_iter().flatten().collect(),
        }
    }

    pub fn match_routing(&self, model: &str) -> Option<&str> {
        if model.is_empty() {
            return None;
        }
        self.routing_rules
            .iter()
            .find(|rule| is_match(&rule.pattern, model))
            .map(|rule| rule.provider.as_str())
    }

    pub fn match_capabilities(&self, model: &str) -> Option<Map<String, Value>> {
        if model.is_empty() {
            return None;
        }
        let matched: Vec<&Map<String, Value>> = self
            .capability_rules
            .iter()
            .filter(|rule| is_match(&rule.pattern, model))
            .map(|rule| &rule.model_info)
            .collect();
        (!matched.is_empty()).then(|| {
            matched
                .into_iter()
                .flat_map(|model_info| {
                    model_info
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                })
                .collect()
        })
    }
}
