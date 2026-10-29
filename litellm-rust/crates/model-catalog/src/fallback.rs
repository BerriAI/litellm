use fancy_regex::{Regex, RegexBuilder};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::wire::Fields;

/// One regex rule generalizing unknown model ids to known families.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
pub struct FallbackRule {
    pub name: String,
    pub pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Regex rules that generalize unknown model ids to known families; not a model entry.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct FallbackGeneralizations {
    pub rules: Vec<FallbackRule>,
}

#[derive(Clone, Debug, Default)]
pub struct CapabilityRules(Vec<CapabilityRule>);

#[derive(Clone, Debug)]
struct CapabilityRule {
    pattern: Regex,
    model_info: Fields,
    fill_missing_for_providers: Vec<String>,
}

impl CapabilityRules {
    pub fn new(rules: &[FallbackRule]) -> Self {
        Self(rules.iter().filter_map(CapabilityRule::compile).collect())
    }

    pub fn matching(&self, model: &str) -> Option<Fields> {
        let matched: Vec<&CapabilityRule> =
            self.0.iter().filter(|rule| rule.matches(model)).collect();
        (!matched.is_empty()).then(|| merged(matched))
    }

    pub fn filling(&self, model: &str, provider: &str) -> Fields {
        merged(
            self.0
                .iter()
                .filter(|rule| rule.fills(provider) && rule.matches(model)),
        )
    }
}

impl CapabilityRule {
    fn compile(rule: &FallbackRule) -> Option<Self> {
        let Some(Value::Object(model_info)) = rule.extra.get("model_info") else {
            return None;
        };
        if rule.extra.contains_key("extends") || model_info.contains_key("litellm_provider") {
            return None;
        }
        let fill_missing_for_providers = match rule.extra.get("fill_missing_for_providers") {
            Some(providers) => Vec::<String>::deserialize(providers).ok()?,
            None => Vec::new(),
        };
        Some(Self {
            pattern: RegexBuilder::new(&rule.pattern)
                .case_insensitive(true)
                .build()
                .ok()?,
            model_info: model_info.clone(),
            fill_missing_for_providers,
        })
    }

    fn matches(&self, model: &str) -> bool {
        self.pattern.is_match(model).unwrap_or(false)
    }

    fn fills(&self, provider: &str) -> bool {
        self.fill_missing_for_providers
            .iter()
            .any(|listed| listed == provider)
    }
}

fn merged<'a>(rules: impl IntoIterator<Item = &'a CapabilityRule>) -> Fields {
    rules
        .into_iter()
        .flat_map(|rule| rule.model_info.clone())
        .collect()
}
