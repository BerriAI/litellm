mod bedrock;
#[cfg(test)]
mod candidate_walk;
mod image;
mod names;

pub use bedrock::{
    bedrock_base_model, bedrock_model_name_from_arn, is_bedrock_cross_region_profile_of,
    split_bedrock_region_path, strip_bedrock_routing_prefix,
};
pub use image::*;

use serde_json::Value;

use crate::fallback::CapabilityRules;
use crate::view::CatalogView;
use crate::{Catalog, wire::Fields};

const BACKFILL_MODES: [&str; 2] = ["chat", "responses"];

/// The seven names main's `get_model_info` tries, built by `Candidates::potential` and walked in `in_lookup_order`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Candidates {
    pub combined_model_name: Option<String>,
    pub model: String,
    pub region_free_combined_model_name: Option<String>,
    pub split_model: Option<String>,
    pub combined_stripped_model_name: Option<String>,
    pub stripped_model_name: Option<String>,
    pub provider_prefixed_model_name: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateSlot {
    CombinedModelName,
    Model,
    RegionFreeCombinedModelName,
    SplitModel,
    CombinedStrippedModelName,
    StrippedModelName,
    ProviderPrefixedModelName,
}

/// Which step matched, so a test can name the step that changed
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Via {
    pub slot: CandidateSlot,
    /// Matched through the case-insensitive index rather than an exact key
    pub case_insensitive: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    Found {
        key: String,
        via: Via,
    },
    /// Every candidate tried, in order
    Missing {
        tried: Vec<String>,
    },
}

impl Catalog {
    pub fn model_info(&self, names: &[&str], provider: &str) -> Option<Fields> {
        let registered = names.iter().find_map(|name| {
            let found = self.lookup(name)?;
            let entry_provider = found
                .entry
                .field("litellm_provider")
                .and_then(Value::as_str);
            provider_matches(entry_provider, provider).then(|| {
                self.with_missing_filled(
                    found.matched_key,
                    found.entry.fields(),
                    entry_provider.unwrap_or_default(),
                )
            })
        });
        if registered.is_some() || names.iter().any(|name| self.lookup(name).is_some()) {
            return registered;
        }
        names
            .iter()
            .find_map(|name| self.capability_rules().matching(name))
    }

    fn with_missing_filled(&self, key: &str, fields: &Fields, entry_provider: &str) -> Fields {
        let backfills = fields.get("mode").is_none_or(|mode| {
            mode.as_str()
                .is_some_and(|mode| BACKFILL_MODES.contains(&mode))
        });
        let missing = if backfills {
            self.capability_rules().filling(key, entry_provider)
        } else {
            Fields::new()
        };
        missing.into_iter().chain(fields.clone()).collect()
    }
}

/// What `get_model_info(model, provider)` finds in a table
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    /// The first of the seven names whose row's provider matches, as its key
    Row(String),
    /// None of the names is a key, and a capability rule matches with these fields
    Rule(Fields),
    Missing,
}

/// Port of the `get_model_info` walk for a known provider: the first of the seven names whose row's provider matches,
/// else a capability rule when none of the names is a key
pub fn lookup<V: CatalogView + ?Sized>(
    table: &V,
    rules: &CapabilityRules,
    model: &str,
    provider: &str,
) -> Found {
    let candidates = Candidates::potential(model, provider);
    let registered = candidates.in_lookup_order().find_map(|name| {
        let key = table.find_key(name)?;
        let row = table.fields(&key)?;
        provider_matches(
            row.get("litellm_provider").and_then(Value::as_str),
            provider,
        )
        .then_some(key)
    });
    if let Some(key) = registered {
        return Found::Row(key);
    }
    generalization(table, rules, &candidates).map_or(Found::Missing, Found::Rule)
}

/// Port of `_get_model_info_from_generalization`: rules lose to any name that is a key, whatever its provider
fn generalization<V: CatalogView + ?Sized>(
    table: &V,
    rules: &CapabilityRules,
    candidates: &Candidates,
) -> Option<Fields> {
    if candidates
        .in_lookup_order()
        .any(|name| table.find_key(name).is_some())
    {
        return None;
    }
    candidates
        .in_lookup_order()
        .find_map(|name| rules.matching(name))
}

/// Port of main's `_check_provider_match`: a row without a `litellm_provider` matches every provider
fn provider_matches(entry_provider: Option<&str>, provider: &str) -> bool {
    let Some(entry_provider) = entry_provider else {
        return true;
    };
    provider.is_empty()
        || entry_provider == provider
        || matches!(provider, "vertex_ai" | "fireworks_ai") && entry_provider.starts_with(provider)
        || provider.starts_with("bedrock") && entry_provider.starts_with("bedrock")
        || matches!(provider, "litellm_proxy" | "github" | "nadir")
        || provider == "azure_ai" && matches!(entry_provider, "azure" | "openai")
}
