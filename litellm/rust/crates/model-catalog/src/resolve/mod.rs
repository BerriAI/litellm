mod image;

pub use image::*;

use serde_json::Value;

use crate::{Catalog, wire::Fields};

const BACKFILL_MODES: [&str; 2] = ["chat", "responses"];

/// The seven names main's `get_model_info` tries, built by Python, walked in this order by Rust
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
