mod image;
mod provider_match;

use std::borrow::Cow;

pub use image::*;
pub use provider_match::*;
use serde_json::Value;

use crate::wire::Fields;

/// The live map as `resolve` sees it. The Python host implements it over the bound dict, tests over a `BTreeMap`
pub trait Keys {
    fn contains(&self, key: &str) -> bool;
    fn row(&self, key: &str) -> Option<RowRef<'_>>;
    /// The actual key for a case-insensitive name, from the one name index (decision 18)
    fn by_normalized(&self, name: &str) -> Option<String>;
}

/// A row handed back by `Keys`, decoded into the typed fields (decision 27)
pub type RowRef<'a> = Cow<'a, Fields>;

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

impl Candidates {
    /// The names in main's lookup order, each with the slot it fills
    pub fn in_order(&self) -> impl Iterator<Item = (CandidateSlot, &str)> {
        [
            (
                CandidateSlot::CombinedModelName,
                self.combined_model_name.as_deref(),
            ),
            (CandidateSlot::Model, Some(self.model.as_str())),
            (
                CandidateSlot::RegionFreeCombinedModelName,
                self.region_free_combined_model_name.as_deref(),
            ),
            (CandidateSlot::SplitModel, self.split_model.as_deref()),
            (
                CandidateSlot::CombinedStrippedModelName,
                self.combined_stripped_model_name.as_deref(),
            ),
            (
                CandidateSlot::StrippedModelName,
                self.stripped_model_name.as_deref(),
            ),
            (
                CandidateSlot::ProviderPrefixedModelName,
                self.provider_prefixed_model_name.as_deref(),
            ),
        ]
        .into_iter()
        .filter_map(|(slot, name)| Some((slot, name?)))
    }
}

/// main's candidate walk. Each name is tried as an exact key, then through the case-insensitive index, and the first
/// whose row passes `provider_matches` wins
pub fn resolve(keys: &impl Keys, candidates: &Candidates, provider: Option<&str>) -> Resolution {
    let found = candidates.in_order().find_map(|(slot, name)| {
        let (key, case_insensitive) = if keys.contains(name) {
            (name.to_owned(), false)
        } else {
            (keys.by_normalized(name)?, true)
        };
        let row = keys.row(&key)?;
        row_provider_matches(&row, provider).then_some(Resolution::Found {
            key,
            via: Via {
                slot,
                case_insensitive,
            },
        })
    });
    found.unwrap_or_else(|| Resolution::Missing {
        tried: candidates
            .in_order()
            .map(|(_, name)| name.to_owned())
            .collect(),
    })
}

fn row_provider_matches(row: &Fields, provider: Option<&str>) -> bool {
    match row.get("litellm_provider") {
        None | Some(Value::Null) => provider_matches(None, provider),
        Some(Value::String(row_provider)) => provider_matches(Some(row_provider), provider),
        Some(_) => provider_matches(Some(""), provider),
    }
}
