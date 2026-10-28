mod image;

use std::borrow::Cow;

pub use image::*;

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
