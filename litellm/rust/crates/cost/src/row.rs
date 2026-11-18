//! What pricing reads from the Model Catalog: a lowered row by its `RowRef`, the ordered candidates of a call, and
//! the discount and margin rules of the config. `RowSource` is the only way `price`, `bound` and `rates` reach a
//! row, so this crate never parses the cost map and never merges rows itself

use std::collections::BTreeSet;

use crate::adjust::{Discounts, Margins};
use crate::price::{Adjustments, Prices, Uplift};
use crate::query::Candidate;

/// The rows and config one call is priced against. Model Catalog implements it over its per-call snapshot
pub trait RowSource {
    type Row: LoweredRow;

    /// The row `candidate` names, `None` when the catalog has no row for it
    fn resolve(&self, candidate: &Candidate) -> Option<RowRef>;
    fn row(&self, at: &RowRef) -> Option<&Self::Row>;
    fn discounts(&self) -> &Discounts;
    fn margins(&self) -> &Margins;
}

/// One catalog row lowered for pricing
pub trait LoweredRow {
    fn prices(&self) -> RowPrices<'_>;
    fn adjustments(&self) -> &Adjustments;
    fn facts(&self) -> &RowFacts;
}

/// A row as Model Catalog resolved it, with where it came from
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct RowRef {
    pub key: Box<str>,
    pub provenance: Provenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum Provenance {
    /// A row the cost map ships
    Catalog,
    /// A row `register_model` or a deployment's `model_info` wrote, under its model name or deployment id
    Registered,
    /// A row carrying a direction of another row, such as a Bedrock profile priced by its base model
    DirectionCarried,
    /// Pricing the caller passed for this call alone
    PerCall,
}

/// The prices main bills a row at. A row whose `tiered_pricing` prices input bills the whole call from the tier the
/// prompt picks, and an empty prompt picks no tier
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowPrices<'a> {
    Flat(&'a Prices),
    Tiered {
        tiers: &'a Prices,
        empty_prompt: &'a Prices,
    },
}

/// Row attributes pricing reads besides its prices
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowFacts {
    pub mode: RowMode,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    /// How main's `float()` read `provider_specific_entry` for the speed and geo uplifts
    pub provider_entry: ProviderEntry,
    pub declared: Declared,
    /// The row lists `tiered_pricing`, which main's search pricing reads ahead of a flat query rate as tiers by the
    /// results the request asks for. Billing lowering leaves result tiers out
    pub prices_by_results: bool,
    /// The row lists `guardrail_cost_per_unit` as a mapping of numbers, which main reads as a guardrail pricing row
    /// even when it prices none of a call's counters
    pub guardrail_units: bool,
    /// `search_queries_cost_per_query`, the flat search price Perplexity reads ahead of `search_context_cost_per_query`
    pub legacy_search_query_rate: Option<crate::money::UnitPrice>,
    /// `litellm_provider` as the row wrote it, which picks the providers whose Decisions calls bill per request
    pub provider: Option<Box<str>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowMode {
    #[default]
    Chat,
    ImageGeneration,
    ImageEdit,
    Other,
}

/// `provider_specific_entry` as main reads it. A value main's `float()` rejects never lowers into `Adjustments`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ProviderEntry {
    #[default]
    Readable,
    /// A truthy entry that is not a mapping, so reading any key raises
    NotAMapping,
    /// The uplifts whose value `float()` rejects
    Unreadable(BTreeSet<Uplift>),
}

/// Facts about the keys the row was written with, which lowering cannot recover from its prices. The auth predicates
/// read raw keys, so `rates` hands these back for the adapter to reproduce them
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Declared {
    /// `input_cost_per_token` or `output_cost_per_token` is a key of the row, null included
    pub token_rate_key: bool,
    /// A `*cost_per*` key holds a positive number, directly or one level down, or the row has `tiered_pricing`
    pub positive_cost_per_key: bool,
    /// The row has a `*cost_per*` key or `tiered_pricing`, whatever its value
    pub cost_per_key: bool,
    /// `ptu_count` and `cost_per_ptu_per_hour` are both set
    pub ptu_flat_cost: bool,
}

impl<'a> RowPrices<'a> {
    pub fn for_prompt(self, prompt_tokens: u64) -> &'a Prices {
        match self {
            Self::Flat(prices) => prices,
            Self::Tiered { empty_prompt, .. } if prompt_tokens == 0 => empty_prompt,
            Self::Tiered { tiers, .. } => tiers,
        }
    }

    /// The prices no tier replaced, the ones main reads by plain key
    pub fn flat(self) -> &'a Prices {
        match self {
            Self::Flat(prices) => prices,
            Self::Tiered { empty_prompt, .. } => empty_prompt,
        }
    }

    pub fn tiers(self) -> Option<&'a Prices> {
        match self {
            Self::Flat(_) => None,
            Self::Tiered { tiers, .. } => Some(tiers),
        }
    }
}

impl ProviderEntry {
    /// Whether main's `float()` raises reading `uplift` for a call that names it
    pub fn rejects(&self, uplift: &Uplift) -> bool {
        match self {
            Self::Readable => false,
            Self::NotAMapping => true,
            Self::Unreadable(uplifts) => uplifts.contains(uplift),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::call::{InferenceGeo, Speed};
    use crate::testing::{BASE, prices};

    #[rstest]
    #[case::empty_prompt_reads_the_flat_prices(0, 1.0)]
    #[case::a_prompt_reads_the_tiers(1, 2.0)]
    fn a_tiered_row_reads_its_tiers_for_any_prompt(
        #[case] prompt_tokens: u64,
        #[case] expected: f64,
    ) {
        let flat = prices(&[(
            crate::price::Meter::TEXT_INPUT,
            Default::default(),
            BASE,
            1.0,
        )]);
        let tiers = prices(&[(
            crate::price::Meter::TEXT_INPUT,
            Default::default(),
            BASE,
            2.0,
        )]);
        let row = RowPrices::Tiered {
            tiers: &tiers,
            empty_prompt: &flat,
        };
        let rate = row
            .for_prompt(prompt_tokens)
            .get(&crate::price::Price::online(
                crate::price::Meter::TEXT_INPUT,
                Default::default(),
            ));
        assert_eq!(rate.map(crate::money::UnitPrice::get), Some(expected));
        assert_eq!(row.flat(), &flat);
    }

    #[rstest]
    #[case::readable(ProviderEntry::Readable, false)]
    #[case::not_a_mapping(ProviderEntry::NotAMapping, true)]
    #[case::other_key_unreadable(
        ProviderEntry::Unreadable(BTreeSet::from([Uplift::Speed(Speed::Fast)])),
        false
    )]
    #[case::this_key_unreadable(
        ProviderEntry::Unreadable(BTreeSet::from([Uplift::InferenceGeo(InferenceGeo::new("us"))])),
        true
    )]
    fn only_an_unreadable_value_for_the_named_uplift_raises(
        #[case] entry: ProviderEntry,
        #[case] raises: bool,
    ) {
        assert_eq!(
            entry.rejects(&Uplift::InferenceGeo(InferenceGeo::new("US"))),
            raises
        );
    }
}
