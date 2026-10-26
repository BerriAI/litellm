use std::collections::BTreeMap;

use litellm_cost::TokenPriceSources;
use litellm_cost::price::{Adjustments, Prices};

use crate::error::Located;
use crate::wire::{Attributes, EntryWire, Fields};

/// One catalog entry read into types.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub attributes: Attributes,
    pub prices: Prices,
    pub token_price_sources: TokenPriceSources,
    pub adjustments: Adjustments,
    pub capabilities: BTreeMap<String, bool>,
    pub unrecognized: Vec<String>,
}

#[derive(Clone, Copy, Default, strum::VariantArray)]
pub enum PriceLowering {
    #[default]
    Generic,
    SdkTokens,
}

impl Model {
    pub fn from_fields(fields: &Fields) -> Result<Self, Vec<Located>> {
        Self::from_fields_with_policy(fields, PriceLowering::Generic)
    }

    pub fn from_fields_with_policy(
        fields: &Fields,
        policy: PriceLowering,
    ) -> Result<Self, Vec<Located>> {
        Self::lower(EntryWire::read(fields), policy)
    }
}
