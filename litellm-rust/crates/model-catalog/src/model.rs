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
    SdkOther,
}

impl Model {
    pub fn from_fields_with_other_source(
        fields: &Fields,
        id: String,
    ) -> Result<(Self, litellm_cost::OtherPriceSource), Vec<Located>> {
        let wire = EntryWire::read_other(fields);
        let source = crate::other::other_price_source(fields, &wire, id);
        Self::lower(wire, PriceLowering::SdkOther).map(|model| (model, source))
    }

    pub fn from_fields(fields: &Fields) -> Result<Self, Vec<Located>> {
        Self::from_fields_with_policy(fields, PriceLowering::Generic)
    }

    pub fn from_fields_with_policy(
        fields: &Fields,
        policy: PriceLowering,
    ) -> Result<Self, Vec<Located>> {
        let wire = match policy {
            PriceLowering::SdkOther => EntryWire::read_other(fields),
            PriceLowering::Generic | PriceLowering::SdkTokens => EntryWire::read(fields),
        };
        Self::lower(wire, policy)
    }
}
