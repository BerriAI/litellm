use std::collections::BTreeMap;

use litellm_cost::price::{Adjustments, Prices};

use crate::error::Located;
use crate::wire::{Attributes, EntryWire, Fields};

/// One catalog entry read into types.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub attributes: Attributes,
    pub prices: Prices,
    pub adjustments: Adjustments,
    pub capabilities: BTreeMap<String, bool>,
    pub unrecognized: Vec<String>,
}

impl Model {
    pub fn from_fields(fields: &Fields) -> Result<Self, Vec<Located>> {
        Self::try_from(EntryWire::read(fields))
    }
}
