use std::collections::BTreeMap;

use litellm_cost::price::{Adjustments, Prices};

use crate::error::Error;
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
    pub fn from_fields(fields: &Fields) -> Result<Self, Vec<Error>> {
        Self::try_from(EntryWire::read(fields))
    }
}
