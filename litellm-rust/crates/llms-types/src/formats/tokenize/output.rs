use serde::{Deserialize, Serialize};
use strum::VariantArray;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, VariantArray)]
#[serde(rename_all = "snake_case")]
pub enum TokenizeOutput {
    Count,
    Ids,
    Tokens,
    Both,
    #[default]
    Available,
}

impl TokenizeOutput {
    pub fn is_available(&self) -> bool {
        *self == Self::Available
    }

    pub fn requires_ids(self) -> bool {
        matches!(self, Self::Ids | Self::Both)
    }

    pub fn requires_tokens(self) -> bool {
        matches!(self, Self::Tokens | Self::Both)
    }
}
