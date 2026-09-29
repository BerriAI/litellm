use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

/// Reasoning effort level accepted or applied by the model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, IntoStaticStr, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    pub const ALL: [Self; 7] = [
        Self::None,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
        Self::Max,
    ];

    pub fn as_str(self) -> &'static str {
        self.into()
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|effort| effort.as_str() == value)
    }
}
