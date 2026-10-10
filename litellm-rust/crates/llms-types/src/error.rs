use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0} is not a finite, non-negative billed amount")]
pub struct InvalidBilledAmount(pub(crate) Value);
