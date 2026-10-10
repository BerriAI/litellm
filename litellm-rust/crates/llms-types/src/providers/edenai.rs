use serde_json::{Map, Value};

use crate::billing::BilledAmount;

/// Eden AI's extension of a response body: the amount it billed for the call as a top-level
/// `cost`, beside the format's own fields.
#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct EdenAIResponseExtension {
    pub cost: Option<BilledAmount>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
