use serde_json::{Map, Value};

use crate::billing::BilledAmount;

/// OpenRouter's extension of a format's usage record: the amount it billed for the call in
/// US dollars and the upstream breakdown behind it. Present on complete responses and on the
/// final usage of a stream.
#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct OpenRouterUsage {
    pub cost: Option<BilledAmount>,
    pub cost_details: Option<OpenRouterCostDetails>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct OpenRouterCostDetails {
    pub upstream_inference_cost: Option<BilledAmount>,
    pub upstream_inference_prompt_cost: Option<BilledAmount>,
    pub upstream_inference_completions_cost: Option<BilledAmount>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
