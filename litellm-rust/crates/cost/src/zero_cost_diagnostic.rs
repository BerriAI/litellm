use serde_json::Value;

use crate::catalog::ModelInfoCatalog;
use crate::model_selection::ModelSelectionRequest;
use crate::responses_usage::ChatUsage;
use crate::usage_dispatch::get_usage_object;

pub const ZERO_COST_COUNTER_NAME: &str = "litellm_zero_cost_requests_total";

#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::Display, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ZeroCostReason {
    MissingPricingKey,
    PricingNotApplied,
    CostCalculationError,
}

impl ZeroCostReason {
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroCostDiagnostic {
    pub reason: ZeroCostReason,
    pub pricing_model: String,
    pub missing_pricing_keys: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroCostFindingRequest<'a> {
    pub model_selection: ModelSelectionRequest<'a>,
    pub logging_details: Option<&'a Value>,
    pub metadata: Option<&'a Value>,
    pub call_type: Option<&'a str>,
    pub response_cost: Option<f64>,
    pub calculation_failed: bool,
    pub cache_hit: bool,
}

pub fn is_unbilled_non_inference_call(
    call_type: Option<&str>,
    metadata: Option<&Value>,
    response: &Value,
) -> bool {
    if !matches!(
        call_type,
        Some(
            "get_responses"
                | "aget_responses"
                | "delete_responses"
                | "adelete_responses"
                | "cancel_responses"
                | "acancel_responses"
                | "list_input_items"
                | "alist_input_items"
                | "vector_store_create"
                | "avector_store_create"
                | "vector_store_retrieve"
                | "avector_store_retrieve"
                | "vector_store_list"
                | "avector_store_list"
                | "vector_store_update"
                | "avector_store_update"
                | "vector_store_delete"
                | "avector_store_delete"
                | "vector_store_file_create"
                | "avector_store_file_create"
                | "vector_store_file_list"
                | "avector_store_file_list"
                | "vector_store_file_retrieve"
                | "avector_store_file_retrieve"
                | "vector_store_file_content"
                | "avector_store_file_content"
                | "vector_store_file_update"
                | "avector_store_file_update"
                | "vector_store_file_delete"
                | "avector_store_file_delete"
        )
    ) || response.get("background") == Some(&Value::Bool(true))
    {
        return false;
    }
    metadata
        .and_then(|metadata| metadata.get("internal_call_origin"))
        .and_then(Value::as_str)
        != Some("background_response_cost_poll")
}

pub fn zero_cost_finding(
    catalog: &ModelInfoCatalog,
    request: ZeroCostFindingRequest<'_>,
) -> Option<(ZeroCostDiagnostic, String)> {
    if request.cache_hit || (request.response_cost.is_none() && !request.calculation_failed) {
        return None;
    }
    if request.response_cost.is_some_and(|cost| cost != 0.0) {
        return None;
    }
    let response = request.model_selection.response?;
    if is_unbilled_non_inference_call(request.call_type, request.metadata, response) {
        return None;
    }
    let usage = get_usage_object(response).ok().flatten()?;
    let pricing =
        catalog.pricing_entry_for_cost_calc(request.model_selection, request.logging_details)?;
    let diagnostic = diagnose_zero_cost(
        &usage,
        &pricing.key,
        &pricing.info,
        request.calculation_failed,
    )?;
    let model = request.model_selection.model.unwrap_or("None");
    let warning = zero_cost_warning(
        &diagnostic,
        request
            .metadata
            .and_then(|metadata| metadata.get("model_group"))
            .and_then(Value::as_str),
        model,
        request.model_selection.provider,
        &usage,
    );
    Some((diagnostic, warning))
}

pub fn used_pricing_keys(usage: &ChatUsage) -> Vec<&'static str> {
    let prompt_audio = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.audio_tokens)
        .unwrap_or(0);
    let completion_audio = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.audio_tokens)
        .unwrap_or(0);
    [
        (
            "input_cost_per_token",
            usage.prompt_tokens.saturating_sub(prompt_audio),
        ),
        ("input_cost_per_audio_token", prompt_audio),
        (
            "output_cost_per_token",
            usage.completion_tokens.saturating_sub(completion_audio),
        ),
        ("output_cost_per_audio_token", completion_audio),
    ]
    .into_iter()
    .filter_map(|(key, count)| (count > 0).then_some(key))
    .collect()
}

fn declares_rate(value: &Value, depth: u8) -> bool {
    if depth == 0 {
        return value.as_f64().is_some_and(|rate| rate > 0.0);
    }
    match value {
        Value::Object(fields) => fields
            .iter()
            .filter(|(key, _)| key.contains("cost") || key.contains("pricing"))
            .any(|(_, value)| declares_rate(value, depth - 1)),
        Value::Array(values) => values.iter().any(|value| declares_rate(value, depth - 1)),
        _ => value.as_f64().is_some_and(|rate| rate > 0.0),
    }
}

pub fn diagnose_zero_cost(
    usage: &ChatUsage,
    pricing_model: &str,
    pricing_entry: &Value,
    calculation_failed: bool,
) -> Option<ZeroCostDiagnostic> {
    let used_keys = used_pricing_keys(usage);
    if used_keys.is_empty() {
        return None;
    }
    let missing_pricing_keys: Vec<_> = used_keys
        .iter()
        .copied()
        .filter(|key| pricing_entry.get(key).is_none_or(Value::is_null))
        .collect();
    if missing_pricing_keys.is_empty()
        && used_keys.iter().all(|key| {
            pricing_entry
                .get(key)
                .and_then(Value::as_f64)
                .is_some_and(|rate| rate == 0.0)
        })
    {
        return None;
    }
    if !declares_rate(pricing_entry, 4) {
        return None;
    }
    let reason = if calculation_failed {
        ZeroCostReason::CostCalculationError
    } else if missing_pricing_keys.is_empty() {
        ZeroCostReason::PricingNotApplied
    } else {
        ZeroCostReason::MissingPricingKey
    };
    Some(ZeroCostDiagnostic {
        reason,
        pricing_model: pricing_model.to_owned(),
        missing_pricing_keys: if calculation_failed {
            Vec::new()
        } else {
            missing_pricing_keys
        },
    })
}

pub fn zero_cost_warning(
    diagnostic: &ZeroCostDiagnostic,
    model_group: Option<&str>,
    model: &str,
    provider: Option<&str>,
    usage: &ChatUsage,
) -> String {
    let cause = match diagnostic.reason {
        ZeroCostReason::MissingPricingKey => format!(
            "pricing entry '{}' has no {}. Set the missing rate in the deployment's model_info or in the model cost map, or set every rate to 0 to mark the model free",
            diagnostic.pricing_model,
            diagnostic.missing_pricing_keys.join(", ")
        ),
        ZeroCostReason::PricingNotApplied => format!(
            "pricing entry '{}' declares non-zero rates for this usage, but the cost calculator returned $0",
            diagnostic.pricing_model
        ),
        ZeroCostReason::CostCalculationError => format!(
            "cost calculation raised for pricing entry '{}', see response_cost_failure_debug_information",
            diagnostic.pricing_model
        ),
    };
    format!(
        "Billable request priced at $0 and logged as such (model_group={} model={} provider={} prompt_tokens={} completion_tokens={}): {}. Counted in {}{{reason=\"{}\"}}",
        model_group
            .filter(|group| !group.is_empty())
            .unwrap_or(model),
        model,
        provider
            .filter(|provider| !provider.is_empty())
            .unwrap_or("unknown"),
        usage.prompt_tokens,
        usage.completion_tokens,
        cause,
        ZERO_COST_COUNTER_NAME,
        diagnostic.reason.as_str()
    )
}
