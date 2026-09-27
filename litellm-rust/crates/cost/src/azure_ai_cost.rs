use serde_json::Value;

use crate::catalog::{ModelCostRequest, ModelInfoCatalog};
use crate::error::CostError;
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::per_second::per_second_pricing_cost;

pub fn is_azure_model_router(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("model-router") || lower.contains("model_router")
}

pub fn is_router_fee_entry(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let bare = lower.strip_prefix("azure_ai/").unwrap_or(&lower);
    matches!(bare, "model-router" | "model_router")
}

pub fn router_fee_entry_name(model: &str) -> &'static str {
    let lower = model.to_ascii_lowercase();
    let bare = lower.strip_prefix("azure_ai/").unwrap_or(&lower);
    if bare == "model-router" {
        "model-router"
    } else {
        "model_router"
    }
}

pub fn router_fee_name<'a>(model: &'a str, request_model: Option<&'a str>) -> Option<&'a str> {
    if is_router_fee_entry(model) {
        return None;
    }
    if is_azure_model_router(model) {
        return Some(model);
    }
    request_model.filter(|name| is_azure_model_router(name))
}

pub fn calculate_azure_model_router_flat_cost(
    model: &str,
    prompt_tokens: u64,
    router_model_info: &Value,
) -> f64 {
    if !is_azure_model_router(model) {
        return 0.0;
    }
    router_model_info
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .filter(|rate| *rate > 0.0)
        .map_or(0.0, |rate| prompt_tokens as f64 * rate)
}

pub fn azure_ai_cost_per_token(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    request_model: Option<&str>,
) -> Result<(f64, f64), CostError> {
    let model_info = catalog.entry(request.model, Some("azure_ai"), request.region);
    let model_info = model_info.as_deref();
    if let Some(cost) =
        model_info.and_then(|info| per_second_pricing_cost(info, request.response_time_ms))
    {
        return Ok(cost);
    }
    let (prompt, completion) = match model_info {
        Some(info) => calculate_generic_cost_from_model_info_with_region(
            request.usage,
            info,
            request.service_tier,
            false,
            request.data_residency,
            request.vertex_location,
            request.at,
        ),
        None if is_azure_model_router(request.model) => (0.0, 0.0),
        None => return Err(CostError::ModelNotFound),
    };
    let fee = azure_ai_router_fee(
        catalog,
        request.model,
        request_model,
        request.usage.prompt_tokens,
    )?
    .unwrap_or(0.0);
    Ok((prompt + fee, completion))
}

pub fn azure_ai_router_fee(
    catalog: &ModelInfoCatalog,
    model: &str,
    request_model: Option<&str>,
    prompt_tokens: u64,
) -> Result<Option<f64>, CostError> {
    let Some(fee_name) = router_fee_name(model, request_model) else {
        return Ok(None);
    };
    let fee_entry = router_fee_entry_name(fee_name);
    let entry = catalog
        .entry(fee_entry, Some("azure_ai"), None)
        .ok_or(CostError::ModelNotFound)?;
    let fee = calculate_azure_model_router_flat_cost(fee_name, prompt_tokens, &entry);
    Ok((fee > 0.0).then_some(fee))
}
