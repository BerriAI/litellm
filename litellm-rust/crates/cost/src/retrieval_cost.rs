use serde_json::Value;

use crate::catalog::ModelInfoCatalog;

pub fn rerank_cost_from_model_info(
    provider: &str,
    model_info: Option<&Value>,
    billed_units: Option<&Value>,
) -> (f64, f64) {
    let (Some(model_info), Some(billed_units)) = (model_info, billed_units) else {
        return (0.0, 0.0);
    };
    let (rate_key, unit_key) = match provider {
        "jina_ai" | "voyage" => ("input_cost_per_token", "total_tokens"),
        _ => ("input_cost_per_query", "search_units"),
    };
    let Some(rate) = model_info.get(rate_key).and_then(Value::as_f64) else {
        return (0.0, 0.0);
    };
    let Some(units) = billed_units.get(unit_key).and_then(Value::as_u64) else {
        return (0.0, 0.0);
    };
    (rate * units as f64, 0.0)
}

pub fn vector_store_search_cost_from_model_info(
    provider: &str,
    api_type: Option<&str>,
    search_api_model_info: Option<&Value>,
) -> (f64, f64) {
    if provider != "vertex_ai" || api_type != Some("search_api") {
        return (0.0, 0.0);
    }
    let rate = search_api_model_info
        .and_then(|info| info.get("input_cost_per_query"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    (rate, 0.0)
}

pub fn rerank_cost(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
    region: Option<&str>,
    billed_units: Option<&Value>,
) -> (f64, f64) {
    rerank_cost_from_model_info(
        provider,
        catalog.entry(model, Some(provider), region).as_deref(),
        billed_units,
    )
}

pub fn vector_store_search_cost(
    catalog: &ModelInfoCatalog,
    provider: &str,
    api_type: Option<&str>,
) -> (f64, f64) {
    vector_store_search_cost_from_model_info(
        provider,
        api_type,
        catalog.entries().get("vertex_ai/search_api"),
    )
}
