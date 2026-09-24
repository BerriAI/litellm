use crate::catalog::{ModelCostRequest, ModelInfoCatalog};
use crate::error::CostError;
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::per_second::per_second_pricing_cost;

pub fn registry_key(model: &str) -> &str {
    let name = model.strip_prefix("databricks/").unwrap_or(model);
    [
        ("dbrx-instruct", "databricks-dbrx-instruct"),
        (
            "meta-llama-3.1-70b-instruct",
            "databricks-meta-llama-3-1-70b-instruct",
        ),
        (
            "meta-llama-3.1-405b-instruct",
            "databricks-meta-llama-3-1-405b-instruct",
        ),
        (
            "mixtral-8x7b-instruct-v0.1",
            "databricks-mixtral-8x7b-instruct",
        ),
        ("bge-large-en", "databricks-bge-large-en"),
        ("gte-large-en", "databricks-gte-large-en"),
        ("llama-2-70b-chat", "databricks-llama-2-70b-chat"),
    ]
    .into_iter()
    .find_map(|(prefix, key)| name.starts_with(prefix).then_some(key))
    .unwrap_or(name)
}

pub fn databricks_cost_per_token(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
) -> Result<(f64, f64), CostError> {
    if let Some(cost) = catalog
        .entry(request.model, Some("databricks"), request.region)
        .and_then(|info| per_second_pricing_cost(&info, request.response_time_ms))
    {
        return Ok(cost);
    }
    let entry = catalog
        .entry(
            registry_key(request.model),
            Some("databricks"),
            request.region,
        )
        .ok_or(CostError::ModelNotFound)?;
    Ok(calculate_generic_cost_from_model_info_with_region(
        request.usage,
        &entry,
        None,
        false,
        None,
        None,
        request.at,
    ))
}
