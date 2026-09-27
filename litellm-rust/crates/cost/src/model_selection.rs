use std::str::FromStr;

use serde_json::Value;

use crate::catalog::ModelInfoCatalog;
use crate::provider::LlmProviders;

#[derive(Clone, Copy, Debug)]
pub struct ModelSelectionRequest<'a> {
    pub model: Option<&'a str>,
    pub response: Option<&'a Value>,
    pub hidden_params: Option<&'a Value>,
    pub base_model: Option<&'a str>,
    pub custom_pricing: bool,
    pub provider: Option<&'a str>,
    pub router_model_id: Option<&'a str>,
    pub region_name: Option<&'a str>,
}

pub fn get_response_model(response: Option<&Value>) -> Option<&str> {
    response?.get("model")?.as_str()
}

pub fn get_hidden_str_for_cost_calc<'a>(
    hidden_params: Option<&'a Value>,
    key: &str,
) -> Option<&'a str> {
    hidden_params?
        .get(key)?
        .as_str()
        .filter(|value| !value.is_empty())
}

fn is_llm_provider(name: &str) -> bool {
    LlmProviders::from_str(name).is_ok()
}

pub fn model_contains_known_llm_provider(model: &str) -> bool {
    model.split('/').next().is_some_and(is_llm_provider)
}

pub fn strip_unregistered_leading_segments(
    model: &str,
    region_name: Option<&str>,
    catalog: &ModelInfoCatalog,
) -> String {
    let cost_map = catalog.entries();
    let segments: Vec<&str> = model.split('/').collect();
    if segments.len() < 2 || cost_map.contains_key(&segments[1..].join("/")) {
        return model.to_owned();
    }
    let head_len = if segments.len() > 2
        && region_name.is_some_and(|region| segments.get(1) == Some(&region))
    {
        2
    } else {
        1
    };
    let head = segments[..head_len].join("/");
    let tail = &segments[head_len..];
    let first_provider = tail
        .iter()
        .position(|segment| is_llm_provider(segment))
        .unwrap_or(tail.len());
    (0..=first_provider.min(tail.len().saturating_sub(1)))
        .map(|start| format!("{head}/{}", tail[start..].join("/")))
        .find(|candidate| cost_map.contains_key(candidate))
        .unwrap_or_else(|| model.to_owned())
}

fn has_explicit_pricing(entry: &Value) -> bool {
    [
        "input_cost_per_token",
        "input_cost_per_second",
        "input_cost_per_query",
        "tiered_pricing",
    ]
    .into_iter()
    .any(|key| entry.get(key).is_some_and(|value| !value.is_null()))
}

pub fn get_provider_for_cost_calc(
    model: Option<&str>,
    custom_llm_provider: Option<&str>,
    catalog: &ModelInfoCatalog,
) -> Option<String> {
    custom_llm_provider.map(str::to_owned).or_else(|| {
        catalog
            .get_llm_provider(model?)
            .map(|resolved| resolved.custom_llm_provider)
    })
}

pub fn select_model_name_for_cost_calc(
    request: ModelSelectionRequest<'_>,
    catalog: &ModelInfoCatalog,
) -> Option<String> {
    let cost_map = catalog.entries();
    let provider = get_provider_for_cost_calc(request.model, request.provider, catalog);
    let response_model = get_response_model(request.response);
    let private_model =
        get_hidden_str_for_cost_calc(request.hidden_params, "provider_response_model");
    let explicit_pricing = request.custom_pricing || request.base_model.is_some();
    let priced_region = (!explicit_pricing
        && (private_model.is_some() || response_model.is_some()))
    .then(|| {
        get_hidden_str_for_cost_calc(request.hidden_params, "region_name").or(request.region_name)
    })
    .flatten();
    let explicitly_selected = if request.custom_pricing {
        request
            .router_model_id
            .filter(|id| cost_map.get(*id).is_some_and(has_explicit_pricing))
            .or(request.model)
    } else {
        request.base_model.or(private_model).or_else(|| {
            response_model
                .is_none()
                .then(|| get_hidden_str_for_cost_calc(request.hidden_params, "model"))
                .flatten()
        })
    };
    let selected = explicitly_selected.or(response_model).or(request.model)?;
    let Some(provider) = provider else {
        return Some(selected.to_owned());
    };
    if model_contains_known_llm_provider(selected) {
        return Some(selected.to_owned());
    }
    let prefix =
        priced_region.map_or_else(|| provider.clone(), |region| format!("{provider}/{region}"));
    Some(strip_unregistered_leading_segments(
        &format!("{prefix}/{selected}"),
        priced_region,
        catalog,
    ))
}
