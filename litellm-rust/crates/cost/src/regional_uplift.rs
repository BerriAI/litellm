use serde_json::Value;

use crate::wire::py_float;

fn multiplier(value: Option<&Value>) -> f64 {
    value
        .filter(|value| !value.is_null())
        .and_then(py_float)
        .unwrap_or(1.0)
}

pub fn get_regional_uplift_multiplier(model_info: &Value, data_residency: Option<&str>) -> f64 {
    let region = match data_residency {
        Some(region) if region.eq_ignore_ascii_case("us") => "us",
        Some(region) if region.eq_ignore_ascii_case("eu") => "eu",
        _ => return 1.0,
    };
    multiplier(model_info.get(format!("regional_processing_uplift_multiplier_{region}")))
}

pub fn get_vertex_regional_endpoint_uplift(
    model_info: &Value,
    vertex_location: Option<&str>,
) -> f64 {
    match vertex_location {
        None => 1.0,
        Some(location) if location.eq_ignore_ascii_case("global") => 1.0,
        Some(_) => multiplier(model_info.get("regional_endpoint_uplift_multiplier")),
    }
}

pub fn get_provider_specific_geo_multiplier(
    model_info: &Value,
    inference_geo: Option<&str>,
) -> f64 {
    let Some(geo) = inference_geo else {
        return 1.0;
    };
    if geo.eq_ignore_ascii_case("global") || geo.eq_ignore_ascii_case("not_available") {
        return 1.0;
    }
    model_info
        .get("provider_specific_entry")
        .and_then(|entry| entry.get(geo.to_ascii_lowercase()))
        .map(|value| multiplier(Some(value)))
        .unwrap_or(1.0)
}

pub fn apply_regional_totals_uplift(
    (prompt, completion): (f64, f64),
    model_info: &Value,
    data_residency: Option<&str>,
    vertex_location: Option<&str>,
) -> (f64, f64) {
    [
        get_regional_uplift_multiplier(model_info, data_residency),
        get_vertex_regional_endpoint_uplift(model_info, vertex_location),
    ]
    .into_iter()
    .fold((prompt, completion), |(prompt, completion), uplift| {
        (prompt * uplift, completion * uplift)
    })
}

pub fn combined_regional_multiplier(
    model_info: &Value,
    inference_geo: Option<&str>,
    data_residency: Option<&str>,
    vertex_location: Option<&str>,
) -> f64 {
    get_regional_uplift_multiplier(model_info, data_residency)
        * get_vertex_regional_endpoint_uplift(model_info, vertex_location)
        * get_provider_specific_geo_multiplier(model_info, inference_geo)
}
