use serde_json::Value;

use crate::responses_usage::ChatUsage;

const DEFAULT_WEB_SEARCH_COST: f64 = 35e-3;
const DEFAULT_MAPS_QUERY_COST: f64 = 14e-3;
const DEFAULT_MAPS_PROMPT_COST: f64 = 25e-3;

fn per_query(model_info: &Value) -> bool {
    model_info
        .get("web_search_billing_unit")
        .and_then(Value::as_str)
        == Some("per_query")
}

fn billed_requests(requests: u64, per_query: bool) -> u64 {
    if per_query {
        requests
    } else {
        u64::from(requests > 0)
    }
}

fn web_search_requests(usage: &ChatUsage) -> u64 {
    let from_details = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.web_search_requests)
        .unwrap_or(0);
    if from_details > 0 {
        return from_details;
    }
    usage
        .extra
        .get("server_tool_use")
        .and_then(|value| value.get("web_search_requests"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

pub fn cost_per_web_search_request(usage: &ChatUsage, model_info: &Value) -> f64 {
    let rate = model_info
        .get("search_context_cost_per_query")
        .and_then(|rates| rates.get("search_context_size_medium"))
        .and_then(Value::as_f64)
        .unwrap_or(DEFAULT_WEB_SEARCH_COST);
    rate * billed_requests(web_search_requests(usage), per_query(model_info)) as f64
}

pub fn google_maps_grounding_requests(usage: Option<&ChatUsage>) -> Option<u64> {
    usage?
        .prompt_tokens_details
        .as_ref()?
        .google_maps_grounding_requests
}

pub fn cost_per_google_maps_grounding_request(usage: &ChatUsage, model_info: &Value) -> f64 {
    let Some(requests) = google_maps_grounding_requests(Some(usage)).filter(|count| *count > 0)
    else {
        return 0.0;
    };
    let per_query = per_query(model_info);
    let default_rate = if per_query {
        DEFAULT_MAPS_QUERY_COST
    } else {
        DEFAULT_MAPS_PROMPT_COST
    };
    let rate = model_info
        .get("google_maps_grounding_cost_per_query")
        .and_then(Value::as_f64)
        .unwrap_or(default_rate);
    rate * billed_requests(requests, per_query) as f64
}
