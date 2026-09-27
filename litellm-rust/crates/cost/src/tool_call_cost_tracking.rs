use serde_json::Value;

use crate::anthropic_cost::{
    get_anthropic_web_search_requests_from_response, get_web_search_requests,
};
use crate::provider::LlmProviders;
use crate::wire::{lax_int, py_float, py_int};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DefaultToolRates {
    pub file_search_per_call: f64,
    pub azure_file_search_per_gb_day: f64,
    pub azure_vector_store_per_gb_day: f64,
    pub azure_computer_input_per_1k_tokens: f64,
    pub azure_computer_output_per_1k_tokens: f64,
    pub code_interpreter_per_session: Option<f64>,
    pub xai_web_search_per_call: f64,
    pub groq_browser_open_per_call: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseKind {
    Chat,
    Responses,
    Anthropic,
    Other,
}

fn number(value: Option<&Value>) -> Option<f64> {
    value.and_then(py_float)
}

pub fn safe_convert_to_int(value: Option<&Value>) -> Option<i64> {
    value.and_then(py_int)
}

pub fn extract_file_search_params(file_search: &Value) -> (Option<f64>, Option<f64>) {
    (
        number(file_search.get("storage_gb")),
        number(file_search.get("days")),
    )
}

pub fn extract_token_counts(computer_use: &Value) -> (Option<i64>, Option<i64>) {
    (
        safe_convert_to_int(computer_use.get("input_tokens")),
        safe_convert_to_int(computer_use.get("output_tokens")),
    )
}

pub fn response_includes_annotation_type(response: &Value, annotation_type: &str) -> bool {
    response
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("message")
                    .and_then(|message| message.get("annotations"))
                    .and_then(Value::as_array)
                    .is_some_and(|annotations| {
                        annotations.iter().any(|annotation| {
                            annotation.get("type").and_then(Value::as_str) == Some(annotation_type)
                        })
                    })
            })
        })
}

pub fn chat_completion_response_includes_annotations(response: &Value) -> bool {
    response
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("message")
                    .and_then(|message| message.get("annotations"))
                    .and_then(Value::as_array)
                    .is_some_and(|annotations| !annotations.is_empty())
            })
        })
}

pub fn response_includes_output_type(response: &Value, output_type: &str) -> bool {
    response
        .get("output")
        .and_then(Value::as_array)
        .is_some_and(|output| {
            output
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some(output_type))
        })
}

pub fn response_object_includes_file_search_call(response: &Value, kind: ResponseKind) -> bool {
    match kind {
        ResponseKind::Chat => response_includes_annotation_type(response, "file_citation"),
        ResponseKind::Responses => response_includes_output_type(response, "file_search_call"),
        _ => false,
    }
}

pub fn response_object_includes_web_search_call(
    response: &Value,
    kind: ResponseKind,
    usage: Option<&Value>,
) -> bool {
    if kind == ResponseKind::Anthropic
        && get_anthropic_web_search_requests_from_response(response).is_some()
    {
        return true;
    }
    match kind {
        ResponseKind::Responses => response_includes_output_type(response, "web_search_call"),
        ResponseKind::Chat => {
            response_includes_annotation_type(response, "url_citation")
                || usage_reports_web_search(usage)
        }
        _ => usage_reports_web_search(usage),
    }
}

pub fn usage_reports_server_side_web_search_calls(details: Option<&Value>) -> bool {
    match details.and_then(|details| details.get("web_search_calls")) {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(calls)) => calls.as_i64().is_some_and(|calls| calls > 0),
        _ => false,
    }
}

fn usage_reports_web_search(usage: Option<&Value>) -> bool {
    let Some(usage) = usage else {
        return false;
    };
    get_web_search_requests(usage.get("server_tool_use")).is_some()
        || usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("web_search_requests"))
            .is_some_and(|requests| !requests.is_null())
        || usage_reports_server_side_web_search_calls(usage.get("server_side_tool_usage_details"))
}

pub fn count_web_search_calls(response: &Value, kind: ResponseKind) -> u64 {
    if kind != ResponseKind::Responses {
        return 1;
    }
    if let Some(reported) = response
        .get("tool_usage")
        .and_then(|usage| usage.get("web_search"))
        .and_then(|search| search.get("num_requests"))
        .and_then(lax_int)
        .and_then(|requests| u64::try_from(requests).ok())
    {
        return reported;
    }
    response
        .get("output")
        .and_then(Value::as_array)
        .map_or(1, |output| {
            output
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("web_search_call"))
                .count()
                .max(1) as u64
        })
}

pub fn get_default_cost_for_web_search(model_info: Option<&Value>) -> f64 {
    model_info
        .and_then(|info| info.get("search_context_cost_per_query"))
        .and_then(|pricing| number(pricing.get("search_context_size_medium")))
        .unwrap_or(0.0)
}

pub fn get_cost_for_web_search(options: Option<&Value>, model_info: Option<&Value>) -> f64 {
    let Some(model_info) = model_info else {
        return 0.0;
    };
    let size = options
        .and_then(|options| options.get("search_context_size"))
        .and_then(Value::as_str);
    let key = match size {
        Some("low") => "search_context_size_low",
        Some("medium") => "search_context_size_medium",
        Some("high") => "search_context_size_high",
        _ => return get_default_cost_for_web_search(Some(model_info)),
    };
    model_info
        .get("search_context_cost_per_query")
        .and_then(|pricing| number(pricing.get(key)))
        .unwrap_or(0.0)
}

pub fn get_cost_for_file_search(
    file_search: Option<&Value>,
    provider: Option<&str>,
    model_info: Option<&Value>,
    storage_gb: Option<f64>,
    days: Option<f64>,
    defaults: DefaultToolRates,
) -> f64 {
    if file_search.is_none() {
        return 0.0;
    }
    let storage = storage_gb.unwrap_or(0.0) * days.unwrap_or(0.0);
    if LlmProviders::AZURE.matches(provider)
        && let Some(rate) =
            model_info.and_then(|info| number(info.get("file_search_cost_per_gb_per_day")))
    {
        return storage * rate;
    }
    if let Some(rate) =
        model_info.and_then(|info| number(info.get("file_search_cost_per_1k_calls")))
    {
        return rate;
    }
    if LlmProviders::AZURE.matches(provider) {
        return storage * defaults.azure_file_search_per_gb_day;
    }
    defaults.file_search_per_call
}

pub fn get_cost_for_vector_store(
    usage: Option<&Value>,
    provider: Option<&str>,
    model_info: Option<&Value>,
    defaults: DefaultToolRates,
) -> f64 {
    let Some(usage) = usage else {
        return 0.0;
    };
    let storage =
        number(usage.get("storage_gb")).unwrap_or(0.0) * number(usage.get("days")).unwrap_or(0.0);
    let rate = model_info.and_then(|info| number(info.get("vector_store_cost_per_gb_per_day")));
    match rate {
        Some(rate) => storage * rate,
        None if LlmProviders::AZURE.matches(provider) => {
            storage * defaults.azure_vector_store_per_gb_day
        }
        None => 0.0,
    }
}

pub fn get_cost_for_computer_use(
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    provider: Option<&str>,
    model_info: Option<&Value>,
    defaults: DefaultToolRates,
) -> f64 {
    if !LlmProviders::AZURE.matches(provider) {
        return 0.0;
    }
    let input_rate = model_info
        .and_then(|info| number(info.get("computer_use_input_cost_per_1k_tokens")))
        .unwrap_or(0.0);
    let output_rate = model_info
        .and_then(|info| number(info.get("computer_use_output_cost_per_1k_tokens")))
        .unwrap_or(0.0);
    let (input_rate, output_rate) = if input_rate != 0.0 || output_rate != 0.0 {
        (input_rate, output_rate)
    } else {
        (
            defaults.azure_computer_input_per_1k_tokens,
            defaults.azure_computer_output_per_1k_tokens,
        )
    };
    input_tokens.unwrap_or(0) as f64 / 1000.0 * input_rate
        + output_tokens.unwrap_or(0) as f64 / 1000.0 * output_rate
}

pub fn get_cost_for_code_interpreter(
    sessions: Option<i64>,
    model_info: Option<&Value>,
    defaults: DefaultToolRates,
) -> f64 {
    let Some(sessions) = sessions else {
        return 0.0;
    };
    let rate = model_info
        .and_then(|info| number(info.get("code_interpreter_cost_per_session")))
        .or(defaults.code_interpreter_per_session)
        .unwrap_or(0.0);
    sessions as f64 * rate
}
