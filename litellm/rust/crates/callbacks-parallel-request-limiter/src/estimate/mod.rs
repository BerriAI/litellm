mod genai;
mod printable;
mod python;

use serde_json::{Map, Value};

use self::python::{
    Py, big_int_as_float, json_length, length, py, python_float_text, python_int_text, str_length,
    text, truncated, truthy,
};
use crate::{arithmetic::no_max_tokens_output_floor, limiter::CapSettings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PythonServed;

pub type Out<T> = Result<T, PythonServed>;

pub const GENAI_CALL_TYPES: [&str; 4] = genai::CALL_TYPES;
const CHARS_PER_TOKEN: i128 = 4;
const RESPONSES: [&str; 2] = ["aresponses", "responses"];
const EMBEDDING: [&str; 2] = ["aembedding", "embedding"];
const TEXT_COMPLETION: [&str; 2] = ["atext_completion", "text_completion"];
const EXTRACTION: [&str; 2] = ["extraction", "aextraction"];
const MEDIA_TYPES: [&str; 10] = [
    "file",
    "input_file",
    "input_audio",
    "image_url",
    "input_image",
    "image",
    "computer_screenshot",
    "image_generation_call",
    "code_interpreter_call",
    "computer_call_output",
];

#[derive(Clone, Copy, Debug)]
pub struct EstimateRequest<'a> {
    pub data: &'a Value,
    pub call_type: &'a str,
    pub min_configured_tpm_limit: Option<i64>,
    pub configured_output_tokens: Option<i128>,
    pub include_media: bool,
}

fn get<'a>(request: &'a Map<String, Value>, key: &str) -> &'a Value {
    request.get(key).unwrap_or(&Value::Null)
}

fn is_embedding_request(request: &Map<String, Value>, call_type: &str) -> bool {
    if EMBEDDING.contains(&call_type) {
        return true;
    }
    if RESPONSES.contains(&call_type) || !call_type.is_empty() {
        return false;
    }
    !get(request, "input").is_null()
}

fn output_cap_value(raw: &Value) -> Out<Option<i128>> {
    match py(raw)? {
        Py::Int(int) => truncated(int as f64),
        Py::BigInt(digits) => match big_int_as_float(digits) {
            Some(float) => truncated(float),
            None => Ok(None),
        },
        Py::Float(float) => truncated(float),
        Py::Str(text) => match python_float_text(text)? {
            Some(float) => truncated(float),
            None => Ok(None),
        },
        Py::None | Py::Bool(_) | Py::List(_) | Py::Dict(_) => Ok(None),
    }
}

pub fn explicit_output_cap(
    request: &Map<String, Value>,
    call_type: &str,
    cap: &CapSettings,
) -> Out<Option<i128>> {
    if genai::CALL_TYPES.contains(&call_type) {
        let Some(config) = genai::request_config(request) else {
            return Ok(None);
        };
        let mut found: Option<i128> = None;
        for field in ["maxOutputTokens", "max_output_tokens"] {
            if let Some(parsed) = output_cap_value(get(config, field))? {
                found = Some(found.map_or(parsed, |current| current.max(parsed)));
            }
        }
        return Ok(found);
    }
    if RESPONSES.contains(&call_type) {
        return Ok(output_cap_value(get(request, "max_output_tokens"))?
            .map(|parsed| parsed.max(i128::from(cap.responses_min_output_tokens))));
    }
    if EMBEDDING.contains(&call_type) {
        return Ok(None);
    }
    let fields: &[&str] = if call_type.is_empty() {
        &["max_tokens", "max_completion_tokens", "max_output_tokens"]
    } else {
        &["max_tokens", "max_completion_tokens"]
    };
    let mut found: Option<i128> = None;
    for field in fields {
        if let Some(parsed) = output_cap_value(get(request, field))? {
            found = Some(found.map_or(parsed, |current| current.max(parsed)));
        }
    }
    Ok(found)
}

fn candidate_value(value: &Value) -> Out<Option<i128>> {
    if !truthy(value)? {
        return Ok(Some(1));
    }
    match py(value)? {
        Py::Bool(_) => Ok(Some(1)),
        Py::Int(int) => Ok(Some(int)),
        Py::BigInt(_) => Err(PythonServed),
        Py::Float(float) => truncated(float),
        Py::Str(text) => python_int_text(text),
        Py::None | Py::List(_) | Py::Dict(_) => Ok(None),
    }
}

pub fn output_candidate_count(request: &Map<String, Value>, call_type: &str) -> Out<i128> {
    let config = if genai::CALL_TYPES.contains(&call_type) {
        genai::request_config(request)
    } else {
        None
    };
    let values = [
        Some(get(request, "n")),
        Some(get(request, "best_of")),
        config.map(|config| get(config, "candidateCount")),
        config.map(|config| get(config, "candidate_count")),
    ];
    let mut count: i128 = 1;
    for value in values
        .into_iter()
        .map(|value| value.unwrap_or(&Value::Null))
    {
        if let Some(candidates) = candidate_value(value)? {
            count = count.max(candidates);
        }
    }
    Ok(count)
}

fn search_results_length(results: &Value) -> Out<i128> {
    let Value::Array(results) = results else {
        return Ok(0);
    };
    let mut total = 0;
    for result in results {
        let Value::Object(result) = result else {
            continue;
        };
        for key in ["source", "title"] {
            if let Some(found) = text(get(result, key))? {
                total += length(found);
            }
        }
        if let Value::Array(blocks) = get(result, "content") {
            for block in blocks {
                if let Value::Object(block) = block
                    && let Some(found) = text(get(block, "text"))?
                {
                    total += length(found);
                }
            }
        }
        let citations = get(result, "citations");
        if !citations.is_null() {
            total += json_length(citations, true)?;
        }
    }
    Ok(total)
}

fn messages_length(messages: &[Value]) -> Out<i128> {
    let mut total = 0;
    for message in messages {
        let Value::Object(message) = message else {
            return Err(PythonServed);
        };
        let content = get(message, "content");
        if truthy(content)? {
            match py(content)? {
                Py::List(parts) => {
                    for part in parts {
                        let Value::Object(part) = part else {
                            return Err(PythonServed);
                        };
                        let found = get(part, "text");
                        if truthy(found)? {
                            total += length(text(found)?.ok_or(PythonServed)?);
                        }
                    }
                }
                Py::Str(found) => total += length(found),
                _ => {}
            }
        }
        total += search_results_length(get(message, "search_results"))?;
    }
    Ok(total)
}

fn may_hold_media(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(may_hold_media),
        Value::Object(fields) => {
            fields
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| MEDIA_TYPES.contains(&kind))
                || fields.values().any(may_hold_media)
        }
        _ => false,
    }
}

fn may_hold_embedding_media(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(may_hold_embedding_media),
        Value::Object(_) => true,
        Value::String(text) => text.starts_with("data:"),
        _ => false,
    }
}

fn holds_compaction(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(|item| {
            item.as_object().is_some_and(|fields| {
                fields.get("type").and_then(Value::as_str) == Some("compaction")
            })
        }),
        _ => false,
    }
}

fn sequence_length(value: &Value) -> Out<i128> {
    match py(value)? {
        Py::Str(found) => Ok(length(found)),
        Py::List(items) => items.iter().map(str_length).sum(),
        _ => Ok(0),
    }
}

fn extraction_text_length(request: &Map<String, Value>) -> Out<i128> {
    let state = text(get(request, "state"))?.map_or(0, length);
    Ok(state + 1 + json_length(get(request, "fields"), false)?)
}

pub fn input_and_output_tokens(
    estimate: EstimateRequest<'_>,
    cap: &CapSettings,
) -> Out<(i128, i128)> {
    let Value::Object(request) = estimate.data else {
        return Ok((0, 0));
    };
    let call_type = estimate.call_type;
    let responses = RESPONSES.contains(&call_type);
    let embedding_call = EMBEDDING.contains(&call_type);
    let translated = genai::translated_messages(request, call_type)?;
    let messages_source = translated.as_ref().unwrap_or(get(request, "messages"));
    let estimable = |key: &str| -> &Value {
        if translated.is_some() {
            &Value::Null
        } else {
            get(request, key)
        }
    };
    let extraction_length = if EXTRACTION.contains(&call_type) {
        Some(extraction_text_length(request)?)
    } else {
        None
    };
    let (messages, prompt, raw_input) = if responses || embedding_call {
        (&Value::Null, &Value::Null, estimable("input"))
    } else if TEXT_COMPLETION.contains(&call_type) {
        (&Value::Null, estimable("prompt"), &Value::Null)
    } else if extraction_length.is_some() {
        (&Value::Null, &Value::Null, &Value::Null)
    } else if !call_type.is_empty() {
        (messages_source, &Value::Null, &Value::Null)
    } else {
        (messages_source, estimable("prompt"), estimable("input"))
    };
    if responses && holds_compaction(raw_input) {
        return Err(PythonServed);
    }
    if estimate.include_media {
        let media_source = if messages.is_null() {
            raw_input
        } else {
            messages
        };
        if (messages.is_array() || responses) && may_hold_media(media_source) {
            return Err(PythonServed);
        }
        if embedding_call && may_hold_embedding_media(raw_input) {
            return Err(PythonServed);
        }
    }
    let total_chars = match (messages, extraction_length) {
        (_, Some(extraction)) => extraction,
        (Value::Array(items), None) if !items.is_empty() => messages_length(items)?,
        _ => match py(prompt)? {
            Py::Str(_) | Py::List(_) => sequence_length(prompt)?,
            _ => sequence_length(raw_input)?,
        },
    };
    let estimated_input = if total_chars > 0 {
        (total_chars / CHARS_PER_TOKEN).max(1)
    } else {
        0
    };
    let explicit = explicit_output_cap(request, call_type, cap)?;
    let generates_no_output =
        is_embedding_request(request, call_type) || EXTRACTION.contains(&call_type);
    let floor = i128::from(no_max_tokens_output_floor(
        estimate.min_configured_tpm_limit,
        cap.default_max_tokens_estimate,
        cap.fraction,
    ));
    let output_floor = if responses {
        floor.max(i128::from(cap.responses_min_output_tokens))
    } else {
        floor
    };
    let max_tokens = if generates_no_output
        || (explicit.is_none()
            && estimated_input == 0
            && estimate.configured_output_tokens.is_none())
    {
        0
    } else if let Some(explicit) = explicit {
        explicit
    } else if let Some(configured) = estimate.configured_output_tokens {
        configured
    } else {
        estimated_input.max(output_floor)
    };
    let candidates = output_candidate_count(request, call_type)?;
    let output = max_tokens.checked_mul(candidates).ok_or(PythonServed)?;
    Ok((estimated_input, output))
}

pub fn tokens_for_request(estimate: EstimateRequest<'_>, cap: &CapSettings) -> Out<i128> {
    let (input, output) = input_and_output_tokens(estimate, cap)?;
    input.checked_add(output).ok_or(PythonServed)
}
