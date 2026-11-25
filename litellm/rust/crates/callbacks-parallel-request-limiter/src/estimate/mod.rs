mod compaction;
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
pub use genai::config_field as genai_config_field;
const CHARS_PER_TOKEN: i128 = 4;
const RESPONSES: [&str; 2] = ["aresponses", "responses"];
const ANTHROPIC_MESSAGES: [&str; 2] = ["anthropic_messages", "aanthropic_messages"];
const EMBEDDING: [&str; 2] = ["aembedding", "embedding"];
const TEXT_COMPLETION: [&str; 2] = ["atext_completion", "text_completion"];
const EXTRACTION: [&str; 2] = ["extraction", "aextraction"];
const REALTIME: [&str; 1] = ["_arealtime"];
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

/// Every top-level request member the estimator reads. A host may leave out any other member whose
/// value it cannot hand over as JSON, since no estimate depends on it.
pub const REQUEST_KEYS: [&str; 21] = [
    "messages",
    "system",
    "instructions",
    "prompt",
    "input",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "n",
    "best_of",
    "state",
    "fields",
    "config",
    "generationConfig",
    "generation_config",
    "contents",
    "systemInstruction",
    "system_instruction",
    "tools",
    "toolConfig",
    "tool_config",
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
        let Some(config) = genai::request_config(request)? else {
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
        genai::request_config(request)?
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

/// The characters of the instruction text a request carries outside its messages or input, which is
/// `system` on Anthropic Messages and `instructions` on Responses.
fn instruction_length(request: &Map<String, Value>, call_type: &str) -> Out<i128> {
    let instructions = if ANTHROPIC_MESSAGES.contains(&call_type) {
        get(request, "system")
    } else if RESPONSES.contains(&call_type) {
        get(request, "instructions")
    } else {
        return Ok(0);
    };
    match py(instructions)? {
        Py::Str(found) => Ok(length(found)),
        Py::List(blocks) => blocks
            .iter()
            .map(|block| match block {
                Value::Object(block) => Ok(text(get(block, "text"))?.map_or(0, length)),
                _ => Ok(0),
            })
            .sum(),
        _ => Ok(0),
    }
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

fn reads_input_as_media(request: &Map<String, Value>) -> bool {
    ["inputType", "input_type"]
        .into_iter()
        .find_map(|key| get(request, key).as_str().filter(|kind| !kind.is_empty()))
        .is_some_and(|kind| ["image", "video", "audio"].contains(&kind))
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
    let billable;
    let raw_input = if responses {
        billable = compaction::billable_input(raw_input)?;
        &billable
    } else {
        raw_input
    };
    if estimate.include_media {
        let media_source = if messages.is_null() {
            raw_input
        } else {
            messages
        };
        if (messages.is_array() || responses) && may_hold_media(media_source) {
            return Err(PythonServed);
        }
        if embedding_call && (may_hold_embedding_media(raw_input) || reads_input_as_media(request))
        {
            return Err(PythonServed);
        }
    }
    let total_chars = instruction_length(request, call_type)?
        + match (messages, extraction_length) {
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
    let generates_no_output = is_embedding_request(request, call_type)
        || EXTRACTION.contains(&call_type)
        || REALTIME.contains(&call_type);
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn estimated(
        data: Value,
        call_type: &str,
        min_configured_tpm_limit: Option<i64>,
        configured_output_tokens: Option<i128>,
        include_media: bool,
    ) -> Out<(i128, i128)> {
        input_and_output_tokens(
            EstimateRequest {
                data: &data,
                call_type,
                min_configured_tpm_limit,
                configured_output_tokens,
                include_media,
            },
            &CapSettings::default(),
        )
    }

    fn compaction(summary: &str) -> Value {
        let block = json!({"type": "compaction", "content": summary}).to_string();
        json!({"type": "compaction", "encrypted_content": block})
    }

    // Every expected value below was derived by reading `estimate_input_and_output_tokens` in
    // `litellm/proxy/hooks/parallel_request_limiter_v3.py` (chars // 4 with a floor of one, the
    // no-max_tokens floor int(4096 * 0.25) = 1024 capped at int(min_tpm * 0.25), the Responses
    // floor of 16, times the candidate count), not by running the Python estimator.
    #[rstest]
    // "hello world!" is 12 characters: 12 // 4 = 3 input; no cap, so max(3, 1024) output.
    #[case::chat(json!({"messages": [{"role": "user", "content": "hello world!"}]}), "acompletion", None, None, (3, 1024))]
    // An explicit max_tokens of 50 across n=2 candidates.
    #[case::chat_explicit_cap_and_candidates(
        json!({"messages": [{"role": "user", "content": "hello world!"}], "max_tokens": 50, "n": 2}),
        "acompletion", None, None, (3, 100)
    )]
    // The operator-declared estimate replaces the heuristic floor.
    #[case::chat_configured(json!({"messages": [{"role": "user", "content": "hello world!"}]}), "acompletion", None, Some(77), (3, 77))]
    // Text parts count, search results add source and title text: 4 + 4 + 6 + 5 = 19 // 4 = 4.
    #[case::chat_parts_and_search_results(
        json!({"messages": [{"role": "tool", "content": [{"type": "text", "text": "abcd"}, {"type": "text", "text": "efgh"}],
               "search_results": [{"source": "source", "title": "title"}]}]}),
        "acompletion", None, None, (4, 1024)
    )]
    // No messages, no prompt: nothing to reserve.
    #[case::chat_empty(json!({"messages": []}), "acompletion", None, None, (0, 0))]
    // 12 characters of messages plus 40 of system: 52 // 4 = 13 input.
    #[case::messages_system(
        json!({"system": "x".repeat(40), "messages": [{"role": "user", "content": "hello world!"}], "max_tokens": 50}),
        "anthropic_messages", None, None, (13, 50)
    )]
    // Only the text of each system block counts: (12 + 40 + 8) // 4 = 15 input.
    #[case::messages_system_blocks(
        json!({"system": [{"type": "text", "text": "x".repeat(40), "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "abcdefgh"}],
               "messages": [{"role": "user", "content": "hello world!"}], "max_tokens": 50}),
        "anthropic_messages", None, None, (15, 50)
    )]
    // A chat request has no top-level system member, so a stray one is not counted.
    #[case::chat_ignores_system(
        json!({"system": "x".repeat(40), "messages": [{"role": "user", "content": "hello world!"}], "max_tokens": 50}),
        "acompletion", None, None, (3, 50)
    )]
    // 40 characters of input plus 80 of instructions: 120 // 4 = 30 input.
    #[case::responses_instructions(
        json!({"instructions": "y".repeat(80), "input": "x".repeat(40), "max_output_tokens": 50}),
        "aresponses", None, None, (30, 50)
    )]
    // 8 characters of prompt: 2 input; the floor capped at int(1000 * 0.25) = 250.
    #[case::text_completion(json!({"prompt": "abcdefgh"}), "atext_completion", Some(1000), None, (2, 250))]
    // A prompt list sums str() of each item.
    #[case::text_completion_list(json!({"prompt": ["abcd", "efgh"]}), "text_completion", None, None, (2, 1024))]
    // 40 characters: 10 input, the Responses floor max(1024, 16).
    #[case::responses(json!({"input": "x".repeat(40)}), "aresponses", None, None, (10, 1024))]
    // An explicit max_output_tokens below 16 is raised to the Responses minimum.
    #[case::responses_small_cap(json!({"input": "x".repeat(40), "max_output_tokens": 5}), "responses", None, None, (10, 16))]
    // From the newest compaction: "summ" (4) plus repr of {'role': 'user', 'content': 'hi'} (33) = 37 // 4 = 9.
    #[case::responses_compaction(
        json!({"input": [{"role": "user", "content": "an earlier question"}, compaction("summ"), {"role": "user", "content": "hi"}]}),
        "aresponses", None, None, (9, 1024)
    )]
    // Embeddings generate no output.
    #[case::embedding(json!({"input": "abcdefgh"}), "aembedding", None, None, (2, 0))]
    #[case::embedding_text_input_type(json!({"input": "abcdefgh", "input_type": "text"}), "aembedding", None, None, (2, 0))]
    #[case::genai_snake_case_config(
        json!({"contents": [{"role": "user", "parts": [{"text": "hello world!"}]}],
               "generation_config": {"max_output_tokens": 40, "candidate_count": 2}}),
        "agenerate_content", None, None, (3, 80)
    )]
    #[case::genai_empty_config_yields(
        json!({"contents": [{"role": "user", "parts": [{"text": "hello world!"}]}],
               "config": {}, "generationConfig": {"maxOutputTokens": 30}}),
        "generate_content", None, None, (3, 30)
    )]
    fn the_estimate_matches_the_python_estimator(
        #[case] data: Value,
        #[case] call_type: &str,
        #[case] min_configured_tpm_limit: Option<i64>,
        #[case] configured_output_tokens: Option<i128>,
        #[case] expected: (i128, i128),
    ) {
        assert_eq!(
            estimated(
                data,
                call_type,
                min_configured_tpm_limit,
                configured_output_tokens,
                true
            ),
            Ok(expected)
        );
    }

    #[rstest]
    #[case::chat_image(
        json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]}),
        "acompletion"
    )]
    #[case::responses_audio(
        json!({"input": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA"}}]}]}),
        "aresponses"
    )]
    #[case::embedding_data_uri(json!({"input": "data:image/png;base64,AAAA"}), "aembedding")]
    #[case::embedding_image_input_type(json!({"input": "AAAA", "input_type": "image"}), "aembedding")]
    #[case::embedding_video_input_type(json!({"input": ["s3://b/c.mp4"], "inputType": "video"}), "aembedding")]
    fn media_is_left_to_python(#[case] data: Value, #[case] call_type: &str) {
        assert_eq!(
            estimated(data, call_type, None, None, true),
            Err(PythonServed)
        );
    }

    // Without media the image part carries no text: no input, and so no output either.
    #[rstest]
    fn the_output_only_estimate_reads_past_media() {
        let data = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]});
        assert_eq!(
            estimated(data, "acompletion", None, None, false),
            Ok((0, 0))
        );
    }

    #[rstest]
    fn a_value_python_alone_reads_is_left_to_python() {
        let opaque = Value::String(litellm_host::raw_json::encode("\"<object>\""));
        let data = json!({"messages": [{"role": "user", "content": opaque}]});
        assert_eq!(
            estimated(data, "acompletion", None, None, true),
            Err(PythonServed)
        );
    }

    #[rstest]
    fn the_total_is_the_input_plus_the_output() {
        let data =
            json!({"messages": [{"role": "user", "content": "hello world!"}], "max_tokens": 50});
        let request = EstimateRequest {
            data: &data,
            call_type: "acompletion",
            min_configured_tpm_limit: None,
            configured_output_tokens: None,
            include_media: true,
        };
        assert_eq!(tokens_for_request(request, &CapSettings::default()), Ok(53));
    }
}
