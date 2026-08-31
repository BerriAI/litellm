use super::encoding::ResolvedTokenizer;
use super::formatting::format_function_definitions;
use crate::CoreError;
use serde_json::{Map, Value};

pub(crate) fn count_messages(
    tokenizer: &ResolvedTokenizer,
    messages: &[Map<String, Value>],
    tools: Option<&[Map<String, Value>]>,
    tool_choice: Option<&Value>,
    count_response_tokens: bool,
    model: &str,
    default_token_count: Option<usize>,
) -> Result<usize, CoreError> {
    let normalized = super::encoding::normalized_model_name(model);
    let (tokens_per_message, tokens_per_name) = message_overhead(normalized.as_ref());

    let mut num_tokens = 0;

    for message in messages {
        num_tokens += tokens_per_message;

        for (key, value) in message {
            if value.is_null() {
                continue;
            }

            let result = count_message_field(tokenizer, key, value, tokens_per_name);
            match result {
                Ok(n) => num_tokens += n,
                Err(e) => return default_token_count.ok_or(e),
            }
        }
    }

    if !count_response_tokens {
        let has_system = messages
            .iter()
            .any(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"));
        num_tokens += count_extra(tokenizer, tools, tool_choice, has_system);
    }

    Ok(num_tokens)
}

fn count_message_field(
    tokenizer: &ResolvedTokenizer,
    key: &str,
    value: &Value,
    tokens_per_name: isize,
) -> Result<usize, CoreError> {
    if key == "tool_calls" || key == "function_call" {
        return count_function_call_tokens(tokenizer, key, value);
    }

    if let Some(s) = value.as_str() {
        let mut n = tokenizer.count(s);
        if key == "name" {
            n = n.wrapping_add_signed(tokens_per_name);
        }
        return Ok(n);
    }

    if key == "content" && value.is_array() {
        return count_content_list(tokenizer, value.as_array().unwrap());
    }

    if key == "search_results" && value.is_array() {
        return Ok(count_search_results(tokenizer, value.as_array().unwrap()));
    }

    Ok(0)
}

fn count_search_results(tokenizer: &ResolvedTokenizer, results: &[Value]) -> usize {
    let mut text = String::new();
    for result in results {
        let Some(obj) = result.as_object() else {
            continue;
        };
        if let Some(s) = obj.get("text").and_then(|v| v.as_str()) {
            text.push_str(s);
            text.push('\n');
        }
    }
    if text.is_empty() {
        0
    } else {
        tokenizer.count(&text)
    }
}

fn message_overhead(model: &str) -> (usize, isize) {
    if model == "gpt-3.5-turbo-0301" {
        (4, -1)
    } else {
        (3, 1)
    }
}

fn count_function_call_tokens(
    tokenizer: &ResolvedTokenizer,
    key: &str,
    value: &Value,
) -> Result<usize, CoreError> {
    match key {
        "tool_calls" => {
            let arr = value.as_array().ok_or(CoreError::InvalidType {
                expected: "array",
                actual: "non-array for tool_calls",
            })?;
            let mut total = 0;
            for tool_call in arr {
                let function = tool_call
                    .get("function")
                    .ok_or(CoreError::MissingField("tool_call.function"))?;
                let args = function
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                total += tokenizer.count(args);
            }
            Ok(total)
        }
        "function_call" => {
            let obj = value.as_object().ok_or(CoreError::InvalidType {
                expected: "object",
                actual: "non-object for function_call",
            })?;
            let args = obj.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
            Ok(tokenizer.count(args))
        }
        _ => Ok(0),
    }
}

fn count_content_list(
    tokenizer: &ResolvedTokenizer,
    content_list: &[Value],
) -> Result<usize, CoreError> {
    let mut num_tokens = 0;

    for item in content_list {
        if let Some(s) = item.as_str() {
            num_tokens += tokenizer.count(s);
            continue;
        }

        let Some(obj) = item.as_object() else {
            continue;
        };

        let Some(type_str) = obj.get("type").and_then(|v| v.as_str()) else {
            continue;
        };

        match type_str {
            "text" => {
                if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    num_tokens += tokenizer.count(text);
                }
            }
            "image_url" | "tool_use" | "tool_result" | "thinking" | "tool_reference" => {
                return Err(CoreError::Unsupported("image or anthropic content blocks"));
            }
            _ => {
                return Err(CoreError::Unsupported("unknown content type in message"));
            }
        }
    }

    Ok(num_tokens)
}

fn count_extra(
    tokenizer: &ResolvedTokenizer,
    tools: Option<&[Map<String, Value>]>,
    tool_choice: Option<&Value>,
    includes_system_message: bool,
) -> usize {
    let mut num_tokens: usize = 3;

    if let Some(tools) = tools
        && !tools.is_empty()
    {
        let mut formatted = String::new();
        format_function_definitions(tools, &mut formatted);
        num_tokens += tokenizer.count(&formatted);
        num_tokens += 9;
    }

    if tools.is_some_and(|t| !t.is_empty()) && includes_system_message {
        num_tokens = num_tokens.saturating_sub(4);
    }

    if let Some(tc) = tool_choice {
        if tc.as_str() == Some("none") {
            num_tokens += 1;
        } else if tc.is_object() {
            num_tokens += 7;
            if let Some(name) = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
            {
                num_tokens += tokenizer.count(name);
            }
        }
    }

    num_tokens
}
