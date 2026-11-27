//! Message redaction for stored requests and responses, the port of LiteLLM's
//! `litellm_core_utils/redact_messages.py` over plain JSON.
//!
//! When message logging is turned off (`litellm.turn_off_message_logging`, the request's dynamic
//! `turn_off_message_logging`, or the `litellm-enable-message-redaction` headers), the row stores
//! the request body and the response with their prompt and completion text replaced by
//! [`REDACTED_BY_LITELLM`]. The glue hands both over as JSON, so only the dict paths of
//! `perform_redaction` apply: a Python object that is not plain data is already a dict here.

use litellm_python_compat::Value;

use crate::RowConfig;
use crate::consts::{
    BEDROCK_GUARDRAIL_TRACE_FIELDS, CLASSIFIER_AUDIT_FIELDS, MATCHED_TEXT_KEYS,
    MESSAGE_REDACTION_ENABLE_HEADERS, NESTED_REQUEST_FIELDS, NUMERIC_COMPRESSION_STAT_KEYS,
    PROMPT_CARRYING_GUARDRAIL_FIELDS, REDACTED_REQUEST_FIELDS, VERTEX_AI_PROVIDER_METADATA_FIELDS,
    WEB_ACTION_LIST_FIELDS, WEB_ACTION_TEXT_FIELDS,
};
use crate::input::{RequestMetadata, RowInput};
use crate::privacy::json_serializable;
use crate::py::{Dict, entries, get, get_or_none, has, text, truthy};

const REDACTED_BY_LITELLM: &str = "redacted-by-litellm";
const DISABLE_HEADER: &str = "litellm-disable-message-redaction";
const SERVED_OUTPUT_TEXTS_KEY: &str = "served_output_texts";
const REDACTED_REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];
const DROPPED_PROVIDER_FIELDS: [&str; 21] = [
    "thinking_blocks",
    "reasoningContentBlocks",
    "reasoningContent",
    "annotations",
    "citations",
    "citation",
    "citationsContent",
    "compaction_blocks",
    "compaction_start",
    "compaction_delta",
    "reasoning_details",
    "web_search_calls",
    "web_search_results",
    "tool_results",
    "code_interpreter_results",
    "search_results",
    "rerank_results",
    "executed_tools",
    "custom_outputs",
    "tool_search_items",
    "token_ids",
];
const METADATA_BUCKETS: [&str; 2] = ["metadata", "litellm_metadata"];

fn redacted() -> Value {
    text(REDACTED_BY_LITELLM)
}

fn is_set(value: &Value, key: &str) -> bool {
    !matches!(get_or_none(value, key), Value::None)
}

fn is_mapping(value: &Value) -> bool {
    matches!(value, Value::Dict(_))
}

/// A copy of `value` changed by `change` when it is a dict, else `value` itself.
fn edited(value: &Value, change: impl FnOnce(&Value, &mut Dict)) -> Value {
    match value {
        Value::Dict(items) => {
            let mut copy = Dict::from_entries(items);
            change(value, &mut copy);
            copy.into_value()
        }
        other => other.clone(),
    }
}

fn mapped(value: &Value, item: impl Fn(&Value) -> Value) -> Value {
    match value {
        Value::List(items) => Value::List(items.iter().map(item).collect()),
        other => other.clone(),
    }
}

/// `should_redact_message_logging(model_call_details)`, in its priority order: the request's
/// dynamic param, then the redaction headers, then the global setting. A disable header wins over
/// all of them.
pub(crate) fn should_redact(config: &RowConfig, input: &RowInput) -> bool {
    let params = &input.litellm_params;
    let bucket = match &params.litellm_metadata {
        RequestMetadata::Mapping(bucket) => Some(bucket),
        _ => params.metadata.mapping(),
    };
    let headers = bucket.map_or(&Value::None, |bucket| get_or_none(bucket, "headers"));
    if truthy(headers) && truthy(get_or_none(headers, DISABLE_HEADER)) {
        return false;
    }
    if let Some(dynamic) = input.kwargs.turn_off_message_logging {
        return dynamic;
    }
    MESSAGE_REDACTION_ENABLE_HEADERS
        .iter()
        .any(|header| truthy(get_or_none(headers, header)))
        || config.turn_off_message_logging
}

fn without_classifier_audit(value: &Value) -> Value {
    Value::Dict(
        entries(value)
            .unwrap_or_default()
            .iter()
            .filter(|(key, _)| {
                !matches!(key, Value::Str(name) if CLASSIFIER_AUDIT_FIELDS.contains(&name.as_str()))
            })
            .cloned()
            .collect(),
    )
}

/// `_redacted_tool_call_dict(call)`.
fn redacted_tool_call_dict(call: &Value) -> Value {
    match get_or_none(call, "function") {
        function @ Value::Dict(_) => edited(call, |_, copy| {
            copy.set(
                "function",
                edited(function, |_, function| {
                    function.set("arguments", redacted())
                }),
            );
        }),
        _ => call.clone(),
    }
}

/// `_redacted_web_search_action(action)`.
fn redacted_web_search_action(action: &Value) -> Value {
    edited(action, |source, copy| {
        for key in WEB_ACTION_TEXT_FIELDS {
            if is_set(source, key) {
                copy.set(key, redacted());
            }
        }
        for key in WEB_ACTION_LIST_FIELDS {
            if is_set(source, key) {
                copy.set(key, Value::None);
            }
        }
    })
}

/// `_redacted_web_search_call(item)`.
fn redacted_web_search_call(item: &Value) -> Value {
    edited(item, |source, copy| {
        if is_set(source, "action") {
            copy.set(
                "action",
                redacted_web_search_action(get_or_none(source, "action")),
            );
        }
    })
}

/// `_redacted_web_search_calls(calls)`: a mapping of calls or a list of them.
fn redacted_web_search_calls(calls: &Value) -> Value {
    match calls {
        Value::Dict(items) => Value::Dict(
            items
                .iter()
                .map(|(key, call)| (key.clone(), redacted_web_search_call(call)))
                .collect(),
        ),
        other => mapped(other, redacted_web_search_call),
    }
}

/// `_redacted_provider_specific_fields(fields)`: reasoning text redacted the way the message's own
/// is, and the provider blocks that echo prompt or completion content dropped.
fn redacted_provider_specific_fields(fields: &Value) -> Value {
    edited(fields, |source, copy| {
        for key in REDACTED_REASONING_FIELDS {
            if is_set(source, key) {
                copy.set(key, redacted());
            }
        }
        for key in DROPPED_PROVIDER_FIELDS {
            if has(source, key) {
                copy.set(key, Value::None);
            }
        }
        if let Some(calls) = get(source, "server_tool_calls") {
            copy.set("server_tool_calls", mapped(calls, redacted_tool_call_dict));
        }
        if let Some(calls) = get(source, "web_search_calls") {
            copy.set("web_search_calls", redacted_web_search_calls(calls));
        }
    })
}

/// A holder with its `provider_specific_fields` redacted, when they are a mapping.
fn with_redacted_provider_fields(holder: &Value) -> Value {
    match get_or_none(holder, "provider_specific_fields") {
        fields @ Value::Dict(_) => edited(holder, |_, copy| {
            copy.set(
                "provider_specific_fields",
                redacted_provider_specific_fields(fields),
            );
        }),
        _ => holder.clone(),
    }
}

/// The message or delta of a choice with its content, refusal, reasoning, audio, images and
/// tool arguments redacted.
fn redacted_choice_body(body: &Value) -> Value {
    edited(body, |source, copy| {
        for field in ["content", "reasoning_content", "refusal"] {
            if is_set(source, field) {
                copy.set(field, redacted());
            }
        }
        for field in ["thinking_blocks", "reasoning_items", "audio", "images"] {
            if has(source, field) {
                copy.set(field, Value::None);
            }
        }
        if let Value::List(calls) = get_or_none(source, "tool_calls") {
            let calls = calls
                .iter()
                .map(|call| match get_or_none(call, "function") {
                    function @ Value::Dict(_) => edited(call, |_, call| {
                        call.set(
                            "function",
                            edited(function, |_, function| {
                                function.set("arguments", redacted())
                            }),
                        );
                    }),
                    _ => call.clone(),
                })
                .collect();
            copy.set("tool_calls", Value::List(calls));
        }
        if let function_call @ Value::Dict(_) = get_or_none(source, "function_call")
            && has(function_call, "arguments")
        {
            copy.set(
                "function_call",
                edited(function_call, |_, call| call.set("arguments", redacted())),
            );
        }
    })
}

/// One dict choice, as `_redact_model_response_dict_choices` leaves it.
fn redacted_choice(choice: &Value) -> Value {
    edited(choice, |source, copy| {
        if let fields @ Value::Dict(_) = get_or_none(source, "provider_specific_fields") {
            copy.set(
                "provider_specific_fields",
                redacted_provider_specific_fields(fields),
            );
        }
        for key in ["message", "delta"] {
            if let holder @ Value::Dict(_) = get_or_none(source, key) {
                copy.set(key, with_redacted_provider_fields(holder));
            }
        }
        if is_set(source, "token_ids") {
            copy.set("token_ids", Value::None);
        }
        let body_key = ["message", "delta"]
            .into_iter()
            .find(|key| is_mapping(get_or_none(source, key)));
        if let Some(key) = body_key {
            let body = copy.get(key).cloned().unwrap_or(Value::None);
            copy.set(key, redacted_choice_body(&body));
        } else if is_set(source, "text") {
            copy.set("text", redacted());
        }
    })
}

/// `redact_vertex_ai_metadata_from_logged_object(obj)` for a dict.
fn without_vertex_metadata(response: &Value) -> Value {
    edited(response, |source, copy| {
        for field in VERTEX_AI_PROVIDER_METADATA_FIELDS {
            if has(source, field) {
                copy.set(field, Value::List(Vec::new()));
            }
        }
        if let hidden @ Value::Dict(_) = get_or_none(source, "_hidden_params") {
            copy.set(
                "_hidden_params",
                edited(hidden, |_, hidden| {
                    for field in VERTEX_AI_PROVIDER_METADATA_FIELDS {
                        hidden.remove(field);
                    }
                }),
            );
        }
    })
}

/// `_without_matched_text(trace)`: every `match` and `regex` value at any depth replaced.
fn without_matched_text(node: &Value) -> Value {
    match node {
        Value::Dict(items) => Value::Dict(
            items
                .iter()
                .map(|(key, value)| {
                    let matched =
                        matches!(key, Value::Str(name) if MATCHED_TEXT_KEYS.contains(&name.as_str()));
                    let stored = if matched {
                        text("[REDACTED]")
                    } else {
                        without_matched_text(value)
                    };
                    (key.clone(), stored)
                })
                .collect(),
        ),
        Value::List(items) => Value::List(items.iter().map(without_matched_text).collect()),
        other => other.clone(),
    }
}

/// `_redact_bedrock_guardrail_trace(response)` for a dict: a Bedrock guardrail trace echoes the
/// matched text of its findings back.
fn without_bedrock_matched_text(response: &Value) -> Value {
    edited(response, |source, copy| {
        for field in BEDROCK_GUARDRAIL_TRACE_FIELDS {
            if let trace @ Value::Dict(_) = get_or_none(source, field) {
                copy.set(field, without_matched_text(trace));
            }
        }
    })
}

/// A chat completion dict with its choices redacted, then its Vertex metadata and Bedrock trace.
fn redacted_model_response(response: &Value) -> Value {
    let with_choices = edited(response, |source, copy| {
        if let choices @ Value::List(_) = get_or_none(source, "choices") {
            copy.set("choices", mapped(choices, redacted_choice));
        }
    });
    without_bedrock_matched_text(&without_vertex_metadata(&with_choices))
}

/// `_request_field_redactions(params)`.
fn request_field_redactions(params: &Value) -> Vec<(&'static str, Value)> {
    let mut redactions: Vec<(&'static str, Value)> = REDACTED_REQUEST_FIELDS
        .into_iter()
        .filter(|field| is_set(params, field))
        .map(|field| (field, redacted()))
        .collect();
    for field in NESTED_REQUEST_FIELDS {
        if let nested @ Value::Dict(_) = get_or_none(params, field) {
            redactions.push((field, with_redacted_request_fields(nested)));
        }
    }
    redactions
}

/// `_with_redacted_request_fields(params)`.
fn with_redacted_request_fields(params: &Value) -> Value {
    edited(params, |source, copy| {
        for (field, value) in request_field_redactions(source) {
            copy.set(field, value);
        }
    })
}

/// `_redacted_responses_api_output_part(part, fields)`.
fn redacted_output_part(part: &Value, fields: &[&str]) -> Value {
    match part {
        Value::Dict(items) => Value::Dict(
            items
                .iter()
                .map(|(key, value)| {
                    let redacts = !matches!(value, Value::None)
                        && matches!(key, Value::Str(name) if fields.contains(&name.as_str()));
                    (
                        key.clone(),
                        if redacts { redacted() } else { value.clone() },
                    )
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `_redacted_responses_api_output_item(item)`.
fn redacted_output_item(item: &Value) -> Value {
    edited(item, |source, copy| {
        let kind = match get_or_none(source, "type") {
            Value::Str(kind) => kind.as_str(),
            _ => "",
        };
        if is_set(source, "text") {
            copy.set("text", redacted());
        }
        if let content @ Value::List(_) = get_or_none(source, "content") {
            copy.set(
                "content",
                mapped(content, |part| {
                    redacted_output_part(part, &["text", "refusal"])
                }),
            );
        }
        if let summary @ Value::List(_) = get_or_none(source, "summary")
            && kind == "reasoning"
        {
            copy.set(
                "summary",
                mapped(summary, |part| redacted_output_part(part, &["text"])),
            );
        }
        let replaced_field = match kind {
            "compaction" => Some("encrypted_content"),
            "function_call" => Some("arguments"),
            "custom_tool_call" => Some("input"),
            _ => None,
        };
        if let Some(field) = replaced_field
            && has(source, field)
        {
            copy.set(field, redacted());
        }
        if kind == "image_generation_call" && is_set(source, "result") {
            copy.set("result", redacted());
        }
        if kind == "web_search_call" && is_set(source, "action") {
            copy.set(
                "action",
                redacted_web_search_action(get_or_none(source, "action")),
            );
        }
    })
}

/// `_redacted_responses_api_response(response)`.
fn redacted_responses_api_response(response: &Value) -> Value {
    let with_output = edited(response, |source, copy| {
        if let output @ Value::List(_) = get_or_none(source, "output") {
            copy.set("output", mapped(output, redacted_output_item));
        }
    });
    with_redacted_request_fields(&with_output)
}

/// `perform_redaction({}, _convert_to_json_serializable_dict(result))`: the response the row
/// stores when message logging is off.
pub(crate) fn redacted_response(response: &Value) -> Value {
    let response = &json_serializable(response);
    if has(response, "choices") {
        redacted_model_response(response)
    } else if has(response, "output") {
        redacted_responses_api_response(response)
    } else {
        Value::Dict(vec![(text("text"), redacted())])
    }
}

/// `numeric_compression_stats(entry)`: the numeric compression counters in the entry's
/// `guardrail_response`, or `None` when it holds none.
fn numeric_compression_stats(entry: &Value) -> Option<Value> {
    let Value::Dict(fields) = get_or_none(entry, "guardrail_response") else {
        return None;
    };
    let stats: Vec<(Value, Value)> = fields
        .iter()
        .filter(|(key, value)| {
            matches!(key, Value::Str(name) if NUMERIC_COMPRESSION_STAT_KEYS.contains(&name.as_str()))
                && matches!(value, Value::Int(_) | Value::Float(_))
        })
        .cloned()
        .collect();
    (!stats.is_empty()).then_some(Value::Dict(stats))
}

/// `_guardrail_entry_without_prompt_text(entry)`: the entry with its prompt-carrying fields
/// redacted, keeping only the numeric compression counters of its `guardrail_response`.
fn guardrail_entry_without_prompt_text(entry: &Value) -> Value {
    let Value::Dict(fields) = entry else {
        return entry.clone();
    };
    let redacted_entry = Value::Dict(
        fields
            .iter()
            .map(|(key, value)| {
                let carries_prompt = !matches!(value, Value::None)
                    && matches!(key, Value::Str(name) if PROMPT_CARRYING_GUARDRAIL_FIELDS.contains(&name.as_str()));
                (
                    key.clone(),
                    if carries_prompt { redacted() } else { value.clone() },
                )
            })
            .collect(),
    );
    match numeric_compression_stats(entry) {
        Some(stats) => edited(&redacted_entry, |_, copy| {
            copy.set("guardrail_response", stats);
        }),
        None => redacted_entry,
    }
}

/// `_guardrail_information_without_prompt_text(guardrail_information)`.
fn guardrail_information_without_prompt_text(information: &Value) -> Value {
    match information {
        Value::List(entries) => Value::List(
            entries
                .iter()
                .map(guardrail_entry_without_prompt_text)
                .collect(),
        ),
        other => guardrail_entry_without_prompt_text(other),
    }
}

/// `_redact_standard_logging_object(payload)`.
fn redacted_standard_logging_object(payload: &Value) -> Value {
    edited(&without_classifier_audit(payload), |source, copy| {
        if is_set(source, "messages") {
            copy.set("messages", redacted_messages());
        }
        if is_set(source, "guardrail_information") {
            copy.set(
                "guardrail_information",
                guardrail_information_without_prompt_text(get_or_none(
                    source,
                    "guardrail_information",
                )),
            );
        }
        if let parameters @ Value::Dict(_) = get_or_none(source, "model_parameters") {
            copy.set("model_parameters", with_redacted_request_fields(parameters));
        }
        let response = get_or_none(source, "response");
        let stored = match response {
            Value::None => return,
            Value::Dict(_) if has(response, "output") => {
                without_vertex_metadata(&redacted_responses_api_response(response))
            }
            Value::Dict(_) if has(response, "choices") => redacted_model_response(response),
            Value::Str(_) => redacted(),
            _ => Value::Dict(vec![(text("text"), redacted())]),
        };
        copy.set("response", stored);
    })
}

fn redacted_messages() -> Value {
    Value::List(vec![Value::Dict(vec![
        (text("role"), text("user")),
        (text("content"), redacted()),
    ])])
}

/// `_redact_hidden_provider_specific_fields(params)`: the provider fields the success handler
/// copied into each metadata bucket's `hidden_params`.
fn with_hidden_provider_fields_redacted(params: &Value) -> Value {
    edited(params, |source, copy| {
        for bucket_key in METADATA_BUCKETS {
            let bucket = get_or_none(source, bucket_key);
            let hidden = get_or_none(bucket, "hidden_params");
            if let fields @ Value::Dict(_) = get_or_none(hidden, "provider_specific_fields") {
                copy.set(
                    bucket_key,
                    edited(bucket, |_, bucket| {
                        bucket.set(
                            "hidden_params",
                            edited(hidden, |_, hidden| {
                                hidden.set(
                                    "provider_specific_fields",
                                    redacted_provider_specific_fields(fields),
                                );
                            }),
                        );
                    }),
                );
            }
        }
    })
}

/// `redacted_litellm_params(params)`: request fields redacted, and the nested proxy request's
/// body redacted like the top-level request.
fn redacted_litellm_params(params: &Value) -> Value {
    edited(&with_redacted_request_fields(params), |_, copy| {
        if let request @ Value::Dict(_) = get_or_none(params, "proxy_server_request") {
            let stored = match get_or_none(request, "body") {
                body @ Value::Dict(_) => {
                    edited(&without_classifier_audit(request), |_, request| {
                        request.set("body", redacted_request(body));
                    })
                }
                _ => without_classifier_audit(request),
            };
            copy.set("proxy_server_request", stored);
        }
    })
}

/// `redact_vertex_ai_metadata_from_litellm_params(details)`.
fn without_vertex_metadata_in_params(params: &Value) -> Value {
    edited(params, |source, copy| {
        for bucket_key in METADATA_BUCKETS {
            let bucket = get_or_none(source, bucket_key);
            if let hidden @ Value::Dict(_) = get_or_none(bucket, "hidden_params") {
                copy.set(
                    bucket_key,
                    edited(bucket, |_, bucket| {
                        bucket.set(
                            "hidden_params",
                            edited(hidden, |_, hidden| {
                                for field in VERTEX_AI_PROVIDER_METADATA_FIELDS {
                                    hidden.remove(field);
                                }
                            }),
                        );
                    }),
                );
            }
        }
    })
}

/// `perform_redaction(details, result=None)` over a request body treated as the call details.
fn redacted_request(details: &Value) -> Value {
    edited(details, |source, copy| {
        if let params @ Value::Dict(_) = get_or_none(source, "litellm_params") {
            copy.set(
                "litellm_params",
                redacted_litellm_params(&with_hidden_provider_fields_redacted(params)),
            );
        }
        copy.set("messages", redacted_messages());
        copy.set("prompt", text(""));
        copy.set("input", text(""));
        let current = copy.clone().into_value();
        let mut updates = request_field_redactions(&current);
        if let optional @ Value::Dict(items) = get_or_none(&current, "optional_params")
            && !items.is_empty()
        {
            updates.push(("optional_params", with_redacted_request_fields(optional)));
        }
        for (field, value) in updates {
            copy.set(field, value);
        }
        copy.remove(SERVED_OUTPUT_TEXTS_KEY);
        if let payload @ Value::Dict(_) = get_or_none(source, "standard_logging_object") {
            copy.set(
                "standard_logging_object",
                redacted_standard_logging_object(payload),
            );
        }
        if let Some(params) = copy
            .get("litellm_params")
            .filter(|params| is_mapping(params))
        {
            let stored = without_vertex_metadata_in_params(params);
            copy.set("litellm_params", stored);
        }
    })
}

/// `_redacted_when_message_logging_is_off(body, ...)` once redaction is on: the body without its
/// classifier audit fields, converted as `_convert_to_json_serializable_dict` converts it, then
/// redacted as `perform_redaction` redacts the call details.
pub(crate) fn redacted_request_body(body: &Value) -> Value {
    redacted_request(&json_serializable(&without_classifier_audit(body)))
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;

    use super::{redacted_request_body, redacted_response};

    fn parsed(source: &str) -> litellm_python_compat::Value {
        json::loads(source).expect("test json parses")
    }

    #[test]
    fn chat_choices_lose_their_text() {
        let response = parsed(
            r#"{"id":"x","choices":[{"message":{"content":"hi","tool_calls":[{"function":{"name":"f","arguments":"{}"}}],"provider_specific_fields":{"citations":[1]}}}],"trace":{"a":[{"match":"secret"}]}}"#,
        );
        assert_eq!(
            json::dumps_compact(&redacted_response(&response)).unwrap(),
            r#"{"id":"x","choices":[{"message":{"content":"redacted-by-litellm","tool_calls":[{"function":{"name":"f","arguments":"redacted-by-litellm"}}],"provider_specific_fields":{"citations":null}}}],"trace":{"a":[{"match":"[REDACTED]"}]}}"#
        );
        assert_eq!(
            json::dumps_compact(&redacted_response(&parsed("[1]"))).unwrap(),
            r#"{"text":"redacted-by-litellm"}"#
        );
    }

    #[test]
    fn request_bodies_keep_their_shape() {
        let body = parsed(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"instructions":"x","classifier_input":{},"served_output_texts":["a"]}"#,
        );
        assert_eq!(
            json::dumps_compact(&redacted_request_body(&body)).unwrap(),
            r#"{"model":"m","messages":[{"role":"user","content":"redacted-by-litellm"}],"instructions":"redacted-by-litellm","prompt":"","input":""}"#
        );
    }

    // Expected value is perform_redaction(model_call_details=body, result=None) on CPython
    #[test]
    fn guardrail_prompt_text_and_images_are_redacted() {
        let body = parsed(
            r#"{"model":"m","images":["data:x"],"standard_logging_object":{"messages":[{"role":"user","content":"hi"}],"guardrail_information":[{"guardrail_name":"g","guardrail_request":{"text":"hi"},"guardrail_response":{"tokens_before":10,"tokens_after":4,"flag":true,"note":"x"},"match_details":null,"classification":"c"},{"guardrail_response":"blocked"},"raw"],"model_parameters":{"images":["data:y"],"temperature":1}}}"#,
        );
        assert_eq!(
            json::dumps_compact(&redacted_request_body(&body)).unwrap(),
            r#"{"model":"m","images":"redacted-by-litellm","standard_logging_object":{"messages":[{"role":"user","content":"redacted-by-litellm"}],"guardrail_information":[{"guardrail_name":"g","guardrail_request":"redacted-by-litellm","guardrail_response":{"tokens_before":10,"tokens_after":4},"match_details":null,"classification":"redacted-by-litellm"},{"guardrail_response":"redacted-by-litellm"},"raw"],"model_parameters":{"images":"redacted-by-litellm","temperature":1}},"messages":[{"role":"user","content":"redacted-by-litellm"}],"prompt":"","input":""}"#
        );
    }

    fn nested(levels: usize, leaf: &str) -> String {
        format!("{}{leaf}{}", r#"{"a":"#.repeat(levels), "}".repeat(levels))
    }

    // Expected values are _convert_to_json_serializable_dict then perform_redaction on CPython
    #[test]
    fn nesting_past_twenty_levels_is_cut() {
        let deep = nested(21, "1");
        let cut = nested(19, r#""<max_depth_exceeded>""#);
        assert_eq!(
            json::dumps_compact(&redacted_request_body(&parsed(&format!(
                r#"{{"model":"m","deep":{deep}}}"#
            ))))
            .unwrap(),
            format!(
                r#"{{"model":"m","deep":{cut},"messages":[{{"role":"user","content":"redacted-by-litellm"}}],"prompt":"","input":""}}"#
            )
        );
        assert_eq!(
            json::dumps_compact(&redacted_response(&parsed(&format!(
                r#"{{"id":"x","choices":[{{"message":{{"content":"hi"}}}}],"deep":{deep}}}"#
            ))))
            .unwrap(),
            format!(
                r#"{{"id":"x","choices":[{{"message":{{"content":"redacted-by-litellm"}}}}],"deep":{cut}}}"#
            )
        );
    }
}
