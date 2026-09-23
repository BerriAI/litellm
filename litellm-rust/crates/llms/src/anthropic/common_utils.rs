use serde_json::Value;
use std::collections::HashMap;

const ENCRYPTED_REASONING_SIGNATURE_PREFIX: &str = "litellm_encrypted_reasoning:";
const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

fn with_content(message: Value, content: Vec<Value>) -> Value {
    match message {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .filter(|(key, _)| key != "content")
                .chain([("content".into(), Value::Array(content))])
                .collect(),
        ),
        other => other,
    }
}

fn content_blocks(message: &Value) -> Option<&Vec<Value>> {
    message.get("content")?.as_array()
}

fn is_empty_content_block(block: &Value) -> bool {
    let field = match block.get("type").and_then(Value::as_str) {
        Some("text") => "text",
        Some("thinking") => "thinking",
        _ => return false,
    };
    block
        .get(field)
        .and_then(Value::as_str)
        .is_none_or(|text| text.trim().is_empty())
}

pub fn strip_empty_content_blocks(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .filter_map(|message| {
            let Some(content) = content_blocks(&message) else {
                return Some(message);
            };
            let filtered: Vec<Value> = content
                .iter()
                .filter(|block| !is_empty_content_block(block))
                .cloned()
                .collect();
            if filtered.len() == content.len() {
                Some(message)
            } else if filtered.is_empty() {
                None
            } else {
                Some(with_content(message, filtered))
            }
        })
        .collect()
}

pub fn normalize_tool_use_id(raw_id: &str) -> String {
    let base = raw_id
        .split_once(THOUGHT_SIGNATURE_SEPARATOR)
        .map_or(raw_id, |(base, _)| base);
    let normalized: String = base
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if normalized.is_empty() {
        "tool_use_id".into()
    } else {
        normalized
    }
}

fn sanitize_tool_use_id_block(block: &Value) -> Value {
    let field = match block.get("type").and_then(Value::as_str) {
        Some("tool_use" | "server_tool_use") => "id",
        Some("tool_result") => "tool_use_id",
        _ => return block.clone(),
    };
    let Some(id) = block.get(field).and_then(Value::as_str) else {
        return block.clone();
    };
    let normalized = normalize_tool_use_id(id);
    if normalized == id {
        return block.clone();
    }
    Value::Object(
        block
            .as_object()
            .into_iter()
            .flat_map(|fields| fields.iter())
            .map(|(key, value)| {
                (
                    key.clone(),
                    if key == field {
                        Value::String(normalized.clone())
                    } else {
                        value.clone()
                    },
                )
            })
            .collect(),
    )
}

pub fn sanitize_tool_use_ids(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|message| match content_blocks(&message) {
            Some(content) => with_content(
                message.clone(),
                content.iter().map(sanitize_tool_use_id_block).collect(),
            ),
            None => message,
        })
        .collect()
}

fn is_encrypted_reasoning_block(block: &Value) -> bool {
    let field = match block.get("type").and_then(Value::as_str) {
        Some("thinking") => "signature",
        Some("redacted_thinking") => "data",
        _ => return false,
    };
    block
        .get(field)
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

pub fn strip_encrypted_reasoning_blocks(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .filter_map(|message| {
            let Some(content) = content_blocks(&message) else {
                return Some(message);
            };
            let filtered: Vec<Value> = content
                .iter()
                .filter(|block| !is_encrypted_reasoning_block(block))
                .cloned()
                .collect();
            if filtered.len() == content.len() {
                Some(message)
            } else if filtered.is_empty() {
                None
            } else {
                Some(with_content(message, filtered))
            }
        })
        .collect()
}

fn strip_provider_specific_fields(block: &Value) -> Value {
    match block {
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .filter(|(key, _)| *key != "provider_specific_fields")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub fn strip_provider_specific_fields_from_messages(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|message| match content_blocks(&message) {
            Some(content) => with_content(
                message.clone(),
                content.iter().map(strip_provider_specific_fields).collect(),
            ),
            None => message,
        })
        .collect()
}

fn advisor_use_id(block: &Value) -> Option<&str> {
    (block.get("type")?.as_str()? == "server_tool_use" && block.get("name")?.as_str()? == "advisor")
        .then(|| block.get("id")?.as_str())
        .flatten()
}

pub fn strip_advisor_blocks(messages: Vec<Value>, tools: &[Value]) -> Vec<Value> {
    if tools
        .iter()
        .any(|tool| tool.get("type").and_then(Value::as_str) == Some("advisor_20260301"))
    {
        return messages;
    }
    messages
        .into_iter()
        .map(|message| {
            if message.get("role").and_then(Value::as_str) != Some("assistant") {
                return message;
            }
            let Some(content) = content_blocks(&message) else {
                return message;
            };
            let advisor_ids: Vec<&str> = content.iter().filter_map(advisor_use_id).collect();
            if advisor_ids.is_empty() {
                return message;
            }
            with_content(
                message.clone(),
                content
                    .iter()
                    .filter(|block| {
                        !advisor_use_id(block).is_some_and(|id| advisor_ids.contains(&id))
                            && !(block.get("type").and_then(Value::as_str)
                                == Some("advisor_tool_result")
                                && block
                                    .get("tool_use_id")
                                    .and_then(Value::as_str)
                                    .is_some_and(|id| advisor_ids.contains(&id)))
                    })
                    .cloned()
                    .collect(),
            )
        })
        .collect()
}

fn flattenable_web_search_result(block: &Value) -> Option<(&str, &Value)> {
    if block.get("type")?.as_str()? != "web_search_tool_result" {
        return None;
    }
    let id = block.get("tool_use_id")?.as_str()?;
    let content = block.get("content")?;
    match content {
        Value::Array(results)
            if results.iter().all(|result| {
                result.get("type").and_then(Value::as_str) == Some("web_search_result")
                    && !result
                        .get("encrypted_content")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty())
            }) =>
        {
            Some((id, content))
        }
        Value::Object(fields)
            if fields.get("type").and_then(Value::as_str)
                == Some("web_search_tool_result_error") =>
        {
            Some((id, content))
        }
        _ => None,
    }
}

fn web_search_line(prefix: &str, result: &Value, field: &str) -> Option<String> {
    result
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(|value| format!("{prefix}: {value}"))
}

fn render_web_search_results(query: &str, content: &Value) -> String {
    let header = if query.is_empty() {
        "Web search results:".to_string()
    } else {
        format!("Web search results for '{query}':")
    };
    match content {
        Value::Object(fields) => {
            let code = fields
                .get("error_code")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or("unavailable");
            format!("{header}\n\nSearch failed: {code}")
        }
        Value::Array(results) if results.is_empty() => {
            format!("{header}\n\nNo results were returned.")
        }
        Value::Array(results) => {
            let body = results
                .iter()
                .map(|result| {
                    [
                        web_search_line("Title", result, "title"),
                        web_search_line("URL", result, "url"),
                        web_search_line("Snippet", result, "snippet"),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("\n")
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            if body.is_empty() {
                header
            } else {
                format!("{header}\n\n{body}")
            }
        }
        _ => header,
    }
}

fn flatten_web_search_results_in_message(message: Value) -> Value {
    let Some(content) = content_blocks(&message) else {
        return message;
    };
    let results: HashMap<&str, &Value> = content
        .iter()
        .filter_map(flattenable_web_search_result)
        .collect();
    if results.is_empty() {
        return message;
    }
    let queries: HashMap<&str, &str> = content
        .iter()
        .filter_map(|block| {
            let id = (block.get("type")?.as_str()? == "server_tool_use")
                .then(|| block.get("id")?.as_str())
                .flatten()?;
            let query = block
                .get("input")
                .and_then(|input| input.get("query"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Some((id, query))
        })
        .collect();
    let rewritten = content
        .iter()
        .filter_map(|block| {
            if let Some((id, result)) = flattenable_web_search_result(block) {
                return Some(serde_json::json!({
                    "type": "text",
                    "text": render_web_search_results(queries.get(id).copied().unwrap_or(""), result)
                }));
            }
            let paired_use = block.get("type").and_then(Value::as_str) == Some("server_tool_use")
                && block
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| results.contains_key(id));
            (!paired_use).then(|| block.clone())
        })
        .collect();
    with_content(message, rewritten)
}

pub fn flatten_unencrypted_web_search_results(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(flatten_web_search_results_in_message)
        .collect()
}

pub fn normalize_messages(messages: Vec<Value>, tools: &[Value]) -> Vec<Value> {
    strip_provider_specific_fields_from_messages(strip_advisor_blocks(
        strip_encrypted_reasoning_blocks(flatten_unencrypted_web_search_results(
            sanitize_tool_use_ids(strip_empty_content_blocks(messages)),
        )),
        tools,
    ))
}

pub fn is_invalid_thinking_error(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("thinking")
        && ((lower.contains("signature")
            && (lower.contains("invalid") || lower.contains("valid string")))
            || lower.contains("must contain thinking"))
}

pub fn strip_thinking_blocks_from_request(body: &Value) -> Value {
    let Value::Object(fields) = body else {
        return body.clone();
    };
    Value::Object(
        fields
            .iter()
            .filter_map(|(name, value)| {
                if name == "thinking" {
                    return None;
                }
                if name != "messages" {
                    return Some((name.clone(), value.clone()));
                }
                let Value::Array(messages) = value else {
                    return Some((name.clone(), value.clone()));
                };
                let filtered: Vec<Value> = messages
                    .iter()
                    .filter_map(|message| {
                        let Some(content) = content_blocks(message) else {
                            return Some(message.clone());
                        };
                        let blocks: Vec<Value> = content
                            .iter()
                            .filter(|block| {
                                !matches!(
                                    block.get("type").and_then(Value::as_str),
                                    Some("thinking" | "redacted_thinking")
                                )
                            })
                            .cloned()
                            .collect();
                        (!blocks.is_empty()).then(|| with_content(message.clone(), blocks))
                    })
                    .collect();
                Some((name.clone(), Value::Array(filtered)))
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn removes_empty_text_and_thinking_without_dropping_tool_use() {
        let messages = vec![json!({"role":"assistant","content":[
            {"type":"text","text":"  "},
            {"type":"thinking","thinking":""},
            {"type":"redacted_thinking","data":"opaque"},
            {"type":"tool_use","id":"tool_1","name":"run","input":{}}
        ]})];
        assert_eq!(
            strip_empty_content_blocks(messages),
            vec![json!({"role":"assistant","content":[
                {"type":"redacted_thinking","data":"opaque"},
                {"type":"tool_use","id":"tool_1","name":"run","input":{}}
            ]})]
        );
    }

    #[test]
    fn normalizes_replayed_tool_ids_consistently() {
        let messages = vec![json!({"role":"assistant","content":[
            {"type":"tool_use","id":"functions.Bash:0__thought__opaque"},
            {"type":"tool_result","tool_use_id":"functions.Bash:0__thought__opaque"}
        ]})];
        assert_eq!(
            sanitize_tool_use_ids(messages)[0]["content"],
            json!([
                {"type":"tool_use","id":"functions_Bash_0"},
                {"type":"tool_result","tool_use_id":"functions_Bash_0"}
            ])
        );
    }

    #[test]
    fn removes_foreign_reasoning_and_provider_fields() {
        let messages = vec![json!({"role":"assistant","content":[
            {"type":"thinking","thinking":"secret","signature":"litellm_encrypted_reasoning:blob"},
            {"type":"text","text":"hello","provider_specific_fields":{"x":1}}
        ]})];
        assert_eq!(
            normalize_messages(messages, &[]),
            vec![json!({"role":"assistant","content":[{"type":"text","text":"hello"}]})]
        );
    }

    #[test]
    fn flattens_replayed_search_results_but_keeps_encrypted_results() {
        let messages = vec![json!({"role":"assistant","content":[
            {"type":"server_tool_use","id":"search_1","name":"web_search","input":{"query":"rust"}},
            {"type":"web_search_tool_result","tool_use_id":"search_1","content":[
                {"type":"web_search_result","title":"Guide","url":"https://example.test","snippet":"Useful"}
            ]},
            {"type":"web_search_tool_result","tool_use_id":"search_2","content":[
                {"type":"web_search_result","encrypted_content":"opaque"}
            ]}
        ]})];
        assert_eq!(
            flatten_unencrypted_web_search_results(messages)[0]["content"],
            json!([
                {"type":"text","text":"Web search results for 'rust':\n\nTitle: Guide\nURL: https://example.test\nSnippet: Useful"},
                {"type":"web_search_tool_result","tool_use_id":"search_2","content":[
                    {"type":"web_search_result","encrypted_content":"opaque"}
                ]}
            ])
        );
    }

    #[test]
    fn invalid_signature_retry_removes_thinking_without_mutating_other_content() {
        assert!(is_invalid_thinking_error(
            "Invalid signature in thinking block"
        ));
        assert!(!is_invalid_thinking_error("Invalid tool signature"));
        let original = json!({"thinking":{"type":"enabled","budget_tokens":1024},"messages":[
            {"role":"assistant","content":[{"type":"thinking","thinking":"old","signature":"bad"},{"type":"text","text":"answer"}]},
            {"role":"assistant","content":[{"type":"redacted_thinking","data":"old"}]}
        ]});
        assert_eq!(
            strip_thinking_blocks_from_request(&original),
            json!({"messages":[
                {"role":"assistant","content":[{"type":"text","text":"answer"}]}
            ]})
        );
        assert!(original.get("thinking").is_some());
    }
}
