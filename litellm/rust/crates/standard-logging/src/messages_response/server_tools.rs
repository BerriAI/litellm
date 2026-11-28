//! The logged fields of the tools Anthropic runs itself (`server_tool_use` blocks and their
//! `*_tool_result` blocks), which both the complete answer's transform and the stream's chunk
//! parser write: `build_web_search_call` and `build_code_interpreter_log_outputs` in
//! `litellm/types/responses/main.py`, and the tool call `convert_tool_use_to_openai_format`
//! writes into `server_tool_calls`.

use serde_json::{Map, Value, json};

/// `HOSTED_WEB_TOOL_NAMES`: the server tools whose calls are logged as `web_search_calls`.
pub(super) fn is_hosted_web_tool(name: &str) -> bool {
    matches!(name, "web_search" | "web_fetch")
}

/// What a code execution's result logs as its code interpreter call's outputs.
pub(super) const CODE_EXECUTION_RESULT: &str = "bash_code_execution_tool_result";

/// A tool search result, which logs no code interpreter call.
pub(super) const TOOL_SEARCH_RESULT: &str = "tool_search_tool_result";

/// A web search or fetch result, which `extract_response_content` keeps apart as
/// `web_search_results`.
pub(super) fn is_web_result(kind: &str) -> bool {
    matches!(kind, "web_search_tool_result" | "web_fetch_tool_result")
}

/// `ActionSearch` over the call's input and the results it found.
fn search_action(input: &Value, content: Option<&Value>) -> Value {
    let query = input
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let sources: Vec<Value> = content
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("web_search_result"))
        .filter_map(|item| item.get("url").and_then(Value::as_str))
        .map(|url| json!({"type": "url", "url": url}))
        .collect();
    let queries: Vec<&str> = (!query.is_empty()).then_some(query).into_iter().collect();
    json!({"query": query, "type": "search", "queries": queries, "sources": sources})
}

/// `ResponseFunctionWebSearch.model_dump()` of `build_web_search_call`. `status` is the one the
/// caller names, else `failed` for an error result and `completed` for any other.
pub(super) fn web_search_call(
    tool_id: &str,
    input: &Value,
    result: Option<&Value>,
    status: Option<&'static str>,
    tool_name: &str,
) -> Value {
    let content = result
        .and_then(Value::as_object)
        .and_then(|result| result.get("content"));
    let errored = content
        .and_then(|content| content.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == format!("{tool_name}_tool_result_error"));
    let status = status.unwrap_or(if errored { "failed" } else { "completed" });
    let action = match tool_name {
        "web_fetch" => {
            json!({"type": "open_page", "url": input.get("url").and_then(Value::as_str)})
        }
        _ => search_action(input, content),
    };
    json!({"id": format!("ws_{tool_id}"), "action": action, "status": status, "type": "web_search_call"})
}

/// `OutputCodeInterpreterCall.model_dump()` of a code execution result, its outputs as
/// `build_code_interpreter_log_outputs` writes them.
pub(super) fn code_interpreter_call(
    result: &Map<String, Value>,
    code: &str,
    container_id: Option<&str>,
) -> Value {
    let content = result.get("content").and_then(Value::as_object);
    let outputs = content.and_then(|content| {
        let text = |key: &str| {
            content
                .get(key)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
        };
        let logs: String = text("stdout")
            .map(str::to_owned)
            .into_iter()
            .chain(text("stderr").map(|stderr| format!("STDERR: {stderr}")))
            .collect();
        (!logs.is_empty()).then(|| json!([{"type": "logs", "logs": logs}]))
    });
    json!({
        "type": "code_interpreter_call",
        "id": result.get("tool_use_id").and_then(Value::as_str).unwrap_or_default(),
        "code": code,
        "container_id": container_id,
        "status": "completed",
        "outputs": outputs,
    })
}

/// A server tool call as `server_tool_calls` lists it: the `ChatCompletionToolCallChunk`
/// `convert_tool_use_to_openai_format` builds, with the caller of a programmatic call.
pub(super) fn server_tool_call(
    id: &str,
    name: &str,
    arguments: &str,
    index: usize,
    caller: Option<&Value>,
) -> Value {
    let mut call = json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        "index": index,
    });
    if let (Some(caller), Value::Object(fields)) = (caller, &mut call) {
        fields.insert("caller".to_owned(), caller.clone());
    }
    call
}
