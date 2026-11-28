//! A complete Messages answer as Python logs it: `AnthropicConfig.transform_response`, then
//! `transform_parsed_response`, called by `_handle_anthropic_messages_response_logging` on the
//! provider's response body.

use std::collections::{HashMap, HashSet};

use litellm_python_compat::json::{dumps, from_json};
use serde::Deserialize;
use serde_json::{Map, Value};

use super::{
    ASSISTANT, CHAT_COMPLETION, CompleteMessage, CompleteToolCall, Error, FUNCTION, LoggedChoice,
    LoggedMessage, LoggedResponse, LoggedUsage, Null, ResponseContext, ToolFunction, finish_reason,
    refusal,
    server_tools::{
        CODE_EXECUTION_RESULT, TOOL_SEARCH_RESULT, code_interpreter_call, is_hosted_web_tool,
        is_web_result, server_tool_call, web_search_call,
    },
    stop_reason_fields,
    usage::{CompleteUsage, Reasoning, ReportedUsage, calculate},
};

/// The fields of a Messages body the transform reads.
#[derive(Deserialize)]
struct Body {
    #[serde(default)]
    id: Option<Value>,
    model: String,
    content: Vec<Map<String, Value>>,
    #[serde(default)]
    stop_reason: Option<Value>,
    #[serde(default)]
    stop_details: Option<Value>,
    usage: Map<String, Value>,
    #[serde(default)]
    context_management: Option<Value>,
    #[serde(default)]
    input_transformations: Option<Value>,
    #[serde(default)]
    container: Option<Value>,
}

#[derive(Deserialize)]
struct ToolUse {
    id: String,
    name: String,
    input: Value,
}

const TYPE: &str = "type";
const TEXT: &str = "text";
const THINKING: &str = "thinking";
const CITATIONS: &str = "citations";
const THINKING_BLOCKS: &str = "thinking_blocks";
const SERVER_TOOL_USE: &str = "server_tool_use";

/// A tool call block as `convert_tool_use_to_openai_format` turns it into a tool call, and the
/// input it was written from.
struct ToolCallBlock {
    call: CompleteToolCall,
    input: Value,
}

/// What `extract_response_content` reads off the content blocks.
#[derive(Default)]
struct Content {
    text: String,
    citations: Option<Vec<Value>>,
    thinking_blocks: Option<Vec<Map<String, Value>>>,
    /// Every `tool_use` and `server_tool_use` block as a tool call.
    tool_calls: Vec<ToolCallBlock>,
    web_search_results: Option<Vec<Value>>,
    tool_results: Option<Vec<Map<String, Value>>>,
    compaction_blocks: Option<Vec<Value>>,
}

fn content(blocks: &[Map<String, Value>]) -> Result<Content, Error> {
    let mut content = Content::default();
    for (index, block) in blocks.iter().enumerate() {
        let kind = block
            .get(TYPE)
            .and_then(Value::as_str)
            .ok_or(Error::UntypedBlock)?;
        match kind {
            TEXT => {
                if let Some(text) = block.get(TEXT).and_then(Value::as_str) {
                    content.text.push_str(text);
                }
            }
            "tool_use" | SERVER_TOOL_USE => content.tool_calls.push(tool_call(block, index)?),
            _ if kind.ends_with("_tool_result") && is_web_result(kind) => content
                .web_search_results
                .get_or_insert_with(Vec::new)
                .push(Value::Object(block.clone())),
            _ if kind.ends_with("_tool_result") => content
                .tool_results
                .get_or_insert_with(Vec::new)
                .push(block.clone()),
            THINKING | "redacted_thinking" => content
                .thinking_blocks
                .get_or_insert_with(Vec::new)
                .push(block.clone()),
            "compaction" => content
                .compaction_blocks
                .get_or_insert_with(Vec::new)
                .push(Value::Object(block.clone())),
            _ => {}
        }
        if let Some(Value::Array(citations)) = block.get(CITATIONS) {
            let supported_text = block.get(TEXT).cloned().unwrap_or_else(|| Value::from(""));
            let cited = citations
                .iter()
                .map(|citation| {
                    let mut citation = citation.as_object().cloned().unwrap_or_default();
                    citation.insert("supported_text".to_owned(), supported_text.clone());
                    Value::Object(citation)
                })
                .collect();
            content
                .citations
                .get_or_insert_with(Vec::new)
                .push(Value::Array(cited));
        }
    }
    Ok(content)
}

/// The ids of the answer's `server_tool_use` blocks (`_server_tool_use_ids`).
fn server_tool_use_ids(blocks: &[Map<String, Value>]) -> HashSet<&str> {
    blocks
        .iter()
        .filter(|block| block.get(TYPE).and_then(Value::as_str) == Some(SERVER_TOOL_USE))
        .filter_map(|block| block.get("id").and_then(Value::as_str))
        .collect()
}

/// How many tool search calls the content blocks make, which `calculate_usage` counts when
/// the usage reports none.
fn tool_searches(blocks: &[Map<String, Value>]) -> u64 {
    let searches = blocks
        .iter()
        .filter(|block| block.get(TYPE).and_then(Value::as_str) == Some(SERVER_TOOL_USE))
        .filter(|block| {
            block
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| name.contains("tool_search"))
        })
        .count();
    u64::try_from(searches).unwrap_or(u64::MAX)
}

/// `_build_web_search_calls`: each hosted web tool call, with the result that answers it.
fn web_search_calls(blocks: &[Map<String, Value>], results: &[Value]) -> Vec<Value> {
    let result_of = |call_id: &str, kind: &str| {
        results.iter().rev().find(|result| {
            result.get("tool_use_id").and_then(Value::as_str) == Some(call_id)
                && result.get(TYPE).and_then(Value::as_str) == Some(kind)
        })
    };
    blocks
        .iter()
        .filter(|block| block.get(TYPE).and_then(Value::as_str) == Some(SERVER_TOOL_USE))
        .filter_map(|block| {
            let name = block.get("name").and_then(Value::as_str)?;
            let call_id = block.get("id").and_then(Value::as_str)?;
            let input = block.get("input")?;
            is_hosted_web_tool(name).then(|| {
                let result = result_of(call_id, &format!("{name}_tool_result"));
                let status = result.is_none().then_some("in_progress");
                web_search_call(call_id, input, result, status, name)
            })
        })
        .collect()
}

/// `_build_code_interpreter_results`: each code execution result, with the command its call
/// ran (`_build_code_by_id_map`).
fn code_interpreter_results(
    tool_results: &[Map<String, Value>],
    tool_calls: &[ToolCallBlock],
    container_id: Option<&str>,
) -> Vec<Value> {
    let code_by_id: HashMap<&str, &str> = tool_calls
        .iter()
        .filter(|block| block.input.is_object())
        .map(|block| {
            let command = block
                .input
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default();
            (block.call.id.as_str(), command)
        })
        .collect();
    tool_results
        .iter()
        .filter(|result| result.get(TYPE).and_then(Value::as_str) == Some(CODE_EXECUTION_RESULT))
        .map(|result| {
            let call_id = result
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let code = code_by_id.get(call_id).copied().unwrap_or_default();
            code_interpreter_call(result, code, container_id)
        })
        .collect()
}

/// `convert_tool_use_to_openai_format`: the input as `json.dumps` writes it, and the caller
/// of a programmatic tool call when the block names one.
fn tool_call(block: &Map<String, Value>, index: usize) -> Result<ToolCallBlock, Error> {
    let ToolUse { id, name, input } =
        ToolUse::deserialize(Value::Object(block.clone())).map_err(Error::Unreadable)?;
    Ok(ToolCallBlock {
        call: CompleteToolCall {
            index,
            caller: block.get("caller").cloned(),
            function: ToolFunction {
                arguments: dumps(&from_json(input.clone())).map_err(Error::ToolInput)?,
                name,
            },
            id,
            kind: FUNCTION,
        },
        input,
    })
}

/// `"".join(block["thinking"] for block in thinking_blocks)`, skipping a block without text.
fn reasoning_content(blocks: &[Map<String, Value>]) -> String {
    blocks
        .iter()
        .filter_map(|block| block.get(THINKING).and_then(Value::as_str))
        .collect()
}

/// The `ModelResponse` Python logs for a complete Messages answer, `body` as the provider sent
/// it.
pub fn complete_response(
    body: &Map<String, Value>,
    context: &ResponseContext<'_>,
) -> Result<LoggedResponse, Error> {
    if body.contains_key("error") {
        return Err(Error::ErrorBody);
    }
    let body = Body::deserialize(Value::Object(body.clone())).map_err(Error::Unreadable)?;
    let Content {
        text,
        citations,
        thinking_blocks,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    } = content(&body.content)?;
    let reasoning_content = thinking_blocks.as_deref().map(reasoning_content);
    let usage = calculate(
        ReportedUsage::read(&body.usage),
        Reasoning {
            content: reasoning_content.as_deref(),
            has_thinking_block: thinking_blocks.is_some(),
        },
        tool_searches(&body.content),
        context.requested_speed,
        context.count_tokens,
    );
    let thinking_value = thinking_blocks
        .as_ref()
        .map(|blocks| Value::Array(blocks.iter().cloned().map(Value::Object).collect()));
    let server_ids = server_tool_use_ids(&body.content);
    let web_search_calls = web_search_calls(
        &body.content,
        web_search_results.as_deref().unwrap_or_default(),
    );
    let container = body.container.filter(|container| !container.is_null());
    let code_interpreter_results = tool_results
        .as_ref()
        .filter(|results| {
            results
                .iter()
                .any(|result| result.get(TYPE).and_then(Value::as_str) != Some(TOOL_SEARCH_RESULT))
        })
        .map(|results| {
            let container_id = container
                .as_ref()
                .and_then(|container| container.get("id"))
                .and_then(Value::as_str);
            Value::Array(code_interpreter_results(results, &tool_calls, container_id))
        });
    let server_tool_calls: Vec<Value> = tool_calls
        .iter()
        .filter(|block| server_ids.contains(block.call.id.as_str()))
        .map(|block| {
            let call = &block.call;
            server_tool_call(
                &call.id,
                &call.function.name,
                &call.function.arguments,
                call.index,
                call.caller.as_ref(),
            )
        })
        .collect();
    let named_when_set = [
        ("context_management", body.context_management),
        ("input_transformations", body.input_transformations),
    ]
    .into_iter()
    .filter_map(|(key, value)| Some((key, value.filter(|v| !v.is_null())?)));
    let web_search_calls = (web_search_results.is_some() || !web_search_calls.is_empty())
        .then_some(Value::Array(web_search_calls));
    let server_tool_fields = [
        ("web_search_results", web_search_results.map(Value::Array)),
        ("web_search_calls", web_search_calls),
        (
            "tool_results",
            tool_results
                .map(|results| Value::Array(results.into_iter().map(Value::Object).collect())),
        ),
        ("code_interpreter_results", code_interpreter_results),
        ("container", container),
        ("compaction_blocks", compaction_blocks.map(Value::Array)),
        (
            "server_tool_calls",
            (!server_tool_calls.is_empty()).then_some(Value::Array(server_tool_calls)),
        ),
    ]
    .into_iter()
    .filter_map(|(key, value)| Some((key, value?)));
    let provider_specific_fields = [
        (CITATIONS, citations.map_or(Value::Null, Value::Array)),
        (THINKING_BLOCKS, thinking_value.unwrap_or(Value::Null)),
    ]
    .into_iter()
    .chain(named_when_set)
    .chain(server_tool_fields)
    .map(|(key, value)| (key.to_owned(), value))
    .chain(stop_reason_fields(
        body.stop_reason.as_ref(),
        body.stop_details.as_ref(),
    ))
    .collect();
    let tool_calls: Vec<CompleteToolCall> = tool_calls
        .into_iter()
        .map(|block| block.call)
        .filter(|call| !server_ids.contains(call.id.as_str()))
        .collect();
    let id = body
        .id
        .as_ref()
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .unwrap_or(context.fallback_id);
    Ok(LoggedResponse {
        id: id.to_owned(),
        created: context.created,
        model: body.model,
        object: CHAT_COMPLETION,
        system_fingerprint: None,
        choices: vec![LoggedChoice {
            finish_reason: finish_reason(body.stop_reason.as_ref().and_then(Value::as_str)),
            index: 0,
            message: LoggedMessage::Complete(CompleteMessage {
                content: (!text.is_empty()).then_some(text),
                role: ASSISTANT,
                refusal: refusal(body.stop_reason.as_ref(), body.stop_details.as_ref()),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                function_call: Null,
                reasoning_content,
                thinking_blocks,
                provider_specific_fields,
            }),
        }],
        usage: LoggedUsage::Complete(CompleteUsage::from(usage)),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::super::tests::{context, counter, fixture};
    use super::*;

    #[rstest]
    #[case::text("text")]
    #[case::max_tokens("max_tokens")]
    #[case::stop_sequence("stop_sequence")]
    #[case::tool_use("tool_use")]
    #[case::two_tools("two_tools")]
    #[case::thinking("thinking")]
    #[case::redacted_thinking("redacted_thinking")]
    #[case::cache("cache")]
    #[case::empty("empty")]
    #[case::refusal("refusal")]
    #[case::pause_turn("pause_turn")]
    #[case::citations("citations")]
    #[case::speed("speed")]
    #[case::thinking_estimate("thinking_estimate")]
    #[case::thinking_tool("thinking_tool")]
    #[case::two_thinking("two_thinking")]
    #[case::server_tool_usage("server_tool_usage")]
    #[case::all_usage_fields("all_usage_fields")]
    #[case::stop_details_other("stop_details_other")]
    #[case::input_transformations("input_transformations")]
    #[case::context_management("context_management")]
    #[case::requested_speed("requested_speed")]
    #[case::cumulative_delta_usage("cumulative_delta_usage")]
    #[case::unicode_text("unicode_text")]
    #[case::tool_no_input("tool_no_input")]
    #[case::tool_caller("tool_caller")]
    #[case::two_citations("two_citations")]
    #[case::null_stop("null_stop")]
    #[case::web_search("web_search")]
    #[case::web_fetch("web_fetch")]
    #[case::web_search_error("web_search_error")]
    #[case::code_execution("code_execution")]
    #[case::code_execution_stderr("code_execution_stderr")]
    #[case::tool_search("tool_search")]
    #[case::tool_search_counted("tool_search_counted")]
    #[case::compaction("compaction")]
    #[case::two_compactions("two_compactions")]
    #[case::iterations("iterations")]
    #[case::iterations_thinking("iterations_thinking")]
    #[case::iterations_partial_thinking("iterations_partial_thinking")]
    fn a_complete_answer_logs_what_python_logs(#[case] name: &str) {
        let fixture = fixture(name);
        let count_tokens = counter(&fixture);
        let logged =
            complete_response(&fixture.body, &context(&fixture, &count_tokens)).expect("loggable");
        assert_eq!(
            serde_json::to_string(&logged).expect("serializable"),
            fixture.complete.to_string()
        );
    }

    #[rstest]
    fn an_answer_without_an_id_logs_the_fallback() {
        let fixture = fixture("text");
        let mut body = fixture.body.clone();
        body.remove("id");
        let count_tokens = counter(&fixture);
        let logged = complete_response(&body, &context(&fixture, &count_tokens)).expect("loggable");
        assert_eq!(logged.id, "chatcmpl-fallback");
    }

    #[rstest]
    #[case::an_error_body(json!({"type": "error", "error": {"type": "overloaded_error"}}))]
    #[case::no_usage(json!({"id": "msg_1", "model": "claude", "content": [], "stop_reason": "end_turn"}))]
    #[case::an_untyped_block(json!({"id": "msg_1", "model": "claude", "content": [{"text": "Hi"}], "stop_reason": "end_turn", "usage": {}}))]
    fn a_body_python_cannot_transform_is_refused(#[case] body: Value) {
        let fixture = fixture("text");
        let count_tokens = counter(&fixture);
        let body = body.as_object().cloned().unwrap_or_default();
        assert!(complete_response(&body, &context(&fixture, &count_tokens)).is_err());
    }
}
