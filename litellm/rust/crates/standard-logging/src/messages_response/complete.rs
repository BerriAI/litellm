//! A complete Messages answer as Python logs it: `AnthropicConfig.transform_response`, then
//! `transform_parsed_response`, called by `_handle_anthropic_messages_response_logging` on the
//! provider's response body.

use litellm_python_compat::json::{dumps, from_json};
use serde::Deserialize;
use serde_json::{Map, Value};

use super::{
    ASSISTANT, CHAT_COMPLETION, CompleteMessage, CompleteToolCall, Error, FUNCTION, LoggedChoice,
    LoggedMessage, LoggedResponse, LoggedUsage, Null, ResponseContext, ToolFunction, finish_reason,
    refusal, stop_reason_fields,
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

/// What `extract_response_content` reads off the content blocks.
#[derive(Default)]
struct Content {
    text: String,
    citations: Option<Vec<Value>>,
    thinking_blocks: Option<Vec<Map<String, Value>>>,
    tool_calls: Vec<CompleteToolCall>,
}

/// A block whose logged fields come from code the Rust builder does not port: hosted tool calls
/// and their results (`web_search_calls`, `code_interpreter_results`, `server_tool_calls`) and
/// compaction.
fn unported(kind: &str) -> bool {
    kind == "server_tool_use" || kind.ends_with("_tool_result") || kind == "compaction"
}

fn content(blocks: &[Map<String, Value>]) -> Result<Content, Error> {
    let mut content = Content::default();
    for (index, block) in blocks.iter().enumerate() {
        let kind = block
            .get(TYPE)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::UnportedBlock("untyped".to_owned()))?;
        match kind {
            _ if unported(kind) => return Err(Error::UnportedBlock(kind.to_owned())),
            TEXT => {
                if let Some(text) = block.get(TEXT).and_then(Value::as_str) {
                    content.text.push_str(text);
                }
            }
            "tool_use" => content.tool_calls.push(tool_call(block, index)?),
            THINKING | "redacted_thinking" => content
                .thinking_blocks
                .get_or_insert_with(Vec::new)
                .push(block.clone()),
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

/// `convert_tool_use_to_openai_format`: the input as `json.dumps` writes it, and the caller
/// of a programmatic tool call when the block names one.
fn tool_call(block: &Map<String, Value>, index: usize) -> Result<CompleteToolCall, Error> {
    let ToolUse { id, name, input } =
        ToolUse::deserialize(Value::Object(block.clone())).map_err(Error::Unreadable)?;
    Ok(CompleteToolCall {
        index,
        caller: block.get("caller").cloned(),
        function: ToolFunction {
            arguments: dumps(&from_json(input)).map_err(Error::ToolInput)?,
            name,
        },
        id,
        kind: FUNCTION,
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
    } = content(&body.content)?;
    let reasoning_content = thinking_blocks.as_deref().map(reasoning_content);
    let usage = calculate(
        ReportedUsage::read(&body.usage)?,
        Reasoning {
            content: reasoning_content.as_deref(),
            has_thinking_block: thinking_blocks.is_some(),
        },
        context.requested_speed,
        context.count_tokens,
    );
    let thinking_value = thinking_blocks
        .as_ref()
        .map(|blocks| Value::Array(blocks.iter().cloned().map(Value::Object).collect()));
    let named_when_set = [
        ("context_management", body.context_management),
        ("input_transformations", body.input_transformations),
        ("container", body.container),
    ]
    .into_iter()
    .filter_map(|(key, value)| Some((key.to_owned(), value.filter(|v| !v.is_null())?)));
    let provider_specific_fields = [
        (
            CITATIONS.to_owned(),
            citations.map_or(Value::Null, Value::Array),
        ),
        (
            THINKING_BLOCKS.to_owned(),
            thinking_value.unwrap_or(Value::Null),
        ),
    ]
    .into_iter()
    .chain(named_when_set)
    .chain(stop_reason_fields(
        body.stop_reason.as_ref(),
        body.stop_details.as_ref(),
    ))
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
    #[case::a_hosted_tool_call(json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}), "server_tool_use")]
    #[case::a_hosted_tool_result(json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}), "web_search_tool_result")]
    #[case::compaction(json!({"type": "compaction", "content": "summary"}), "compaction")]
    fn a_block_the_rust_builder_does_not_port_is_refused(#[case] block: Value, #[case] kind: &str) {
        let fixture = fixture("text");
        let mut body = fixture.body.clone();
        body.insert("content".to_owned(), json!([block]));
        let count_tokens = counter(&fixture);
        let refused = complete_response(&body, &context(&fixture, &count_tokens));
        assert!(matches!(refused, Err(Error::UnportedBlock(found)) if found == kind));
    }

    #[rstest]
    #[case::an_error_body(json!({"type": "error", "error": {"type": "overloaded_error"}}))]
    #[case::no_usage(json!({"id": "msg_1", "model": "claude", "content": [], "stop_reason": "end_turn"}))]
    fn a_body_python_cannot_transform_is_refused(#[case] body: Value) {
        let fixture = fixture("text");
        let count_tokens = counter(&fixture);
        let body = body.as_object().cloned().unwrap_or_default();
        assert!(complete_response(&body, &context(&fixture, &count_tokens)).is_err());
    }
}
