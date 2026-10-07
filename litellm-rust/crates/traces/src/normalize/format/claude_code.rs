use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{Extraction, Format, SpanFacts};
use crate::{
    Error,
    normalize::{
        CLAUDE_CODE_AGENT, CLAUDE_CODE_EVENTS_SCOPE, CLAUDE_CODE_SCOPE, CallEvidence,
        ObservationType, RoleEvidence, SpanContext, attr, present, tokens,
    },
    otlp::DecodedEvent,
};

/// Claude Code's built-in tracing, identified by its instrumentation scope.
pub(crate) struct ClaudeCode;

enum SpanType {
    AssistantResponse,
    ToolResult,
    ApiRequestBody,
    Compaction,
    Interaction,
    LlmRequest,
    Tool,
    Other,
}

fn span_type(name: &str, attributes: &BTreeMap<String, String>) -> SpanType {
    let kind = attr(attributes, "span.type");
    let kind = if kind.is_empty() {
        name.strip_prefix("claude_code.").unwrap_or(name)
    } else {
        kind
    };
    match kind {
        "assistant_response" => SpanType::AssistantResponse,
        "tool_result" => SpanType::ToolResult,
        "api_request_body" => SpanType::ApiRequestBody,
        "compaction" => SpanType::Compaction,
        "interaction" => SpanType::Interaction,
        "llm_request" => SpanType::LlmRequest,
        "tool" => SpanType::Tool,
        _ => SpanType::Other,
    }
}

/// `agent:custom:search_agent` -> `search_agent`: the subagent a request ran for.
fn subagent(attributes: &BTreeMap<String, String>) -> Option<&str> {
    let mut parts = attr(attributes, "query_source")
        .strip_prefix("agent:")?
        .splitn(2, ':');
    let (_kind, name) = (parts.next()?, parts.next()?);
    (!name.is_empty()).then_some(name)
}

fn split_header(text: &str) -> Option<(&str, &str)> {
    let (header, body) = text.strip_prefix('[')?.split_once("]\n")?;
    Some((header, body))
}

fn without_header<'a>(text: &'a str, prefix: &str) -> &'a str {
    split_header(text)
        .filter(|(header, _)| header.starts_with(prefix))
        .map_or(text, |(_, body)| body)
}

fn tool_arguments(attributes: &BTreeMap<String, String>) -> Option<&str> {
    let arguments = without_header(attr(attributes, "tool_input"), "TOOL INPUT");
    serde_json::from_str::<Map<String, Value>>(arguments)
        .is_ok()
        .then_some(arguments)
}

fn tool_input(attributes: &BTreeMap<String, String>) -> String {
    if let Some(arguments) = tool_arguments(attributes) {
        return arguments.to_owned();
    }
    let fields: Map<String, Value> = [
        ("command", "full_command"),
        ("file_path", "file_path"),
        ("bash_argv0", "bash_argv0"),
    ]
    .into_iter()
    .filter_map(|(key, source)| {
        let value = attr(attributes, source);
        (!value.is_empty()).then(|| (key.to_owned(), Value::String(value.to_owned())))
    })
    .collect();
    if fields.is_empty() {
        String::new()
    } else {
        Value::Object(fields).to_string()
    }
}

fn tool_output(attributes: &BTreeMap<String, String>, events: &[DecodedEvent]) -> String {
    events
        .iter()
        .filter(|event| event.name == "tool.output")
        .flat_map(|event| {
            ["output", "content", "diff"]
                .into_iter()
                .map(|key| attr(&event.attributes, key))
        })
        .find(|value| !value.is_empty())
        .unwrap_or_else(|| without_header(attr(attributes, "new_context"), "TOOL RESULT"))
        .to_owned()
}

fn context_message(context: &str) -> Value {
    let (role, content) = match split_header(context) {
        Some(("USER" | "USER PROMPT", body)) => ("user", body),
        Some(("ASSISTANT", body)) => ("assistant", body),
        Some((header, body)) if header.starts_with("TOOL RESULT") => ("tool", body),
        _ => ("user", context),
    };
    json!({"role": role, "content": content})
}

fn user_prompt(attributes: &BTreeMap<String, String>) -> String {
    let prompt = attr(attributes, "user_prompt");
    if prompt.is_empty() {
        String::new()
    } else {
        json!([{"role": "user", "content": prompt}]).to_string()
    }
}

fn llm_input(attributes: &BTreeMap<String, String>) -> String {
    let messages: Vec<Value> = [
        Some(attr(attributes, "system_prompt_preview"))
            .filter(|system| !system.is_empty())
            .map(|system| json!({"role": "system", "content": system})),
        Some(attr(attributes, "new_context"))
            .filter(|context| !context.is_empty())
            .map(context_message),
    ]
    .into_iter()
    .flatten()
    .collect();
    if messages.is_empty() {
        String::new()
    } else {
        Value::Array(messages).to_string()
    }
}

fn llm_output(attributes: &BTreeMap<String, String>) -> String {
    let output = attr(attributes, "response.model_output");
    if output.is_empty() {
        String::new()
    } else {
        json!({"role": "assistant", "content": output}).to_string()
    }
}

fn exported_tool_results(attributes: &BTreeMap<String, String>) -> String {
    let Ok(body) = serde_json::from_str::<Value>(attr(attributes, "body")) else {
        return json!({"warning": "Claude's API body export is missing or truncated. Some tool results may be unavailable."}).to_string();
    };
    let message = body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .rev()
                .find(|message| message.get("role").and_then(Value::as_str) != Some("system"))
        })
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"));
    let Some(content) = message.and_then(|message| message.get("content")) else {
        return json!({"warning": "Claude's API body export has an unexpected message shape. Some tool results may be unavailable."}).to_string();
    };
    if !content.is_array() && !content.is_string() {
        return json!({"warning": "Claude's API body export has an unexpected content shape. Some tool results may be unavailable."}).to_string();
    }
    let results: Vec<Value> = content.as_array()
        .into_iter()
        .flatten()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
        .map(|block| {
            let content = match block.get("content") {
                Some(Value::String(text)) => text.clone(),
                Some(Value::Array(blocks)) => blocks.iter().map(|block| {
                    block.get("text").and_then(Value::as_str).unwrap_or("[Non-text tool output omitted by Claude export]")
                }).collect::<Vec<_>>().join("\n"),
                _ => String::new(),
            };
            json!({"id": block.get("tool_use_id"), "content": content, "is_error": block.get("is_error").and_then(Value::as_bool).unwrap_or(false)})
        }).collect();
    json!({"tool_results": results}).to_string()
}

fn input_tokens(attributes: &BTreeMap<String, String>) -> Result<u32, Error> {
    ["input_tokens", "cache_read_tokens", "cache_creation_tokens"]
        .into_iter()
        .try_fold(0u32, |total, key| {
            total
                .checked_add(tokens(attributes, key)?)
                .ok_or(Error::TokenCountOutOfRange)
        })
}

impl Format for ClaudeCode {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        matches!(context.scope, CLAUDE_CODE_SCOPE | CLAUDE_CODE_EVENTS_SCOPE)
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let attributes = context.attributes;
        let kind = span_type(context.name, attributes);
        let base = SpanFacts {
            role: Some(RoleEvidence::Declared(ObservationType::Framework)),
            agent_name: Some(CLAUDE_CODE_AGENT.to_owned()),
            tool_call_id: present(attributes, &["gen_ai.tool.call.id"]),
            ..SpanFacts::default()
        };
        let (facts, consumed): (SpanFacts, Vec<&'static str>) = match kind {
            SpanType::AssistantResponse => (
                SpanFacts {
                    role: Some(RoleEvidence::Declared(ObservationType::Chain)),
                    agent_name: Some(subagent(attributes).unwrap_or(CLAUDE_CODE_AGENT).to_owned()),
                    model: present(attributes, &["model"]),
                    output: json!({"role": "assistant", "content": attr(attributes, "response")})
                        .to_string(),
                    ..base
                },
                vec!["response"],
            ),
            SpanType::ToolResult => (
                SpanFacts {
                    input: tool_input(attributes),
                    tool_call_id: present(attributes, &["tool_use_id"]),
                    ..base
                },
                if tool_arguments(attributes).is_some() {
                    vec!["tool_input"]
                } else {
                    Vec::new()
                },
            ),
            SpanType::Compaction => (
                SpanFacts {
                    role: Some(RoleEvidence::Declared(ObservationType::Chain)),
                    output: json!({"role": "system", "content": if attr(attributes, "success") == "true" {
                        "Context compacted"
                    } else {
                        "Context compaction failed"
                    }}).to_string(),
                    ..base
                },
                Vec::new(),
            ),
            SpanType::ApiRequestBody => (
                SpanFacts {
                    output: exported_tool_results(attributes),
                    ..base
                },
                vec!["body"],
            ),
            SpanType::Interaction => (
                SpanFacts {
                    role: Some(RoleEvidence::Declared(ObservationType::Agent)),
                    input: user_prompt(attributes),
                    ..base
                },
                vec!["user_prompt"],
            ),
            SpanType::LlmRequest => (
                SpanFacts {
                    role: Some(RoleEvidence::Declared(ObservationType::Llm)),
                    agent_name: Some(subagent(attributes).unwrap_or(CLAUDE_CODE_AGENT).to_owned()),
                    model: present(attributes, &["model", "gen_ai.request.model"]),
                    input_tokens: input_tokens(attributes)?,
                    output_tokens: tokens(attributes, "output_tokens")?,
                    input: llm_input(attributes),
                    output: llm_output(attributes),
                    calls: present(attributes, &["gen_ai.response.id", "request_id"])
                        .map_or(CallEvidence::Unknown, |id| {
                            CallEvidence::complete(crate::normalize::claude_call_key(id))
                        }),
                    ..base
                },
                vec!["new_context", "response.model_output"],
            ),
            SpanType::Tool => (
                SpanFacts {
                    role: Some(RoleEvidence::Declared(ObservationType::Tool)),
                    input: tool_input(attributes),
                    output: tool_output(attributes, context.events),
                    ..base
                },
                if tool_arguments(attributes).is_some() {
                    vec!["tool_input"]
                } else {
                    Vec::new()
                },
            ),
            SpanType::Other => (base, Vec::new()),
        };
        Ok(Extraction {
            facts,
            display_name: if matches!(kind, SpanType::Tool) {
                present(attributes, &["tool_name"])
            } else {
                None
            },
            consumed_attributes: consumed,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::Value;

    use super::CLAUDE_CODE_SCOPE;
    use crate::{
        Error,
        normalize::{Normalization, NormalizedSpan, ObservationType},
        otlp::DecodedEvent,
    };

    fn normalization(
        name: &str,
        attributes: &BTreeMap<String, String>,
        events: &[DecodedEvent],
    ) -> Result<Normalization, Error> {
        crate::normalize::normalize(&crate::normalize::SpanContext {
            scope: CLAUDE_CODE_SCOPE,
            name,
            parent_span_id: "parent",
            attributes,
            events,
            resource_attributes: &BTreeMap::new(),
        })
    }

    fn normalize(
        name: &str,
        attributes: &BTreeMap<String, String>,
        events: &[DecodedEvent],
    ) -> Result<NormalizedSpan, Error> {
        normalization(name, attributes, events).map(|normalization| normalization.span)
    }

    fn attributes(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    fn notification_prompts_keep_user_provenance_and_compaction_is_system() {
        let prompt_text =
            "<task-notification><summary>Agent Reader completed</summary></task-notification>";
        let notification = normalize(
            "claude_code.interaction",
            &attributes(&[("user_prompt", prompt_text)]),
            &[],
        )
        .unwrap();
        let prompt: Value = serde_json::from_str(&notification.input).unwrap();
        assert_eq!(
            prompt[0],
            serde_json::json!({"role":"user","content":prompt_text})
        );
        let compaction = normalize(
            "claude_code.compaction",
            &attributes(&[("success", "true")]),
            &[],
        )
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&compaction.output).unwrap(),
            serde_json::json!({"role":"system","content":"Context compacted"})
        );
    }

    #[rstest]
    fn tool_without_detailed_input_lists_known_arguments() {
        let span = normalize(
            "claude_code.tool",
            &attributes(&[
                ("span.type", "tool"),
                ("tool_name", "Bash"),
                ("full_command", "git status"),
                ("bash_argv0", "git"),
            ]),
            &[],
        )
        .expect("valid span");
        let input: Value = serde_json::from_str(&span.input).expect("argument object");
        assert_eq!(input["command"], "git status");
        assert_eq!(input["bash_argv0"], "git");
        assert!(input.get("file_path").is_none());
        assert!(input.get("role").is_none());
    }

    #[rstest]
    fn malformed_tool_input_falls_back_and_stays_in_attributes() {
        let attrs = attributes(&[
            ("span.type", "tool"),
            ("tool_input", "[TOOL INPUT: Read]\nnot json"),
            ("file_path", "/workspace/a.py"),
        ]);
        let span = normalize("claude_code.tool", &attrs, &[]).expect("valid span");
        let input: Value = serde_json::from_str(&span.input).expect("argument object");
        assert_eq!(input["file_path"], "/workspace/a.py");
        assert!(
            !normalization("claude_code.tool", &attrs, &[])
                .expect("valid span")
                .consumed_attributes
                .contains(&"tool_input")
        );
    }

    #[rstest]
    #[case::event_output(
        vec![DecodedEvent { name: "tool.output".to_owned(), attributes: attributes(&[("output", "stdout text")]) }],
        "stdout text"
    )]
    #[case::event_diff(
        vec![DecodedEvent { name: "tool.output".to_owned(), attributes: attributes(&[("diff", "+line")]) }],
        "+line"
    )]
    #[case::other_event_ignored(
        vec![DecodedEvent { name: "other".to_owned(), attributes: attributes(&[("output", "nope")]) }],
        "{\"stdout\":\"ctx\"}"
    )]
    fn tool_output_prefers_event_then_context(
        #[case] events: Vec<DecodedEvent>,
        #[case] expected: &str,
    ) {
        let span = normalize(
            "claude_code.tool",
            &attributes(&[
                ("span.type", "tool"),
                ("new_context", "[TOOL RESULT: Bash]\n{\"stdout\":\"ctx\"}"),
            ]),
            &events,
        )
        .expect("valid span");
        assert_eq!(span.output, expected);
    }

    #[rstest]
    fn llm_tool_result_context_becomes_tool_message() {
        let span = normalize(
            "claude_code.llm_request",
            &attributes(&[
                ("span.type", "llm_request"),
                ("new_context", "[TOOL RESULT: toolu_1]\n1\timport os"),
            ]),
            &[],
        )
        .expect("valid span");
        let input: Value = serde_json::from_str(&span.input).expect("messages");
        assert_eq!(input[0]["role"], "tool");
        assert_eq!(input[0]["content"], "1\timport os");
        assert_eq!(span.output, "");
        assert_eq!(span.framework, Some(crate::Integration::ClaudeCode));
    }

    #[rstest]
    fn llm_token_sum_overflow_is_rejected() {
        let result = normalize(
            "claude_code.llm_request",
            &attributes(&[
                ("span.type", "llm_request"),
                ("input_tokens", "4294967295"),
                ("cache_read_tokens", "1"),
            ]),
            &[],
        );
        assert!(matches!(result, Err(Error::TokenCountOutOfRange)));
    }

    #[rstest]
    #[case::span_type_wins("claude_code.tool", "hook", ObservationType::Framework)]
    #[case::name_fallback("claude_code.interaction", "", ObservationType::Agent)]
    #[case::unknown("claude_code.something_new", "", ObservationType::Framework)]
    fn span_type_attribute_then_name_select_the_observation(
        #[case] name: &str,
        #[case] kind: &str,
        #[case] expected: ObservationType,
    ) {
        let attrs = if kind.is_empty() {
            BTreeMap::new()
        } else {
            attributes(&[("span.type", kind)])
        };
        let span = normalize(name, &attrs, &[]).expect("valid span");
        assert_eq!(span.observation_type, expected);
    }
}
