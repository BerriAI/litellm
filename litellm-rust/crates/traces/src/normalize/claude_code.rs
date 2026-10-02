use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr, first, tokens};
use crate::{Error, otlp::DecodedEvent};

pub(crate) const CLAUDE_CODE_SCOPE: &str = "com.anthropic.claude_code.tracing";
pub(crate) const CLAUDE_CODE_AGENT: &str = "claude-code";
const AGENT_SDK_FRAMEWORK: &str = "claude-agent-sdk";

pub(super) struct ClaudeCodeNormalizer;

enum SpanType {
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
        "interaction" => SpanType::Interaction,
        "llm_request" => SpanType::LlmRequest,
        "tool" => SpanType::Tool,
        _ => SpanType::Other,
    }
}

fn framework(attributes: &BTreeMap<String, String>) -> &'static str {
    if attr(attributes, "query_source_safe") == "sdk"
        || attr(attributes, "system_prompt_preview").contains("cc_entrypoint=sdk")
    {
        AGENT_SDK_FRAMEWORK
    } else {
        CLAUDE_CODE_AGENT
    }
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

fn input_tokens(attributes: &BTreeMap<String, String>) -> Result<u32, Error> {
    ["input_tokens", "cache_read_tokens", "cache_creation_tokens"]
        .into_iter()
        .try_fold(0u32, |total, key| {
            total
                .checked_add(tokens(attributes, key)?)
                .ok_or(Error::TokenCountOutOfRange)
        })
}

impl SpanNormalizer for ClaudeCodeNormalizer {
    fn matches(&self, scope_name: &str, _attributes: &BTreeMap<String, String>) -> bool {
        scope_name == CLAUDE_CODE_SCOPE
    }

    fn consumed_attributes(&self, attributes: &BTreeMap<String, String>) -> [&'static str; 2] {
        match span_type("", attributes) {
            SpanType::Interaction => ["user_prompt", ""],
            SpanType::LlmRequest => ["new_context", "response.model_output"],
            SpanType::Tool if tool_arguments(attributes).is_some() => ["tool_input", ""],
            SpanType::Tool | SpanType::Other => ["", ""],
        }
    }

    fn display_name(&self, attributes: &BTreeMap<String, String>) -> Option<String> {
        let tool_name = attr(attributes, "tool_name");
        (matches!(span_type("", attributes), SpanType::Tool) && !tool_name.is_empty())
            .then(|| tool_name.to_owned())
    }

    fn normalize(
        &self,
        name: &str,
        _parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
        events: &[DecodedEvent],
    ) -> Result<NormalizedSpan, Error> {
        let base = NormalizedSpan {
            observation_type: ObservationType::Framework,
            agent_name: CLAUDE_CODE_AGENT.to_owned(),
            framework: framework(attributes).to_owned(),
            litellm_request_id: String::new(),
            model: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            input: String::new(),
            output: String::new(),
        };
        Ok(match span_type(name, attributes) {
            SpanType::Interaction => NormalizedSpan {
                observation_type: ObservationType::Agent,
                input: user_prompt(attributes),
                ..base
            },
            SpanType::LlmRequest => NormalizedSpan {
                observation_type: ObservationType::Llm,
                litellm_request_id: first(attributes, "gen_ai.response.id", "request_id")
                    .to_owned(),
                model: first(attributes, "model", "gen_ai.request.model").to_owned(),
                input_tokens: input_tokens(attributes)?,
                output_tokens: tokens(attributes, "output_tokens")?,
                input: llm_input(attributes),
                output: llm_output(attributes),
                ..base
            },
            SpanType::Tool => NormalizedSpan {
                observation_type: ObservationType::Tool,
                input: tool_input(attributes),
                output: tool_output(attributes, events),
                ..base
            },
            SpanType::Other => base,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::Value;

    use super::{CLAUDE_CODE_SCOPE, ClaudeCodeNormalizer, SpanNormalizer};
    use crate::{Error, normalize::ObservationType, otlp::DecodedEvent};

    fn attributes(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    fn tool_without_detailed_input_lists_known_arguments() {
        let span = ClaudeCodeNormalizer
            .normalize(
                "claude_code.tool",
                "parent",
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
        let span = ClaudeCodeNormalizer
            .normalize("claude_code.tool", "parent", &attrs, &[])
            .expect("valid span");
        let input: Value = serde_json::from_str(&span.input).expect("argument object");
        assert_eq!(input["file_path"], "/workspace/a.py");
        assert!(
            !ClaudeCodeNormalizer
                .consumed_attributes(&attrs)
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
        let span = ClaudeCodeNormalizer
            .normalize(
                "claude_code.tool",
                "parent",
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
        let span = ClaudeCodeNormalizer
            .normalize(
                "claude_code.llm_request",
                "parent",
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
        assert_eq!(span.framework, "claude-code");
    }

    #[rstest]
    fn llm_token_sum_overflow_is_rejected() {
        let result = ClaudeCodeNormalizer.normalize(
            "claude_code.llm_request",
            "parent",
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
        let span = ClaudeCodeNormalizer
            .normalize(name, "parent", &attrs, &[])
            .expect("valid span");
        assert_eq!(span.observation_type, expected);
        assert!(ClaudeCodeNormalizer.matches(CLAUDE_CODE_SCOPE, &attrs));
    }
}
