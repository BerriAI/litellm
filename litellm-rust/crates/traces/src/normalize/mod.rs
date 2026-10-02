use std::collections::BTreeMap;

use crate::{DecodeError, otlp::DecodedEvent};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ObservationType {
    Agent,
    Llm,
    Tool,
    Chain,
    Framework,
}

#[derive(Debug, Serialize)]
pub struct NormalizedSpan {
    pub observation_type: ObservationType,
    pub agent_name: String,
    pub framework: String,
    pub litellm_request_id: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub input: String,
    pub output: String,
}

pub(crate) struct Normalization {
    pub span: NormalizedSpan,
    pub display_name: Option<String>,
    pub consumed_attributes: [&'static str; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NormalizedFieldDefinition {
    pub name: &'static str,
    pub clickhouse_column: &'static str,
    pub clickhouse_type: &'static str,
    pub meaning: &'static str,
}

pub const NORMALIZED_FIELD_DEFINITIONS: [NormalizedFieldDefinition; 9] = [
    NormalizedFieldDefinition {
        name: "observation_type",
        clickhouse_column: "ObservationType",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Agent, LLM, tool, chain, or framework span",
    },
    NormalizedFieldDefinition {
        name: "agent_name",
        clickhouse_column: "AgentName",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Agent associated with this span",
    },
    NormalizedFieldDefinition {
        name: "framework",
        clickhouse_column: "Framework",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Agent framework or SDK that emitted this span, e.g. claude-agent-sdk",
    },
    NormalizedFieldDefinition {
        name: "litellm_request_id",
        clickhouse_column: "LiteLLMRequestId",
        clickhouse_type: "String",
        meaning: "LiteLLM response ID used to link a span to a spend log",
    },
    NormalizedFieldDefinition {
        name: "model",
        clickhouse_column: "Model",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Model used by this span",
    },
    NormalizedFieldDefinition {
        name: "input_tokens",
        clickhouse_column: "InputTokens",
        clickhouse_type: "UInt32",
        meaning: "Input token count",
    },
    NormalizedFieldDefinition {
        name: "output_tokens",
        clickhouse_column: "OutputTokens",
        clickhouse_type: "UInt32",
        meaning: "Output token count",
    },
    NormalizedFieldDefinition {
        name: "input",
        clickhouse_column: "Input",
        clickhouse_type: "String",
        meaning: "Normalized input payload",
    },
    NormalizedFieldDefinition {
        name: "output",
        clickhouse_column: "Output",
        clickhouse_type: "String",
        meaning: "Normalized output payload",
    },
];

trait SpanNormalizer {
    fn matches(&self, scope_name: &str, attributes: &BTreeMap<String, String>) -> bool;
    fn consumed_attributes(&self, attributes: &BTreeMap<String, String>) -> [&'static str; 2];
    fn normalize(
        &self,
        name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
        events: &[DecodedEvent],
    ) -> Result<NormalizedSpan, DecodeError>;
    fn display_name(&self, _attributes: &BTreeMap<String, String>) -> Option<String> {
        None
    }
}

mod claude_code;
mod genai;
mod langsmith;
mod openinference;

use claude_code::ClaudeCodeNormalizer;
pub(crate) use claude_code::{CLAUDE_CODE_AGENT, CLAUDE_CODE_SCOPE};
use genai::GenAiNormalizer;
use langsmith::LangSmithNormalizer;
use openinference::OpenInferenceNormalizer;

fn attr<'a>(attributes: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    attributes.get(key).map(String::as_str).unwrap_or_default()
}

fn first<'a>(attributes: &'a BTreeMap<String, String>, left: &str, right: &str) -> &'a str {
    let value = attr(attributes, left);
    if value.is_empty() {
        attr(attributes, right)
    } else {
        value
    }
}

fn tokens(attributes: &BTreeMap<String, String>, key: &str) -> Result<u32, DecodeError> {
    let value = attr(attributes, key).trim();
    if value.is_empty() {
        return Ok(0);
    }
    match value.parse::<i128>() {
        Ok(number) if (0..=u32::MAX as i128).contains(&number) => Ok(number as u32),
        Ok(_) => Err(DecodeError::TokenCountOutOfRange),
        Err(_)
            if value
                .trim_start_matches(['+', '-'])
                .bytes()
                .all(|byte| byte.is_ascii_digit()) =>
        {
            Err(DecodeError::TokenCountOutOfRange)
        }
        Err(_) => Ok(0),
    }
}

fn usage_tokens(attributes: &BTreeMap<String, String>) -> Result<(u32, u32), DecodeError> {
    Ok((
        tokens(attributes, "gen_ai.usage.input_tokens")?,
        tokens(attributes, "gen_ai.usage.output_tokens")?,
    ))
}

#[derive(Default, Deserialize)]
struct AgentMetadata {
    #[serde(default)]
    lc_agent_name: String,
    #[serde(default)]
    ls_integration: String,
}

fn recorded_agent_name(
    name: &str,
    attributes: &BTreeMap<String, String>,
    span: &NormalizedSpan,
) -> String {
    let explicit = [
        span.agent_name.as_str(),
        attr(attributes, "gen_ai.agent.name"),
        attr(attributes, "agent.name"),
        attr(attributes, "openclaw.agent"),
    ]
    .into_iter()
    .find(|value| !value.is_empty());
    if let Some(value) = explicit {
        return value.to_owned();
    }
    let metadata =
        serde_json::from_str::<AgentMetadata>(attr(attributes, "metadata")).unwrap_or_default();
    if !metadata.lc_agent_name.is_empty() {
        return metadata.lc_agent_name;
    }
    if span.observation_type == ObservationType::Agent {
        let node = attr(attributes, "graph.node.id");
        if !node.is_empty() {
            return node.to_owned();
        }
        if metadata.ls_integration == "langgraph" && name != "LangGraph" && !is_middleware(name) {
            return name.to_owned();
        }
    }
    String::new()
}

fn is_middleware(name: &str) -> bool {
    [
        ".wrap_model_call",
        ".wrap_tool_call",
        ".before_agent",
        ".after_agent",
        ".before_model",
        ".after_model",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

pub fn normalize(
    scope_name: &str,
    name: &str,
    parent_span_id: &str,
    attributes: &BTreeMap<String, String>,
    events: &[DecodedEvent],
) -> Result<Normalization, DecodeError> {
    let normalizers: [&dyn SpanNormalizer; 4] = [
        &ClaudeCodeNormalizer,
        &LangSmithNormalizer,
        &OpenInferenceNormalizer,
        &GenAiNormalizer,
    ];
    let normalizer = normalizers
        .into_iter()
        .find(|normalizer| normalizer.matches(scope_name, attributes))
        .expect("GenAI fallback always matches");
    let span = normalizer.normalize(name, parent_span_id, attributes, events)?;
    let agent_name = recorded_agent_name(name, attributes, &span);
    let observation_type = if !parent_span_id.is_empty()
        && scope_name == "openinference.instrumentation.langchain"
        && is_middleware(name)
    {
        ObservationType::Framework
    } else {
        span.observation_type
    };
    Ok(Normalization {
        span: NormalizedSpan {
            agent_name,
            observation_type,
            ..span
        },
        display_name: normalizer.display_name(attributes),
        consumed_attributes: normalizer.consumed_attributes(attributes),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use rstest::rstest;

    use super::{NORMALIZED_FIELD_DEFINITIONS, ObservationType, normalize};

    #[rstest]
    #[case::langsmith("langsmith", [("langsmith.span.kind", "llm"), ("openinference.span.kind", "TOOL")], ObservationType::Llm)]
    #[case::openinference("other", [("openinference.span.kind", "LLM"), ("gen_ai.operation.name", "execute_tool")], ObservationType::Llm)]
    #[case::genai("other", [("gen_ai.operation.name", "execute_tool"), ("gen_ai.usage.input_tokens", "7")], ObservationType::Tool)]
    #[case::claude_code("com.anthropic.claude_code.tracing", [("span.type", "llm_request"), ("openinference.span.kind", "TOOL")], ObservationType::Llm)]
    fn convention_dispatch_preserves_precedence(
        #[case] scope: &str,
        #[case] attributes: [(&str, &str); 2],
        #[case] expected: ObservationType,
    ) {
        let attributes = attributes
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        let fields = normalize(scope, "step", "parent", &attributes, &[])
            .expect("valid tokens")
            .span;
        assert_eq!(fields.observation_type, expected);
        if expected == ObservationType::Tool {
            assert_eq!(fields.input_tokens, 7);
        }
    }

    #[rstest]
    fn field_definitions_match_serialized_normalized_span() {
        let fields = normalize("", "root", "", &BTreeMap::new(), &[])
            .expect("valid tokens")
            .span;
        let serialized = serde_json::to_value(fields).expect("serializable fields");
        let keys: BTreeSet<_> = serialized
            .as_object()
            .expect("field object")
            .keys()
            .map(String::as_str)
            .collect();
        let mapped: BTreeSet<_> = NORMALIZED_FIELD_DEFINITIONS
            .iter()
            .map(|field| field.name)
            .collect();
        assert_eq!(keys, mapped);
    }

    #[rstest]
    fn token_counts_accept_surrounding_whitespace() {
        let attributes =
            BTreeMap::from([("gen_ai.usage.input_tokens".to_owned(), " 7 ".to_owned())]);
        let fields = normalize("", "root", "", &attributes, &[])
            .expect("valid tokens")
            .span;
        assert_eq!(fields.input_tokens, 7);
    }

    #[rstest]
    #[case::negative("-1")]
    #[case::overflow("4294967296")]
    fn token_counts_outside_storage_range_are_rejected(#[case] value: &str) {
        let attributes =
            BTreeMap::from([("gen_ai.usage.input_tokens".to_owned(), value.to_owned())]);
        assert!(normalize("", "root", "", &attributes, &[]).is_err());
    }
}
