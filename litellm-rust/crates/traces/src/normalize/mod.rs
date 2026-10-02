use std::collections::BTreeMap;

use crate::DecodeError;
use serde::Serialize;

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
    pub litellm_request_id: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub input: String,
    pub output: String,
}

pub(crate) struct Normalization {
    pub span: NormalizedSpan,
    pub consumed_attributes: [&'static str; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NormalizedFieldDefinition {
    pub name: &'static str,
    pub clickhouse_column: &'static str,
    pub clickhouse_type: &'static str,
    pub meaning: &'static str,
}

pub const NORMALIZED_FIELD_DEFINITIONS: [NormalizedFieldDefinition; 8] = [
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
    ) -> Result<NormalizedSpan, DecodeError>;
}

mod genai;
mod langsmith;
mod openinference;

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

pub fn normalize(
    scope_name: &str,
    name: &str,
    parent_span_id: &str,
    attributes: &BTreeMap<String, String>,
) -> Result<Normalization, DecodeError> {
    let normalizers: [&dyn SpanNormalizer; 3] = [
        &LangSmithNormalizer,
        &OpenInferenceNormalizer,
        &GenAiNormalizer,
    ];
    let normalizer = normalizers
        .into_iter()
        .find(|normalizer| normalizer.matches(scope_name, attributes))
        .expect("GenAI fallback always matches");
    Ok(Normalization {
        span: normalizer.normalize(name, parent_span_id, attributes)?,
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
    fn convention_dispatch_preserves_precedence(
        #[case] scope: &str,
        #[case] attributes: [(&str, &str); 2],
        #[case] expected: ObservationType,
    ) {
        let attributes = attributes
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        let fields = normalize(scope, "step", "parent", &attributes)
            .expect("valid tokens")
            .span;
        assert_eq!(fields.observation_type, expected);
        if expected == ObservationType::Tool {
            assert_eq!(fields.input_tokens, 7);
        }
    }

    #[rstest]
    fn field_definitions_match_serialized_normalized_span() {
        let fields = normalize("", "root", "", &BTreeMap::new())
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
        let fields = normalize("", "root", "", &attributes)
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
        assert!(normalize("", "root", "", &attributes).is_err());
    }
}
