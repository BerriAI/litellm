//! Span normalization in two steps: a [`Convention`] extracts what a span records in its format,
//! then an [`Instrumentation`] interprets those facts with what is known about the SDK that emitted
//! it. Relationships between spans (wrappers, ownership, spend) are resolved later, over the whole
//! trace, because parents and children can arrive in separate exports.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use crate::{Error, otlp::DecodedEvent};
use serde::{Serialize, Serializer};

mod convention;
mod instrumentation;
mod messages;
mod metadata;

pub(crate) use convention::claude_code::{CLAUDE_CODE_AGENT, CLAUDE_CODE_SCOPE};
use instrumentation::Instrumentation;
pub(crate) use messages::{HIDDEN_BLOCK_TYPES, encode};
pub use metadata::{AgentMetadata, AgentType, Integration};

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
#[cfg_attr(feature = "schema", schemars(rename = "SpanType"))]
pub enum ObservationType {
    Agent,
    Llm,
    Tool,
    Chain,
    Framework,
    Retriever,
    Embedding,
    Reranker,
    Guardrail,
    Evaluator,
    Prompt,
    Decision,
}

/// A model request a span stands for, by the identifier its instrumentation recorded.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CallKey {
    /// LiteLLM's own id for the request (`spend_logs.request_id`).
    LiteLlmRequest(String),
    /// The provider response id returned to the caller (`spend_logs.response_id`).
    ProviderResponse(String),
    /// The span is the HTTP request itself; LiteLLM logs its `traceparent` span id.
    Transport,
}

impl fmt::Display for CallKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LiteLlmRequest(id) => write!(formatter, "litellm_request:{id}"),
            Self::ProviderResponse(id) => write!(formatter, "provider_response:{id}"),
            Self::Transport => formatter.write_str("transport:"),
        }
    }
}

impl Serialize for CallKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// Which model requests a span accounts for. `Complete` comes only from an instrumentation's known
/// contract (one chat span is one response), never from how many ids happened to be found.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum CallEvidence {
    #[default]
    Unknown,
    Partial(BTreeSet<CallKey>),
    Complete(BTreeSet<CallKey>),
}

impl CallEvidence {
    pub(crate) fn complete(key: CallKey) -> Self {
        Self::Complete(BTreeSet::from([key]))
    }

    /// The same evidence with one more key: an id named outside the convention adds to what the
    /// convention found, but says nothing about completeness.
    fn with(self, key: CallKey) -> Self {
        match self {
            Self::Unknown => Self::complete(key),
            Self::Partial(keys) => Self::Partial(keys.into_iter().chain([key]).collect()),
            Self::Complete(keys) => Self::Complete(keys.into_iter().chain([key]).collect()),
        }
    }

    fn key_set(&self) -> Option<&BTreeSet<CallKey>> {
        match self {
            Self::Unknown => None,
            Self::Partial(keys) | Self::Complete(keys) => Some(keys),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Partial(_) => "partial",
            Self::Complete(_) => "complete",
        }
    }
}

/// What a span says about its own role. A `WrapperCandidate` may only wrap the real operation
/// (a crew kickoff around its agents); the trace graph decides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoleEvidence {
    Unspecified,
    Declared(ObservationType),
    WrapperCandidate(ObservationType),
}

pub(crate) struct SpanContext<'a> {
    pub scope: &'a str,
    pub name: &'a str,
    pub parent_span_id: &'a str,
    pub attributes: &'a BTreeMap<String, String>,
    pub events: &'a [DecodedEvent],
}

#[derive(Debug, Serialize)]
pub struct NormalizedSpan {
    pub observation_type: ObservationType,
    pub wrapper_candidate: bool,
    pub agent_name: String,
    pub framework: String,
    pub agent_metadata: AgentMetadata,
    pub litellm_request_id: String,
    pub call_keys: Vec<CallKey>,
    pub call_evidence: &'static str,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub input: String,
    pub input_preview: String,
    pub output: String,
    pub tool_call_id: String,
}

pub(crate) struct Normalization {
    pub span: NormalizedSpan,
    pub display_name: Option<String>,
    pub consumed_attributes: Box<[&'static str]>,
}

pub fn normalize(
    scope_name: &str,
    name: &str,
    parent_span_id: &str,
    attributes: &BTreeMap<String, String>,
    events: &[DecodedEvent],
) -> Result<Normalization, Error> {
    let context = SpanContext {
        scope: scope_name,
        name,
        parent_span_id,
        attributes,
        events,
    };
    let extraction = convention::extract(&context)?;
    Ok(Instrumentation::detect(&context).interpret(
        &context,
        extraction,
        metadata::extract(&context),
    ))
}

/// An attribute's text together with the key it came from, so consumption follows extraction.
pub(crate) struct AttributeText<'a> {
    pub source: &'static str,
    pub text: &'a str,
}

/// The first of `keys` that is recorded and not empty.
fn select_attribute<'a>(
    attributes: &'a BTreeMap<String, String>,
    keys: &[&'static str],
) -> Option<AttributeText<'a>> {
    keys.iter().copied().find_map(|source| {
        attributes
            .get(source)
            .filter(|text| !text.is_empty())
            .map(|text| AttributeText {
                source,
                text: text.as_str(),
            })
    })
}

fn present(attributes: &BTreeMap<String, String>, keys: &[&'static str]) -> Option<String> {
    select_attribute(attributes, keys).map(|attribute| attribute.text.to_owned())
}

fn attr<'a>(attributes: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    attributes.get(key).map(String::as_str).unwrap_or_default()
}

fn tokens(attributes: &BTreeMap<String, String>, key: &str) -> Result<u32, Error> {
    let value = attr(attributes, key).trim();
    if value.is_empty() {
        return Ok(0);
    }
    match value.parse::<i128>() {
        Ok(number) if (0..=u32::MAX as i128).contains(&number) => Ok(number as u32),
        Ok(_) => Err(Error::TokenCountOutOfRange),
        Err(_)
            if value
                .trim_start_matches(['+', '-'])
                .bytes()
                .all(|byte| byte.is_ascii_digit()) =>
        {
            Err(Error::TokenCountOutOfRange)
        }
        Err(_) => Ok(0),
    }
}

fn usage_tokens(attributes: &BTreeMap<String, String>) -> Result<(u32, u32), Error> {
    Ok((
        tokens(attributes, "gen_ai.usage.input_tokens")?,
        tokens(attributes, "gen_ai.usage.output_tokens")?,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::{ObservationType, normalize};

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
