//! Span normalization in two steps: a [`format::Format`] extracts what a span records in its format,
//! then an [`Instrumentation`] interprets those facts with what is known about the SDK that emitted
//! it. Relationships between spans (wrappers, ownership, spend) are resolved later, over the whole
//! trace, because parents and children can arrive in separate exports.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};

use crate::{Error, otlp::DecodedEvent};
use serde::{Deserialize, Serialize};

mod format;
mod instrumentation;
mod messages;
mod metadata;

pub(crate) const CLAUDE_CODE_SCOPE: &str = "com.anthropic.claude_code.tracing";
pub(crate) const CLAUDE_CODE_AGENT: &str = "claude-code";
use instrumentation::Instrumentation;
pub(crate) use messages::{HIDDEN_BLOCK_TYPES, MessagePayload, encode};
pub use metadata::{AgentMetadata, AgentType, Integration};

#[macro_rules_attribute::apply(wire_type)]
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
#[derive(
    Clone,
    Debug,
    Eq,
    Ord,
    PartialEq,
    PartialOrd,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
pub enum CallKey {
    /// LiteLLM's gateway call id, with a fallback to legacy spend request ids.
    LiteLlmRequest(String),
    /// The provider response id returned to the caller (`spend_logs.response_id`).
    ProviderResponse(String),
    /// The span is the HTTP request itself; LiteLLM logs its `traceparent` span id.
    Transport,
    GatewayAttempt,
}

impl fmt::Display for CallKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LiteLlmRequest(id) => write!(formatter, "litellm_request:{id}"),
            Self::ProviderResponse(id) => write!(formatter, "provider_response:{id}"),
            Self::Transport => formatter.write_str("transport:"),
            Self::GatewayAttempt => formatter.write_str("gateway_attempt:"),
        }
    }
}

impl FromStr for CallKey {
    type Err = crate::InvalidCallKey;

    fn from_str(encoded: &str) -> Result<Self, Self::Err> {
        match encoded.split_once(':') {
            Some(("provider_response", id)) if !id.is_empty() => {
                Ok(Self::ProviderResponse(id.to_owned()))
            }
            Some(("litellm_request", id)) if !id.is_empty() => {
                Ok(Self::LiteLlmRequest(id.to_owned()))
            }
            Some(("transport", "")) => Ok(Self::Transport),
            Some(("gateway_attempt", "")) => Ok(Self::GatewayAttempt),
            _ => Err(crate::InvalidCallKey),
        }
    }
}

impl TryFrom<String> for CallKey {
    type Error = crate::InvalidCallKey;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CallEvidenceKind {
    Unknown,
    Partial,
    Complete,
}

/// Which model requests a span accounts for. `Complete` comes only from an instrumentation's known
/// contract (one chat span is one response), never from how many ids happened to be found.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub enum CallEvidence {
    #[default]
    Unknown,
    Partial(BTreeSet<CallKey>),
    Complete(BTreeSet<CallKey>),
}

impl CallEvidence {
    pub(crate) fn row_keys(row: &crate::query::named::TraceSpansRow) -> BTreeSet<CallKey> {
        if row.call_keys.is_empty() && !row.litellm_request_id.is_empty() {
            BTreeSet::from([CallKey::ProviderResponse(row.litellm_request_id.clone())])
        } else {
            row.call_keys.iter().cloned().collect()
        }
    }

    pub(crate) fn from_row(row: &crate::query::named::TraceSpansRow) -> Self {
        let kind = row
            .call_evidence
            .unwrap_or(if Self::row_keys(row).is_empty() {
                CallEvidenceKind::Unknown
            } else {
                CallEvidenceKind::Complete
            });
        match kind {
            CallEvidenceKind::Complete => Self::Complete(Self::row_keys(row)),
            CallEvidenceKind::Partial => Self::Partial(Self::row_keys(row)),
            CallEvidenceKind::Unknown => Self::Unknown,
        }
    }

    pub(crate) fn complete(key: CallKey) -> Self {
        Self::Complete(BTreeSet::from([key]))
    }

    /// The same evidence with one more key: an id named outside the convention adds to what the
    /// convention found, but says nothing about completeness.
    fn with(self, key: CallKey) -> Self {
        match self {
            Self::Unknown => Self::Partial(BTreeSet::from([key])),
            Self::Partial(keys) => Self::Partial(keys.into_iter().chain([key]).collect()),
            Self::Complete(keys) => Self::Complete(keys.into_iter().chain([key]).collect()),
        }
    }

    pub fn key_set(&self) -> Option<&BTreeSet<CallKey>> {
        match self {
            Self::Unknown => None,
            Self::Partial(keys) | Self::Complete(keys) => Some(keys),
        }
    }

    pub fn kind(&self) -> CallEvidenceKind {
        match self {
            Self::Unknown => CallEvidenceKind::Unknown,
            Self::Partial(_) => CallEvidenceKind::Partial,
            Self::Complete(_) => CallEvidenceKind::Complete,
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
    pub resource_attributes: &'a BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedSpan {
    pub observation_type: ObservationType,
    pub wrapper_candidate: bool,
    pub agent_name: Option<String>,
    pub framework: Option<Integration>,
    pub agent_metadata: AgentMetadata,
    pub calls: CallEvidence,
    pub model: Option<String>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub input: String,
    pub input_preview: String,
    pub output: String,
    pub tool_call_id: Option<String>,
}

pub(crate) struct Normalization {
    pub span: NormalizedSpan,
    pub display_name: Option<String>,
    pub consumed_attributes: Box<[&'static str]>,
}

pub(crate) fn normalize(context: &SpanContext<'_>) -> Result<Normalization, Error> {
    let extraction = format::extract(context)?;
    Ok(Instrumentation::detect(context).interpret(context, extraction, metadata::extract(context)))
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

fn token_alias(attributes: &BTreeMap<String, String>, keys: &[&'static str]) -> Result<u32, Error> {
    select_attribute(attributes, keys)
        .map_or(Ok(0), |attribute| tokens(attributes, attribute.source))
}

fn usage_tokens(attributes: &BTreeMap<String, String>) -> Result<(u32, u32), Error> {
    Ok((
        token_alias(
            attributes,
            &["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"],
        )?,
        token_alias(
            attributes,
            &[
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.completion_tokens",
            ],
        )?,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::{ObservationType, SpanContext, normalize};

    #[rstest]
    #[case::langsmith("langsmith", [("langsmith.span.kind", "llm"), ("openinference.span.kind", "TOOL")], ObservationType::Llm)]
    #[case::openinference("other", [("openinference.span.kind", "LLM"), ("gen_ai.operation.name", "execute_tool")], ObservationType::Llm)]
    #[case::genai("other", [("gen_ai.operation.name", "execute_tool"), ("gen_ai.usage.input_tokens", "7")], ObservationType::Tool)]
    #[case::claude_code("com.anthropic.claude_code.tracing", [("span.type", "llm_request"), ("openinference.span.kind", "TOOL")], ObservationType::Llm)]
    fn format_dispatch_preserves_precedence(
        #[case] scope: &str,
        #[case] attributes: [(&str, &str); 2],
        #[case] expected: ObservationType,
    ) {
        let attributes = attributes
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        let fields = normalize(&SpanContext {
            scope,
            name: "step",
            parent_span_id: "parent",
            attributes: &attributes,
            events: &[],
            resource_attributes: &BTreeMap::new(),
        })
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
        let fields = normalize(&SpanContext {
            scope: "",
            name: "root",
            parent_span_id: "",
            attributes: &attributes,
            events: &[],
            resource_attributes: &BTreeMap::new(),
        })
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
        assert!(
            normalize(&SpanContext {
                scope: "",
                name: "root",
                parent_span_id: "",
                attributes: &attributes,
                events: &[],
                resource_attributes: &BTreeMap::new()
            })
            .is_err()
        );
    }
}
