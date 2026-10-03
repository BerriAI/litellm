//! Step one of normalization: what a span records, read in the format it was recorded in.

use super::{AttributeText, CallEvidence, RoleEvidence, SpanContext};
use crate::Error;

pub(crate) mod claude_code;
pub(crate) mod genai;
pub(crate) mod langsmith;
pub(crate) mod openinference;

/// What a span records, read in its convention's format.
#[derive(Debug, Default)]
pub(crate) struct SpanFacts {
    pub role: Option<RoleEvidence>,
    pub agent_name: Option<String>,
    pub model: Option<String>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub input: String,
    pub output: String,
    pub tool_call_id: Option<String>,
    pub calls: CallEvidence,
    /// Set when the latest user message is not simply read from `input`.
    pub input_preview: Option<String>,
}

/// A convention's complete reading of a span, including which attributes it consumed.
pub(crate) struct Extraction {
    pub facts: SpanFacts,
    pub display_name: Option<String>,
    pub consumed_attributes: Vec<&'static str>,
}

impl Extraction {
    pub(crate) fn map_facts(self, adjust: impl FnOnce(SpanFacts) -> SpanFacts) -> Self {
        Self {
            facts: adjust(self.facts),
            ..self
        }
    }
}

/// A payload read from one attribute, which the extraction then reports as consumed.
#[derive(Default)]
pub(crate) struct Payload {
    pub text: String,
    pub consumed: Option<&'static str>,
}

impl From<AttributeText<'_>> for Payload {
    fn from(attribute: AttributeText<'_>) -> Self {
        Self {
            text: attribute.text.to_owned(),
            consumed: Some(attribute.source),
        }
    }
}

/// A span format: whether a span is recorded in it, and what the span then records.
pub(crate) trait Convention {
    fn matches(&self, context: &SpanContext<'_>) -> bool;
    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error>;
}

/// In precedence order. `gen_ai` accepts every span, so it is last.
const CONVENTIONS: [&dyn Convention; 4] = [
    &claude_code::ClaudeCode,
    &langsmith::LangSmith,
    &openinference::OpenInference,
    &genai::GenAi,
];

pub(crate) fn extract(context: &SpanContext<'_>) -> Result<Extraction, Error> {
    CONVENTIONS
        .into_iter()
        .find(|convention| convention.matches(context))
        .unwrap_or(&genai::GenAi)
        .extract(context)
}
