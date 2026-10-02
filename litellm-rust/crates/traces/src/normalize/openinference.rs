use std::collections::BTreeMap;

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr, tokens, usage_tokens};
use crate::{DecodeError, otlp::DecodedEvent};

pub(super) struct OpenInferenceNormalizer;

impl SpanNormalizer for OpenInferenceNormalizer {
    fn matches(&self, _scope_name: &str, attributes: &BTreeMap<String, String>) -> bool {
        attributes.contains_key("openinference.span.kind")
    }

    fn consumed_attributes(&self, _attributes: &BTreeMap<String, String>) -> [&'static str; 2] {
        ["input.value", "output.value"]
    }

    fn normalize(
        &self,
        _name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
        _events: &[DecodedEvent],
    ) -> Result<NormalizedSpan, DecodeError> {
        let (usage_input, usage_output) = usage_tokens(attributes)?;
        let observation_type = match attr(attributes, "openinference.span.kind")
            .to_ascii_uppercase()
            .as_str()
        {
            "AGENT" => ObservationType::Agent,
            "LLM" => ObservationType::Llm,
            "TOOL" => ObservationType::Tool,
            _ if parent_span_id.is_empty() => ObservationType::Agent,
            _ => ObservationType::Chain,
        };
        Ok(NormalizedSpan {
            observation_type,
            agent_name: attr(attributes, "agent.name").to_owned(),
            framework: String::new(),
            litellm_request_id: String::new(),
            model: attr(attributes, "llm.model_name").to_owned(),
            input_tokens: if attributes.contains_key("llm.token_count.prompt") {
                tokens(attributes, "llm.token_count.prompt")?
            } else {
                usage_input
            },
            output_tokens: if attributes.contains_key("llm.token_count.completion") {
                tokens(attributes, "llm.token_count.completion")?
            } else {
                usage_output
            },
            input: attr(attributes, "input.value").to_owned(),
            output: attr(attributes, "output.value").to_owned(),
        })
    }
}
