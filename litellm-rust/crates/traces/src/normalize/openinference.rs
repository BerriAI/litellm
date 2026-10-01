use std::collections::BTreeMap;

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr, tokens};

pub(super) struct OpenInferenceNormalizer;

impl SpanNormalizer for OpenInferenceNormalizer {
    fn matches(&self, _scope_name: &str, attributes: &BTreeMap<String, String>) -> bool {
        attributes.contains_key("openinference.span.kind")
    }

    fn normalize(
        &self,
        _name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
    ) -> NormalizedSpan {
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
        NormalizedSpan {
            observation_type,
            agent_name: attr(attributes, "agent.name").to_owned(),
            litellm_request_id: String::new(),
            model: attr(attributes, "llm.model_name").to_owned(),
            input_tokens: tokens(attributes, "llm.token_count.prompt"),
            output_tokens: tokens(attributes, "llm.token_count.completion"),
            input: attr(attributes, "input.value").to_owned(),
            output: attr(attributes, "output.value").to_owned(),
        }
    }
}
