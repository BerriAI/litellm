use std::collections::BTreeMap;

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr, first, usage_tokens};
use crate::DecodeError;

pub(super) struct GenAiNormalizer;

impl SpanNormalizer for GenAiNormalizer {
    fn matches(&self, _scope_name: &str, _attributes: &BTreeMap<String, String>) -> bool {
        true
    }

    fn consumed_attributes(&self, attributes: &BTreeMap<String, String>) -> [&'static str; 2] {
        [
            if attr(attributes, "gen_ai.input.messages").is_empty() {
                "gen_ai.tool.call.arguments"
            } else {
                "gen_ai.input.messages"
            },
            if attr(attributes, "gen_ai.output.messages").is_empty() {
                "gen_ai.tool.call.result"
            } else {
                "gen_ai.output.messages"
            },
        ]
    }

    fn normalize(
        &self,
        _name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
    ) -> Result<NormalizedSpan, DecodeError> {
        let (input_tokens, output_tokens) = usage_tokens(attributes)?;
        let observation_type = match attr(attributes, "gen_ai.operation.name") {
            "invoke_agent" => ObservationType::Agent,
            "chat" | "text_completion" | "generate_content" => ObservationType::Llm,
            "execute_tool" => ObservationType::Tool,
            _ if parent_span_id.is_empty() => ObservationType::Agent,
            _ => ObservationType::Chain,
        };
        Ok(NormalizedSpan {
            observation_type,
            agent_name: attr(attributes, "gen_ai.agent.name").to_owned(),
            litellm_request_id: attr(attributes, "gen_ai.response.id").to_owned(),
            model: first(attributes, "gen_ai.request.model", "gen_ai.response.model").to_owned(),
            input_tokens,
            output_tokens,
            input: first(
                attributes,
                "gen_ai.input.messages",
                "gen_ai.tool.call.arguments",
            )
            .to_owned(),
            output: first(
                attributes,
                "gen_ai.output.messages",
                "gen_ai.tool.call.result",
            )
            .to_owned(),
        })
    }
}
