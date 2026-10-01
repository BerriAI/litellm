use std::collections::BTreeMap;

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr, first};

pub(super) struct GenAiNormalizer;

impl SpanNormalizer for GenAiNormalizer {
    fn matches(&self, _scope_name: &str, _attributes: &BTreeMap<String, String>) -> bool {
        true
    }

    fn normalize(
        &self,
        _name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
    ) -> NormalizedSpan {
        let observation_type = match attr(attributes, "gen_ai.operation.name") {
            "invoke_agent" => ObservationType::Agent,
            _ if parent_span_id.is_empty() => ObservationType::Agent,
            "chat" | "text_completion" | "generate_content" => ObservationType::Llm,
            "execute_tool" => ObservationType::Tool,
            _ => ObservationType::Chain,
        };
        NormalizedSpan {
            observation_type,
            agent_name: attr(attributes, "gen_ai.agent.name").to_owned(),
            litellm_request_id: attr(attributes, "gen_ai.response.id").to_owned(),
            model: first(attributes, "gen_ai.request.model", "gen_ai.response.model").to_owned(),
            input_tokens: 0,
            output_tokens: 0,
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
        }
    }
}
