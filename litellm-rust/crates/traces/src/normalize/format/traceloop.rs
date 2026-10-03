use super::{Extraction, Format, SpanFacts, genai::GenAi};
use crate::{
    Error,
    normalize::{
        ObservationType, RoleEvidence, SpanContext, attr, messages, present, select_attribute,
    },
};

pub(crate) struct Traceloop;

impl Format for Traceloop {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context
            .attributes
            .keys()
            .any(|key| key.starts_with("traceloop."))
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let base = GenAi.extract(context)?;
        let role = match attr(context.attributes, "traceloop.span.kind") {
            "agent" => Some(ObservationType::Agent),
            "tool" => Some(ObservationType::Tool),
            "workflow" | "task" => Some(ObservationType::Chain),
            _ => match present(
                context.attributes,
                &["traceloop.llm.request.type", "llm.request.type"],
            )
            .as_deref()
            {
                Some("embedding" | "embeddings") => Some(ObservationType::Embedding),
                Some("chat" | "completion") => Some(ObservationType::Llm),
                _ => None,
            },
        };
        let input = select_attribute(context.attributes, &["traceloop.entity.input"]);
        let output = select_attribute(context.attributes, &["traceloop.entity.output"]);
        Ok(Extraction {
            facts: SpanFacts {
                role: role.map(RoleEvidence::Declared).or(base.facts.role),
                input: input
                    .as_ref()
                    .map_or(base.facts.input, |value| messages::canonical(value.text)),
                output: output
                    .as_ref()
                    .map_or(base.facts.output, |value| messages::canonical(value.text)),
                ..base.facts
            },
            display_name: present(context.attributes, &["traceloop.entity.name"])
                .or(base.display_name),
            consumed_attributes: base
                .consumed_attributes
                .into_iter()
                .chain(
                    [input, output]
                        .into_iter()
                        .flatten()
                        .map(|value| value.source),
                )
                .collect(),
        })
    }
}
