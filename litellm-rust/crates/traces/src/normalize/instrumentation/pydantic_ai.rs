use super::{Extraction, SpanContext, SpanFacts, messages, select_attribute};
use crate::normalize::format::genai::Operation;

pub(super) const SCOPE: &str = "pydantic-ai";

pub(super) fn adjust(context: &SpanContext<'_>, extraction: Extraction) -> Extraction {
    if !matches!(
        Operation::from_context(context),
        Some(Operation::InvokeAgent)
    ) {
        return extraction;
    }
    let Extraction {
        facts,
        display_name,
        consumed_attributes,
    } = extraction;
    let input = facts
        .input
        .is_empty()
        .then(|| select_attribute(context.attributes, &["pydantic_ai.all_messages"]))
        .flatten();
    let output = facts
        .output
        .is_empty()
        .then(|| select_attribute(context.attributes, &["final_result"]))
        .flatten();
    Extraction {
        facts: SpanFacts {
            input: input
                .as_ref()
                .map_or(facts.input, |payload| messages::canonical(payload.text)),
            output: output
                .as_ref()
                .map_or(facts.output, |payload| payload.text.to_owned()),
            ..facts
        },
        display_name,
        consumed_attributes: consumed_attributes
            .into_iter()
            .chain(
                [input, output]
                    .into_iter()
                    .flatten()
                    .map(|payload| payload.source),
            )
            .collect(),
    }
}
