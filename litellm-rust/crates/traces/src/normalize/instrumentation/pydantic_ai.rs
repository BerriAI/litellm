use super::{Extraction, SpanContext, SpanFacts, messages, select_attribute};
use super::{Integration, Rule};
use crate::normalize::format::genai::Operation;

pub(super) const SCOPE: &str = "pydantic-ai";

pub(super) fn adjust(context: &SpanContext<'_>, extraction: Extraction) -> Extraction {
    if !matches!(
        Operation::from_context(context),
        Some(Operation::InvokeAgent)
    ) {
        return extraction;
    }
    let input = extraction
        .facts
        .input
        .is_empty()
        .then(|| select_attribute(context.attributes, &["pydantic_ai.all_messages"]))
        .flatten();
    let output = extraction
        .facts
        .output
        .is_empty()
        .then(|| select_attribute(context.attributes, &["final_result"]))
        .flatten();
    let fallback = SpanFacts {
        input: input
            .as_ref()
            .map_or(String::new(), |payload| messages::canonical(payload.text)),
        output: output
            .as_ref()
            .map_or(String::new(), |payload| payload.text.to_owned()),
        ..SpanFacts::default()
    };
    extraction
        .map_facts(|facts| facts.or(fallback))
        .consuming(input)
        .consuming(output)
}

pub(super) struct PydanticAi;

impl Rule for PydanticAi {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.scope == SCOPE
    }
    fn integration(&self, _: &SpanContext<'_>) -> Option<Integration> {
        Some(Integration::PydanticAi)
    }
    fn adjust(
        &self,
        context: &SpanContext<'_>,
        extraction: super::Extraction,
    ) -> super::Extraction {
        adjust(context, extraction)
    }
}
