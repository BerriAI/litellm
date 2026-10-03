use serde_json::Value;

use super::{Extraction, Format, SpanFacts, genai::GenAi};
use crate::{
    Error,
    normalize::{SpanContext, messages, select_attribute},
};

pub(crate) struct Logfire;

impl Format for Logfire {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.attributes.contains_key("all_messages_events")
            || ((context.scope.starts_with("logfire") || context.scope == "pydantic-ai")
                && (context.attributes.contains_key("events")
                    || context.attributes.contains_key("prompt")))
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let base = GenAi.extract(context)?;
        let input = base
            .facts
            .input
            .is_empty()
            .then(|| select_attribute(context.attributes, &["prompt"]))
            .flatten();
        let output = base
            .facts
            .output
            .is_empty()
            .then(|| select_attribute(context.attributes, &["final_result"]))
            .flatten();
        let recorded = select_attribute(context.attributes, &["all_messages_events", "events"]);
        let values = recorded
            .as_ref()
            .and_then(|value| serde_json::from_str::<Vec<Value>>(value.text).ok())
            .unwrap_or_default();
        let events: Vec<_> = values
            .iter()
            .filter_map(|value| messages::event_message(value.get("event.name")?.as_str()?, value))
            .collect();
        Ok(Extraction {
            facts: SpanFacts {
                input: if base.facts.input.is_empty() {
                    input
                        .as_ref()
                        .map(|value| messages::canonical(value.text))
                        .or_else(|| messages::event_payload(&events, false))
                        .unwrap_or_default()
                } else {
                    base.facts.input
                },
                output: if base.facts.output.is_empty() {
                    output
                        .as_ref()
                        .map(|value| value.text.to_owned())
                        .or_else(|| messages::event_payload(&events, true))
                        .unwrap_or_default()
                } else {
                    base.facts.output
                },
                ..base.facts
            },
            display_name: base.display_name,
            consumed_attributes: base
                .consumed_attributes
                .into_iter()
                .chain(
                    [input, output, recorded]
                        .into_iter()
                        .flatten()
                        .map(|value| value.source),
                )
                .collect(),
        })
    }
}
