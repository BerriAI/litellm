use super::{SpanFacts, messages};

pub(super) const SCOPE: &str = "gcp.vertex.agent";

pub(super) fn adjust(facts: SpanFacts) -> SpanFacts {
    SpanFacts {
        input_preview: messages::state_preview(&facts.input, "new_message").or(facts.input_preview),
        ..facts
    }
}
