use super::{ObservationType, RoleEvidence, SpanFacts};

pub(super) fn adjust(facts: SpanFacts) -> SpanFacts {
    if facts.agent_name.as_deref() != Some("Agent") {
        return facts;
    }
    SpanFacts {
        role: Some(RoleEvidence::WrapperCandidate(ObservationType::Agent)),
        agent_name: None,
        ..facts
    }
}
