use super::{ObservationType, RoleEvidence, SpanContext, SpanFacts, attr};
use crate::normalize::{CLAUDE_CODE_AGENT, CLAUDE_CODE_SCOPE};
use std::collections::BTreeMap;

pub(super) const SCOPE: &str = CLAUDE_CODE_SCOPE;

pub(super) fn adjust(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    if attr(context.attributes, "parent.source") != "env"
        || facts.role != Some(RoleEvidence::Declared(ObservationType::Agent))
    {
        return facts;
    }
    SpanFacts {
        role: Some(RoleEvidence::WrapperCandidate(ObservationType::Agent)),
        ..facts
    }
}

pub(super) fn framework(attributes: &BTreeMap<String, String>) -> &'static str {
    if attr(attributes, "query_source_safe") == "sdk"
        || attr(attributes, "system_prompt_preview").contains("cc_entrypoint=sdk")
    {
        "claude-agent-sdk"
    } else {
        CLAUDE_CODE_AGENT
    }
}
