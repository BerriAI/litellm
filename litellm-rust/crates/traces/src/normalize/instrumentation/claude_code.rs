use super::{
    Integration, ObservationType, RoleEvidence, Rule, SpanContext, SpanFacts, attr, present,
};
use crate::normalize::{CLAUDE_CODE_AGENT, CLAUDE_CODE_EVENTS_SCOPE, CLAUDE_CODE_SCOPE};
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

fn framework(attributes: &BTreeMap<String, String>) -> Integration {
    if attr(attributes, "query_source_safe") == "sdk"
        || attr(attributes, "system_prompt_preview").contains("cc_entrypoint=sdk")
    {
        Integration::ClaudeAgentSdk
    } else {
        Integration::ClaudeCode
    }
}

pub(super) struct ClaudeCode;

impl Rule for ClaudeCode {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        matches!(context.scope, SCOPE | CLAUDE_CODE_EVENTS_SCOPE)
    }
    fn integration(&self, context: &SpanContext<'_>) -> Option<Integration> {
        Some(framework(context.attributes))
    }
    fn adjust(
        &self,
        context: &SpanContext<'_>,
        extraction: super::Extraction,
    ) -> super::Extraction {
        extraction.map_facts(|facts| adjust(context, facts))
    }

    fn agent_name(&self, context: &SpanContext<'_>, recorded: Option<String>) -> Option<String> {
        match (
            present(context.resource_attributes, &["gen_ai.agent.name"]),
            recorded.as_deref(),
        ) {
            (Some(name), None | Some(CLAUDE_CODE_AGENT)) => Some(name),
            (None, Some(CLAUDE_CODE_AGENT)) => {
                present(context.resource_attributes, &["service.name"]).or(recorded)
            }
            _ => recorded,
        }
    }
}
