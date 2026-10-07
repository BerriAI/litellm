use super::{
    AgentMetadata, Integration, ObservationType, RoleEvidence, SpanContext, SpanFacts, attr,
    messages,
};
use crate::normalize::present;
use std::collections::BTreeMap;

pub(super) fn adjust(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    let middleware = !context.parent_span_id.is_empty() && is_langchain_middleware(context.name);
    SpanFacts {
        role: if middleware {
            Some(RoleEvidence::Declared(ObservationType::Framework))
        } else {
            facts.role
        },
        input_preview: messages::state_preview(&facts.input, "messages"),
        ..facts
    }
}

pub(super) fn agent_name(context: &SpanContext<'_>, metadata: &AgentMetadata) -> Option<String> {
    let node = attr(context.attributes, "graph.node.id");
    if !node.is_empty() {
        return Some(node.to_owned());
    }
    (metadata.ls_integration == Some(Integration::Langgraph)
        && context.name != "LangGraph"
        && !is_langchain_middleware(context.name))
    .then(|| context.name.to_owned())
}

const MIDDLEWARE_SUFFIXES: [&str; 6] = [
    ".wrap_model_call",
    ".wrap_tool_call",
    ".before_agent",
    ".after_agent",
    ".before_model",
    ".after_model",
];

pub(super) fn is_langchain_middleware(name: &str) -> bool {
    MIDDLEWARE_SUFFIXES
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn span_type(
    name: &str,
    parent_span_id: &str,
    attributes: &BTreeMap<String, String>,
) -> ObservationType {
    match ObservationType::try_from(attr(attributes, "langsmith.span.kind")) {
        Ok(kind) if kind != ObservationType::Chain => kind,
        _ if parent_span_id.is_empty()
            || name == attr(attributes, "langsmith.metadata.lc_agent_name") =>
        {
            ObservationType::Agent
        }
        _ if is_langchain_middleware(name) => ObservationType::Framework,
        _ => ObservationType::Chain,
    }
}

pub(super) fn langsmith(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    let kind = span_type(context.name, context.parent_span_id, context.attributes);
    let input = (kind == ObservationType::Agent)
        .then(|| messages::state_conversation(&facts.input))
        .flatten();
    let output = (kind == ObservationType::Agent)
        .then(|| messages::state_conversation(&facts.output))
        .flatten()
        .and_then(|conversation| conversation.last().map(messages::encode));
    SpanFacts {
        role: Some(RoleEvidence::Declared(kind)),
        agent_name: present(context.attributes, &["langsmith.metadata.lc_agent_name"]),
        input: input.map_or(facts.input, |conversation| messages::encode(&conversation)),
        output: output.unwrap_or(facts.output),
        ..facts
    }
}
