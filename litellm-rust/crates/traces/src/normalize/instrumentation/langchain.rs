use super::{
    AgentMetadata, Integration, ObservationType, RoleEvidence, SpanContext, SpanFacts, attr,
    messages,
};
use crate::normalize::format::langsmith::is_langchain_middleware;

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
