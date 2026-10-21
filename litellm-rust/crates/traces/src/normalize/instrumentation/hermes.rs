use super::{Integration, Rule, SpanContext, present};

const SCOPE: &str = "hermes-otel-plugin";

pub(super) struct Hermes;

impl Rule for Hermes {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.scope == SCOPE
    }

    fn integration(&self, _: &SpanContext<'_>) -> Option<Integration> {
        None
    }

    fn agent_name(&self, context: &SpanContext<'_>, recorded: Option<String>) -> Option<String> {
        if recorded.as_deref() == Some("hermes-agent") {
            return present(context.resource_attributes, &["gen_ai.agent.name"]).or(recorded);
        }
        recorded
    }
}
