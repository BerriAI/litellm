use super::{ObservationType, RoleEvidence, SpanContext, SpanFacts, attr};
use serde_json::Value;

pub(super) fn adjust(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    let agent = context
        .name
        .ends_with(".run_agent_step")
        .then(|| current_agent_name(attr(context.attributes, "input.value")))
        .flatten();
    let role = match agent {
        Some(_) => Some(RoleEvidence::Declared(ObservationType::Agent)),
        None if context.name.ends_with("._prepare_chat_with_tools") => {
            Some(RoleEvidence::Declared(ObservationType::Chain))
        }
        None => facts.role,
    };
    let engine_state = context.parent_span_id.is_empty() && has_key(&facts.input, "start_event");
    SpanFacts {
        role,
        agent_name: agent.map(str::to_owned).or(facts.agent_name),
        input_preview: if engine_state {
            Some(String::new())
        } else {
            facts.input_preview
        },
        ..facts
    }
}

fn current_agent_name(input: &str) -> Option<&str> {
    let (_, rest) = input.split_once("current_agent_name='")?;
    let (agent, _) = rest.split_once('\'')?;
    (!agent.is_empty()).then_some(agent)
}

fn has_key(input: &str, key: &str) -> bool {
    serde_json::from_str::<serde_json::Map<String, Value>>(input)
        .is_ok_and(|object| object.contains_key(key))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::current_agent_name;

    #[rstest]
    #[case::named("ev=current_agent_name='delegate'", Some("delegate"))]
    #[case::missing("ev=other", None)]
    #[case::empty("current_agent_name=''", None)]
    #[case::unterminated("current_agent_name='delegate", None)]
    fn agent_name_requires_a_complete_nonempty_value(
        #[case] input: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(current_agent_name(input), expected);
    }
}
