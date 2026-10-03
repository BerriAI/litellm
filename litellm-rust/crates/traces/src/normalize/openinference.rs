use serde_json::Value;

use super::{
    CallEvidence, CallKey, Extraction, ObservationType, RoleEvidence, SpanContext, SpanFacts, attr,
    first, messages, tokens, usage_tokens,
};
use crate::Error;

fn role(context: &SpanContext<'_>) -> Option<RoleEvidence> {
    let root = context.parent_span_id.is_empty();
    match ObservationType::try_from(attr(context.attributes, "openinference.span.kind")) {
        // A root chain (crew kickoff, workflow run) may be the agent run or only wrap its agents.
        Ok(ObservationType::Chain) if root => {
            Some(RoleEvidence::WrapperCandidate(ObservationType::Agent))
        }
        Ok(kind) => Some(RoleEvidence::Declared(kind)),
        _ if root => None,
        _ => Some(RoleEvidence::Declared(ObservationType::Chain)),
    }
}

/// LLM instrumentations record the provider response as `output.value`: a raw response is one
/// request (`id`); a LangChain `LLMResult` carries one per prompt.
fn calls(output: &str) -> CallEvidence {
    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return CallEvidence::Unknown;
    };
    if let Some(id) = value.get("id").and_then(Value::as_str) {
        return CallEvidence::complete(CallKey::ProviderResponse(id.to_owned()));
    }
    messages::langchain_result(&value).map_or(CallEvidence::Unknown, |result| result.calls)
}

/// `llm.<direction>_messages.*` when the instrumentation flattened the messages, else `raw`.
fn payload(
    context: &SpanContext<'_>,
    flattened: &str,
    raw: &'static str,
    consumed: &mut Vec<&'static str>,
) -> String {
    if let Some(conversation) = messages::flattened(context.attributes, flattened) {
        return messages::encode(&conversation);
    }
    let text = attr(context.attributes, raw);
    if !text.is_empty() {
        consumed.push(raw);
    }
    text.to_owned()
}

fn present(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

pub(super) fn extract(context: &SpanContext<'_>) -> Result<Extraction, Error> {
    let attributes = context.attributes;
    let (usage_input, usage_output) = usage_tokens(attributes)?;
    let role = role(context);
    let mut consumed = Vec::new();
    let input = payload(context, "llm.input_messages", "input.value", &mut consumed);
    let output = payload(
        context,
        "llm.output_messages",
        "output.value",
        &mut consumed,
    );
    Ok(Extraction {
        facts: SpanFacts {
            role,
            agent_name: present(attr(attributes, "agent.name")),
            model: present(first(attributes, "llm.model_name", "embedding.model_name")),
            input_tokens: if attributes.contains_key("llm.token_count.prompt") {
                tokens(attributes, "llm.token_count.prompt")?
            } else {
                usage_input
            },
            output_tokens: if attributes.contains_key("llm.token_count.completion") {
                tokens(attributes, "llm.token_count.completion")?
            } else {
                usage_output
            },
            input,
            output,
            tool_call_id: present(attr(attributes, "tool.id")),
            calls: if role == Some(RoleEvidence::Declared(ObservationType::Llm)) {
                calls(attr(attributes, "output.value"))
            } else {
                CallEvidence::Unknown
            },
            input_preview: None,
        },
        display_name: None,
        consumed_attributes: consumed,
    })
}
