use std::collections::BTreeMap;

use litellm_llms_types::recognized::Recognized;
use serde::{Deserialize, de::IgnoredAny};
use serde_json::Value;

use super::{Extraction, Format, Payload, SpanFacts};
use crate::{
    Error,
    normalize::{
        CallEvidence, CallKey, ObservationType, RoleEvidence, SpanContext, attr, messages, present,
        select_attribute, tokens, usage_tokens,
    },
};

/// Arize OpenInference: spans carry `openinference.span.kind`.
pub(crate) struct OpenInference;

#[derive(Deserialize)]
struct ResponseIdentity {
    #[serde(default, deserialize_with = "messages::present")]
    id: Option<Recognized<String>>,
    #[serde(flatten)]
    _other: BTreeMap<String, IgnoredAny>,
}

#[derive(Deserialize)]
struct ProviderResponse {
    raw: Option<Recognized<ResponseIdentity>>,
    #[serde(flatten)]
    response: ResponseIdentity,
}

impl ProviderResponse {
    fn id(&self) -> Option<&str> {
        let identity = match &self.response.id {
            Some(id) => return id.known().map(String::as_str),
            None => self.raw.as_ref()?.known()?,
        };
        identity.id.as_ref()?.known().map(String::as_str)
    }
}

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
    if let Ok(response) = ProviderResponse::deserialize(&value)
        && let Some(id) = response.id()
    {
        return CallEvidence::complete(CallKey::ProviderResponse(id.to_owned()));
    }
    messages::langchain_result(&value).map_or(CallEvidence::Unknown, |result| result.calls)
}

/// `llm.<direction>_messages.*` when the instrumentation flattened the messages, else `raw`.
fn payload(context: &SpanContext<'_>, flattened: &str, raw: &'static str) -> Payload {
    if let Some(conversation) = messages::flattened(context.attributes, flattened) {
        return Payload {
            text: messages::encode(&conversation),
            consumed: None,
        };
    }
    select_attribute(context.attributes, &[raw])
        .map(Payload::from)
        .unwrap_or_default()
}

/// OpenInference's own count when recorded, else the `gen_ai.usage.*` one.
fn token_count(attributes: &BTreeMap<String, String>, key: &str, usage: u32) -> Result<u32, Error> {
    if attributes.contains_key(key) {
        tokens(attributes, key)
    } else {
        Ok(usage)
    }
}

impl Format for OpenInference {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.attributes.contains_key("openinference.span.kind")
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let attributes = context.attributes;
        let (usage_input, usage_output) = usage_tokens(attributes)?;
        let role = role(context);
        let input = payload(context, "llm.input_messages", "input.value");
        let output = payload(context, "llm.output_messages", "output.value");
        Ok(Extraction {
            facts: SpanFacts {
                role,
                agent_name: present(attributes, &["agent.name"]),
                model: present(attributes, &["llm.model_name", "embedding.model_name"]),
                input_tokens: token_count(attributes, "llm.token_count.prompt", usage_input)?,
                output_tokens: token_count(attributes, "llm.token_count.completion", usage_output)?,
                input: input.text,
                output: output.text,
                tool_call_id: present(attributes, &["tool.id"]),
                calls: if role == Some(RoleEvidence::Declared(ObservationType::Llm)) {
                    calls(attr(attributes, "output.value"))
                } else {
                    CallEvidence::Unknown
                },
                input_preview: None,
            },
            display_name: None,
            consumed_attributes: [input.consumed, output.consumed]
                .into_iter()
                .flatten()
                .collect(),
        })
    }
}
