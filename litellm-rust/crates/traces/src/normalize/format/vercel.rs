use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Extraction, Format, SpanFacts, genai::GenAi};
use crate::{
    Error,
    normalize::{
        ObservationType, RoleEvidence, SpanContext, attr, messages, present, select_attribute,
        token_alias,
    },
};

pub(crate) struct Vercel;

#[derive(Deserialize)]
struct Prompt {
    messages: Option<Value>,
    prompt: Option<String>,
    system: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct ToolCall {
    #[serde(rename(deserialize = "toolCallId"))]
    id: String,
    #[serde(rename(deserialize = "toolName"))]
    name: String,
    #[serde(alias = "args", alias = "input")]
    arguments: Value,
}

fn prompt(raw: &str) -> String {
    let Ok(value) = serde_json::from_str::<Prompt>(raw) else {
        return messages::canonical(raw);
    };
    let content = value.messages.unwrap_or_else(|| {
        Value::Array(
            value
                .prompt
                .into_iter()
                .map(|text| serde_json::json!({"role": "user", "content": text}))
                .collect(),
        )
    });
    let conversation: Vec<Value> = value
        .system
        .into_iter()
        .map(|text| serde_json::json!({"role": "system", "content": text}))
        .chain(content.as_array().into_iter().flatten().cloned())
        .collect();
    if conversation.is_empty() {
        return raw.to_owned();
    }
    messages::canonical(&messages::encode(&conversation))
}

impl Format for Vercel {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.attributes.contains_key("ai.operationId")
            || (context.scope == "ai"
                && context.attributes.keys().any(|key| key.starts_with("ai.")))
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let base = GenAi.extract(context)?;
        let operation = attr(context.attributes, "ai.operationId");
        let role = match operation {
            "ai.toolCall" => Some(ObservationType::Tool),
            "ai.embed" | "ai.embedMany" | "ai.embed.doEmbed" | "ai.embedMany.doEmbed" => {
                Some(ObservationType::Embedding)
            }
            "ai.generateText"
            | "ai.streamText"
            | "ai.generateObject"
            | "ai.streamObject"
            | "ai.generateText.doGenerate"
            | "ai.streamText.doStream"
            | "ai.generateObject.doGenerate"
            | "ai.streamObject.doStream" => Some(ObservationType::Llm),
            _ => None,
        };
        let input = base
            .facts
            .input
            .is_empty()
            .then(|| {
                select_attribute(
                    context.attributes,
                    &[
                        "ai.toolCall.args",
                        "ai.prompt.messages",
                        "ai.prompt",
                        "ai.value",
                        "ai.values",
                    ],
                )
            })
            .flatten();
        let output = base
            .facts
            .output
            .is_empty()
            .then(|| {
                select_attribute(
                    context.attributes,
                    &[
                        "ai.toolCall.result",
                        "ai.response.object",
                        "ai.response.text",
                        "ai.embeddings",
                        "ai.embedding",
                    ],
                )
            })
            .flatten();
        let calls = base
            .facts
            .output
            .is_empty()
            .then(|| select_attribute(context.attributes, &["ai.response.toolCalls"]))
            .flatten();
        let response = calls
            .as_ref()
            .and_then(|value| serde_json::from_str::<Vec<ToolCall>>(value.text).ok());
        let legacy_output = match response {
            Some(calls) => messages::canonical(&messages::encode(&serde_json::json!([{
                "role": "assistant", "content": output.as_ref().map_or("", |value| value.text), "tool_calls": calls,
            }]))),
            None => output
                .as_ref()
                .map_or(String::new(), |value| value.text.to_owned()),
        };
        Ok(Extraction {
            facts: base.facts.or(SpanFacts {
                role: role.map(RoleEvidence::Declared),
                model: present(context.attributes, &["ai.model.id"]),
                input_tokens: token_alias(
                    context.attributes,
                    &[
                        "gen_ai.usage.input_tokens",
                        "gen_ai.usage.prompt_tokens",
                        "ai.usage.promptTokens",
                        "ai.usage.tokens",
                    ],
                )?,
                output_tokens: token_alias(
                    context.attributes,
                    &[
                        "gen_ai.usage.output_tokens",
                        "gen_ai.usage.completion_tokens",
                        "ai.usage.completionTokens",
                    ],
                )?,
                input: input
                    .as_ref()
                    .map_or(String::new(), |value| match value.source {
                        "ai.prompt" | "ai.prompt.messages" => prompt(value.text),
                        _ => value.text.to_owned(),
                    }),
                output: legacy_output,
                tool_call_id: present(context.attributes, &["ai.toolCall.id"]),
                ..SpanFacts::default()
            }),
            display_name: present(context.attributes, &["ai.toolCall.name"]).or(base.display_name),
            consumed_attributes: base.consumed_attributes,
        }
        .consuming(input)
        .consuming(output)
        .consuming(calls))
    }
}
