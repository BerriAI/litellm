use super::{Convention, Extraction, Payload, SpanFacts};
use crate::{
    Error,
    normalize::{
        ObservationType, RoleEvidence, SpanContext, attr, messages, present, select_attribute,
        usage_tokens,
    },
};

/// OpenTelemetry GenAI semantic conventions: the fallback, since any span may carry `gen_ai.*`.
pub(crate) struct GenAi;

#[derive(strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Operation {
    CreateAgent,
    InvokeAgent,
    InvokeWorkflow,
    Chat,
    TextCompletion,
    GenerateContent,
    ExecuteTool,
    Embeddings,
    Retrieval,
}

impl Operation {
    pub(crate) fn from_context(context: &SpanContext<'_>) -> Option<Self> {
        Self::try_from(attr(context.attributes, "gen_ai.operation.name")).ok()
    }

    fn role(self) -> ObservationType {
        match self {
            Self::InvokeAgent => ObservationType::Agent,
            Self::CreateAgent => ObservationType::Framework,
            Self::InvokeWorkflow => ObservationType::Chain,
            Self::Chat | Self::TextCompletion | Self::GenerateContent => ObservationType::Llm,
            Self::ExecuteTool => ObservationType::Tool,
            Self::Embeddings => ObservationType::Embedding,
            Self::Retrieval => ObservationType::Retriever,
        }
    }
}

const INPUT_KEYS: [&str; 4] = [
    "gen_ai.input.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.retrieval.query.text",
    "gen_ai.prompt",
];

const OUTPUT_KEYS: [&str; 4] = [
    "gen_ai.output.messages",
    "gen_ai.tool.call.result",
    "gen_ai.retrieval.documents",
    "gen_ai.completion",
];

/// The messages key comes first and is put in the common format; other payloads stay as recorded.
fn payload(context: &SpanContext<'_>, keys: &[&'static str]) -> Payload {
    let Some(attribute) = select_attribute(context.attributes, keys) else {
        return Payload::default();
    };
    Payload {
        text: if attribute.source == keys[0] {
            messages::canonical(attribute.text)
        } else {
            attribute.text.to_owned()
        },
        consumed: Some(attribute.source),
    }
}

impl Convention for GenAi {
    fn matches(&self, _context: &SpanContext<'_>) -> bool {
        true
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let attributes = context.attributes;
        let (input_tokens, output_tokens) = usage_tokens(attributes)?;
        let input = payload(context, &INPUT_KEYS);
        let output = payload(context, &OUTPUT_KEYS);
        Ok(Extraction {
            facts: SpanFacts {
                role: Operation::from_context(context)
                    .map(|operation| RoleEvidence::Declared(operation.role())),
                model: present(
                    attributes,
                    &["gen_ai.request.model", "gen_ai.response.model"],
                ),
                input_tokens,
                output_tokens,
                input: input.text,
                output: output.text,
                tool_call_id: present(attributes, &["gen_ai.tool.call.id"]),
                ..SpanFacts::default()
            },
            display_name: None,
            consumed_attributes: [input.consumed, output.consumed]
                .into_iter()
                .flatten()
                .collect(),
        })
    }
}
