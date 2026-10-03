use super::{
    Extraction, ObservationType, RoleEvidence, SpanContext, SpanFacts, attr, first, messages,
    select_attribute, usage_tokens,
};
use crate::Error;

#[derive(strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub(super) enum Operation {
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
    pub(super) fn from_context(context: &SpanContext<'_>) -> Option<Self> {
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

/// A message payload in the common format, or a tool payload unchanged.
fn payload(
    context: &SpanContext<'_>,
    keys: &[&'static str],
    consumed: &mut Vec<&'static str>,
) -> String {
    let Some(payload) = select_attribute(context.attributes, keys) else {
        return String::new();
    };
    consumed.push(payload.source);
    if payload.source == keys[0] {
        messages::canonical(payload.text)
    } else {
        payload.text.to_owned()
    }
}

pub(super) fn extract(context: &SpanContext<'_>) -> Result<Extraction, Error> {
    let attributes = context.attributes;
    let (input_tokens, output_tokens) = usage_tokens(attributes)?;
    let role =
        Operation::from_context(context).map(|operation| RoleEvidence::Declared(operation.role()));
    let mut consumed = Vec::new();
    let input = payload(
        context,
        &[
            "gen_ai.input.messages",
            "gen_ai.tool.call.arguments",
            "gen_ai.retrieval.query.text",
            "gen_ai.prompt",
        ],
        &mut consumed,
    );
    let output = payload(
        context,
        &[
            "gen_ai.output.messages",
            "gen_ai.tool.call.result",
            "gen_ai.retrieval.documents",
            "gen_ai.completion",
        ],
        &mut consumed,
    );
    let model = first(attributes, "gen_ai.request.model", "gen_ai.response.model");
    let tool_call_id = attr(attributes, "gen_ai.tool.call.id");
    Ok(Extraction {
        facts: SpanFacts {
            role,
            agent_name: None,
            model: (!model.is_empty()).then(|| model.to_owned()),
            input_tokens,
            output_tokens,
            input,
            output,
            tool_call_id: (!tool_call_id.is_empty()).then(|| tool_call_id.to_owned()),
            ..SpanFacts::default()
        },
        display_name: None,
        consumed_attributes: consumed,
    })
}
