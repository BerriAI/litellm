//! What is known about the SDK that emitted a span, applied to its convention's [`SpanFacts`].
//! Each rule needs positive evidence from that SDK; anything less stays a [`RoleEvidence`] for the
//! trace graph to settle.

use super::{
    AgentMetadata, AgentType, CallEvidence, CallKey, Integration, Normalization, NormalizedSpan,
    ObservationType, RoleEvidence, SpanContext, attr,
    format::{Extraction, SpanFacts},
    messages, present, select_attribute,
};

const OPENINFERENCE_PREFIX: &str = "openinference.instrumentation.";

pub(super) mod claude_agent_sdk;
pub(super) mod claude_code;
pub(super) mod google_adk;
pub(super) mod http_client;
pub(super) mod langchain;
pub(super) mod llama_index;
pub(super) mod pydantic_ai;
pub(super) mod strands;
pub(super) mod vercel;

pub(super) enum Instrumentation {
    ClaudeCode,
    /// `openinference.instrumentation.<name>`.
    OpenInference(Integration),
    PydanticAi,
    /// SDKs whose own tracer is known only by its scope.
    Named(Integration),
    HttpClient,
    Unknown,
}

impl Instrumentation {
    pub(super) fn detect(context: &SpanContext<'_>) -> Self {
        let scope = context.scope;
        if scope == claude_code::SCOPE {
            return Self::ClaudeCode;
        }
        if let Some(name) = scope.strip_prefix(OPENINFERENCE_PREFIX) {
            return Self::OpenInference(Integration::from(name.replace('_', "-")));
        }
        if http_client::matches(context) {
            return Self::HttpClient;
        }
        match scope {
            pydantic_ai::SCOPE => Self::PydanticAi,
            google_adk::SCOPE => Self::Named(Integration::GoogleAdk),
            // `@ai-sdk/otel` uses `gen_ai`; the SDK's built-in telemetry used `ai`.
            _ if vercel::matches(context) => Self::Named(Integration::VercelAiSdk),
            _ if strands::matches(context) => Self::Named(Integration::Strands),
            _ => Self::Unknown,
        }
    }

    fn framework(&self, context: &SpanContext<'_>) -> String {
        match self {
            Self::ClaudeCode => claude_code::framework(context.attributes).to_owned(),
            Self::OpenInference(name) | Self::Named(name) => name.to_string(),
            Self::PydanticAi => Integration::PydanticAi.to_string(),
            Self::HttpClient | Self::Unknown => String::new(),
        }
    }

    fn adjust(&self, context: &SpanContext<'_>, extraction: Extraction) -> Extraction {
        match self {
            Self::ClaudeCode => extraction.map_facts(|facts| claude_code::adjust(context, facts)),
            Self::OpenInference(Integration::Langchain) => {
                extraction.map_facts(|facts| langchain::adjust(context, facts))
            }
            Self::OpenInference(Integration::LlamaIndex) => {
                extraction.map_facts(|facts| llama_index::adjust(context, facts))
            }
            Self::OpenInference(Integration::ClaudeAgentSdk) => {
                extraction.map_facts(claude_agent_sdk::adjust)
            }
            Self::OpenInference(Integration::GoogleAdk) => extraction.map_facts(google_adk::adjust),
            Self::PydanticAi => pydantic_ai::adjust(context, extraction),
            Self::HttpClient => extraction.map_facts(http_client::adjust),
            Self::OpenInference(_) | Self::Named(_) | Self::Unknown => extraction,
        }
    }

    pub(super) fn interpret(
        &self,
        context: &SpanContext<'_>,
        extraction: Extraction,
        metadata: AgentMetadata,
    ) -> Normalization {
        let Extraction {
            facts,
            display_name,
            consumed_attributes,
        } = self.adjust(
            context,
            extraction.map_facts(|facts| with_response_id(context, facts)),
        );
        let role = match (facts.role, metadata.ls_agent_type) {
            (
                None
                | Some(RoleEvidence::Declared(ObservationType::Agent | ObservationType::Chain))
                | Some(RoleEvidence::WrapperCandidate(ObservationType::Agent)),
                Some(agent_type),
            ) => Some(RoleEvidence::Declared(match agent_type {
                AgentType::Root | AgentType::Subagent => ObservationType::Agent,
                AgentType::Middleware | AgentType::Compaction => ObservationType::Framework,
            })),
            (role, _) => role,
        };
        let (observation_type, wrapper_candidate) = match role.unwrap_or(RoleEvidence::Unspecified)
        {
            RoleEvidence::Declared(kind) => (kind, false),
            RoleEvidence::WrapperCandidate(kind) => (kind, true),
            // An unlabelled root may be the agent run itself, or only wrap the agents below it.
            RoleEvidence::Unspecified if context.parent_span_id.is_empty() => {
                (ObservationType::Agent, true)
            }
            RoleEvidence::Unspecified => (ObservationType::Chain, false),
        };
        let agent_name =
            recorded_agent_name(context, facts.agent_name, observation_type, &metadata);
        let framework = metadata
            .ls_integration
            .as_ref()
            .map_or_else(|| self.framework(context), ToString::to_string);
        let model = facts
            .model
            .or_else(|| metadata.ls_model_name.clone())
            .unwrap_or_default();
        let display_name = if observation_type == ObservationType::Tool {
            display_name.or_else(|| metadata.ls_tool_name.clone())
        } else {
            display_name
        };
        let keys: Vec<CallKey> = facts
            .calls
            .key_set()
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        let input_preview = facts
            .input_preview
            .unwrap_or_else(|| messages::input_preview(&facts.input));
        Normalization {
            span: NormalizedSpan {
                observation_type,
                wrapper_candidate,
                agent_name,
                framework,
                agent_metadata: metadata,
                litellm_request_id: keys
                    .iter()
                    .find_map(|key| match key {
                        CallKey::LiteLlmRequest(id) | CallKey::ProviderResponse(id) => {
                            Some(id.clone())
                        }
                        CallKey::Transport => None,
                    })
                    .unwrap_or_default(),
                call_keys: keys,
                call_evidence: facts.calls.label(),
                model,
                input_tokens: facts.input_tokens,
                output_tokens: facts.output_tokens,
                input: facts.input,
                input_preview,
                output: facts.output,
                tool_call_id: facts.tool_call_id.unwrap_or_default(),
            },
            display_name,
            consumed_attributes: consumed_attributes.into_boxed_slice(),
        }
    }
}

/// `gen_ai.response.id` names one provider response, whichever convention recorded it.
fn with_response_id(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    match present(context.attributes, &["gen_ai.response.id"]) {
        Some(id) => SpanFacts {
            calls: facts.calls.with(CallKey::ProviderResponse(id)),
            ..facts
        },
        None => facts,
    }
}

fn recorded_agent_name(
    context: &SpanContext<'_>,
    extracted: Option<String>,
    observation_type: ObservationType,
    metadata: &AgentMetadata,
) -> String {
    if let Some(name) = extracted {
        return name;
    }
    let attributes = context.attributes;
    let explicit = [
        attr(attributes, "gen_ai.agent.name"),
        attr(attributes, "agent.name"),
        attr(attributes, "openclaw.agent"),
    ]
    .into_iter()
    .find(|value| !value.is_empty());
    if let Some(value) = explicit {
        return value.to_owned();
    }
    if let Some(name) = metadata
        .lc_agent_name
        .as_ref()
        .or(metadata.ls_subagent_type.as_ref())
    {
        return name.clone();
    }
    if observation_type == ObservationType::Agent {
        return langchain::agent_name(context, metadata).unwrap_or_default();
    }
    String::new()
}
