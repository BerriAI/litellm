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
pub(super) mod hermes;
pub(super) mod http_client;
pub(super) mod langchain;
pub(super) mod llama_index;
pub(super) mod pydantic_ai;

pub(super) trait Rule: Sync {
    fn matches(&self, context: &SpanContext<'_>) -> bool;
    fn integration(&self, context: &SpanContext<'_>) -> Option<Integration>;
    fn agent_name(&self, _: &SpanContext<'_>, recorded: Option<String>) -> Option<String> {
        recorded
    }
    fn adjust(&self, _: &SpanContext<'_>, extraction: Extraction) -> Extraction {
        extraction
    }
}

struct Scoped {
    scope: &'static str,
    integration: Integration,
    prefix: bool,
}

impl Rule for Scoped {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        if self.prefix {
            context.scope.starts_with(self.scope)
        } else {
            context.scope == self.scope
        }
    }

    fn integration(&self, _: &SpanContext<'_>) -> Option<Integration> {
        Some(self.integration.clone())
    }
}

struct OpenInference;

impl Rule for OpenInference {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.scope.starts_with(OPENINFERENCE_PREFIX)
    }

    fn integration(&self, context: &SpanContext<'_>) -> Option<Integration> {
        context
            .scope
            .strip_prefix(OPENINFERENCE_PREFIX)
            .filter(|name| !name.is_empty())
            .map(|name| Integration::from(name.replace('_', "-")))
    }

    fn agent_name(&self, context: &SpanContext<'_>, recorded: Option<String>) -> Option<String> {
        recorded.filter(|name| {
            name != "Agent" || self.integration(context) != Some(Integration::ClaudeAgentSdk)
        })
    }

    fn adjust(&self, context: &SpanContext<'_>, extraction: Extraction) -> Extraction {
        match self.integration(context) {
            Some(Integration::Langchain) => {
                extraction.map_facts(|facts| langchain::adjust(context, facts))
            }
            Some(Integration::LlamaIndex) => {
                extraction.map_facts(|facts| llama_index::adjust(context, facts))
            }
            Some(Integration::ClaudeAgentSdk) => extraction.map_facts(claude_agent_sdk::adjust),
            Some(Integration::GoogleAdk) => extraction.map_facts(google_adk::adjust),
            _ => extraction,
        }
    }
}

const RULES: [&dyn Rule; 9] = [
    &claude_code::ClaudeCode,
    &hermes::Hermes,
    &OpenInference,
    &http_client::HttpClient,
    &pydantic_ai::PydanticAi,
    &Scoped {
        scope: google_adk::SCOPE,
        integration: Integration::GoogleAdk,
        prefix: false,
    },
    &Scoped {
        scope: "gen_ai",
        integration: Integration::VercelAiSdk,
        prefix: false,
    },
    &Scoped {
        scope: "ai",
        integration: Integration::VercelAiSdk,
        prefix: false,
    },
    &Scoped {
        scope: "strands.",
        integration: Integration::Strands,
        prefix: true,
    },
];

pub(super) struct Instrumentation(Option<&'static dyn Rule>);

impl Instrumentation {
    pub(super) fn detect(context: &SpanContext<'_>) -> Self {
        Self(RULES.into_iter().find(|rule| rule.matches(context)))
    }

    fn adjust(&self, context: &SpanContext<'_>, extraction: Extraction) -> Extraction {
        match self.0 {
            Some(rule) => rule.adjust(context, extraction),
            None => extraction,
        }
    }

    pub(super) fn interpret(
        &self,
        context: &SpanContext<'_>,
        extraction: Extraction,
        metadata: AgentMetadata,
    ) -> Normalization {
        let prepared = if context.scope != claude_code::SCOPE
            && (context.scope == "langsmith"
                || context.attributes.contains_key("langsmith.span.kind"))
        {
            extraction.map_facts(|facts| langchain::langsmith(context, facts))
        } else {
            extraction
        };
        let Extraction {
            facts,
            display_name,
            consumed_attributes,
        } = self.adjust(context, prepared);
        let facts = with_call_ids(context, facts);
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
        let recorded_name =
            recorded_agent_name(context, facts.agent_name, observation_type, &metadata);
        let sdk_name = match self.0 {
            Some(rule) => rule.agent_name(context, recorded_name),
            None => recorded_name,
        };
        let agent_name =
            sdk_name.or_else(|| present(context.resource_attributes, &["gen_ai.agent.name"]));
        let framework = metadata
            .ls_integration
            .clone()
            .or_else(|| self.0.and_then(|rule| rule.integration(context)));
        let model = facts.model.or_else(|| metadata.ls_model_name.clone());
        let display_name = if observation_type == ObservationType::Tool {
            display_name.or_else(|| metadata.ls_tool_name.clone())
        } else {
            display_name
        };
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
                calls: facts.calls,
                model,
                input_tokens: facts.input_tokens,
                output_tokens: facts.output_tokens,
                input: facts.input,
                input_preview,
                output: facts.output,
                tool_call_id: facts.tool_call_id,
            },
            display_name,
            consumed_attributes: consumed_attributes.into_boxed_slice(),
        }
    }
}

fn with_call_ids(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    let calls = [
        present(context.attributes, &["gen_ai.response.id"]).map(|id| {
            if matches!(
                context.scope,
                crate::normalize::CLAUDE_CODE_SCOPE | crate::normalize::CLAUDE_CODE_EVENTS_SCOPE
            ) {
                crate::normalize::claude_call_key(id)
            } else {
                CallKey::ProviderResponse(id)
            }
        }),
        present(context.attributes, &["litellm.call_id"]).map(CallKey::LiteLlmRequest),
    ]
    .into_iter()
    .flatten()
    .fold(facts.calls, CallEvidence::with);
    SpanFacts { calls, ..facts }
}

fn recorded_agent_name(
    context: &SpanContext<'_>,
    extracted: Option<String>,
    observation_type: ObservationType,
    metadata: &AgentMetadata,
) -> Option<String> {
    if let Some(name) = extracted {
        return Some(name);
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
        return Some(value.to_owned());
    }
    if let Some(name) = metadata
        .lc_agent_name
        .as_ref()
        .or(metadata.ls_subagent_type.as_ref())
    {
        return Some(name.clone());
    }
    if observation_type == ObservationType::Agent {
        return langchain::agent_name(context, metadata);
    }
    None
}
