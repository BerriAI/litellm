//! What is known about the SDK that emitted a span, applied to its convention's [`SpanFacts`].
//! Each rule needs positive evidence from that SDK; anything less stays a [`RoleEvidence`] for the
//! trace graph to settle.

use serde_json::Value;

use super::{
    AgentMetadata, AgentType, CLAUDE_CODE_SCOPE, CallEvidence, CallKey, Extraction, Integration,
    Normalization, NormalizedSpan, ObservationType, RoleEvidence, SpanContext, SpanFacts, attr,
    claude_code, messages, select_attribute,
};

const OPENINFERENCE_PREFIX: &str = "openinference.instrumentation.";

/// OpenTelemetry HTTP client instrumentations: each span is one outgoing request.
const HTTP_CLIENT_SCOPES: [&str; 7] = [
    "opentelemetry.instrumentation.httpx",
    "opentelemetry.instrumentation.requests",
    "opentelemetry.instrumentation.aiohttp_client",
    "opentelemetry.instrumentation.urllib3",
    "opentelemetry.instrumentation.urllib",
    "@opentelemetry/instrumentation-http",
    "@opentelemetry/instrumentation-undici",
];

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
        if scope == CLAUDE_CODE_SCOPE {
            return Self::ClaudeCode;
        }
        if let Some(name) = scope.strip_prefix(OPENINFERENCE_PREFIX) {
            return Self::OpenInference(Integration::from(name.replace('_', "-")));
        }
        if HTTP_CLIENT_SCOPES.contains(&scope) {
            return Self::HttpClient;
        }
        match scope {
            "pydantic-ai" => Self::PydanticAi,
            "gcp.vertex.agent" => Self::Named(Integration::GoogleAdk),
            // `@ai-sdk/otel` uses `gen_ai`; the SDK's built-in telemetry used `ai`.
            "gen_ai" | "ai" => Self::Named(Integration::VercelAiSdk),
            _ if scope.starts_with("strands.") => Self::Named(Integration::Strands),
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

    fn adjust(&self, context: &SpanContext<'_>, extraction: &mut Extraction) {
        let facts = &mut extraction.facts;
        match self {
            Self::ClaudeCode => {
                // Started from a TRACEPARENT in the environment: an Agent SDK query span may
                // already stand for this agent turn.
                if attr(context.attributes, "parent.source") == "env"
                    && facts.role == Some(RoleEvidence::Declared(ObservationType::Agent))
                {
                    facts.role = Some(RoleEvidence::WrapperCandidate(ObservationType::Agent));
                }
            }
            Self::OpenInference(Integration::Langchain) => {
                if !context.parent_span_id.is_empty() && is_langchain_middleware(context.name) {
                    facts.role = Some(RoleEvidence::Declared(ObservationType::Framework));
                }
                // LangGraph state: `{"messages": [...]}`.
                facts.input_preview = state_messages_preview(&facts.input, "messages");
            }
            Self::OpenInference(Integration::LlamaIndex) => {
                if context.name.ends_with(".run_agent_step") {
                    if let Some(agent) = current_agent_name(attr(context.attributes, "input.value"))
                    {
                        facts.role = Some(RoleEvidence::Declared(ObservationType::Agent));
                        facts.agent_name = Some(agent.to_owned());
                    }
                } else if context.name.ends_with("._prepare_chat_with_tools") {
                    // Builds the request; the `achat` that follows sends it.
                    facts.role = Some(RoleEvidence::Declared(ObservationType::Chain));
                }
                // A workflow run's arguments are engine state; the user message is not recorded.
                if context.parent_span_id.is_empty() && has_key(&facts.input, "start_event") {
                    facts.input_preview = Some(String::new());
                }
            }
            Self::OpenInference(Integration::ClaudeAgentSdk) => {
                // The subagent span is named after the `Agent` tool, not the subagent it runs.
                if facts.agent_name.as_deref() == Some("Agent") {
                    facts.role = Some(RoleEvidence::WrapperCandidate(ObservationType::Agent));
                    facts.agent_name = Some(String::new());
                }
            }
            Self::OpenInference(Integration::GoogleAdk) => {
                // `Runner.run_async` arguments: the user turn is `new_message`.
                if let Some(preview) = state_messages_preview(&facts.input, "new_message") {
                    facts.input_preview = Some(preview);
                }
            }
            Self::PydanticAi => {
                // An agent run records its messages and result only under pydantic-ai's names.
                if matches!(
                    super::genai::Operation::from_context(context),
                    Some(super::genai::Operation::InvokeAgent)
                ) {
                    if facts.input.is_empty()
                        && let Some(payload) =
                            select_attribute(context.attributes, &["pydantic_ai.all_messages"])
                    {
                        facts.input = messages::canonical(payload.text);
                        extraction.consumed_attributes.push(payload.source);
                    }
                    if facts.output.is_empty()
                        && let Some(payload) =
                            select_attribute(context.attributes, &["final_result"])
                    {
                        facts.output = payload.text.to_owned();
                        extraction.consumed_attributes.push(payload.source);
                    }
                }
            }
            Self::HttpClient => {
                facts.role = Some(RoleEvidence::Declared(ObservationType::Framework));
                facts.calls = CallEvidence::complete(CallKey::Transport);
            }
            Self::OpenInference(_) | Self::Named(_) | Self::Unknown => {}
        }
    }

    pub(super) fn interpret(
        &self,
        context: &SpanContext<'_>,
        mut extraction: Extraction,
        metadata: AgentMetadata,
    ) -> Normalization {
        add_response_id(context, &mut extraction.facts);
        self.adjust(context, &mut extraction);
        let Extraction {
            facts,
            display_name,
            consumed_attributes,
        } = extraction;
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
fn add_response_id(context: &SpanContext<'_>, facts: &mut SpanFacts) {
    let id = attr(context.attributes, "gen_ai.response.id");
    if id.is_empty() {
        return;
    }
    let key = CallKey::ProviderResponse(id.to_owned());
    facts.calls = match std::mem::take(&mut facts.calls) {
        CallEvidence::Unknown => CallEvidence::complete(key),
        CallEvidence::Partial(mut keys) => {
            keys.insert(key);
            CallEvidence::Partial(keys)
        }
        CallEvidence::Complete(mut keys) => {
            keys.insert(key);
            CallEvidence::Complete(keys)
        }
    };
}

/// The convention's agent name when it read one (even "none"), else the generic agent attributes.
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
        let node = attr(attributes, "graph.node.id");
        if !node.is_empty() {
            return node.to_owned();
        }
        if metadata.ls_integration == Some(Integration::Langgraph)
            && context.name != "LangGraph"
            && !is_langchain_middleware(context.name)
        {
            return context.name.to_owned();
        }
    }
    String::new()
}

fn is_langchain_middleware(name: &str) -> bool {
    [
        ".wrap_model_call",
        ".wrap_tool_call",
        ".before_agent",
        ".after_agent",
        ".before_model",
        ".after_model",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

/// The run step's `ev` repr names the agent it runs: `current_agent_name='search_agent'`.
fn current_agent_name(input: &str) -> Option<&str> {
    let (_, rest) = input.split_once("current_agent_name='")?;
    let (agent, _) = rest.split_once('\'')?;
    (!agent.is_empty()).then_some(agent)
}

fn has_key(input: &str, key: &str) -> bool {
    serde_json::from_str::<serde_json::Map<String, Value>>(input)
        .is_ok_and(|object| object.contains_key(key))
}

fn state_messages_preview(input: &str, key: &str) -> Option<String> {
    let mut object = serde_json::from_str::<serde_json::Map<String, Value>>(input).ok()?;
    let conversation = messages::parse(&object.remove(key)?)?;
    Some(messages::preview(&conversation))
}
