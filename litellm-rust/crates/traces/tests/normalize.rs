use litellm_traces::{DecodedSpan, ObservationType, decode_otlp};
use rstest::rstest;
use serde_json::Value;

fn assert_invariants(span: &DecodedSpan) {
    let normalized = &span.normalized;
    if normalized.wrapper_candidate {
        assert_eq!(normalized.observation_type, ObservationType::Agent);
    }
    if normalized.observation_type == ObservationType::Tool {
        assert!(
            !span.name.is_empty(),
            "tool has no display name: {}",
            span.span_id
        );
    }
    if let Some(id) = span
        .attributes
        .get("gen_ai.response.id")
        .filter(|id| !id.is_empty())
    {
        assert!(
            normalized
                .calls
                .key_set()
                .into_iter()
                .flatten()
                .any(|key| key.to_string() == format!("provider_response:{id}"))
        );
        assert_ne!(
            normalized.calls.kind(),
            litellm_traces::CallEvidenceKind::Unknown
        );
    }
    if normalized.calls.kind() == litellm_traces::CallEvidenceKind::Unknown {
        assert!(
            normalized
                .calls
                .key_set()
                .is_none_or(|keys| keys.is_empty())
        );
    } else {
        assert!(
            !normalized
                .calls
                .key_set()
                .is_none_or(|keys| keys.is_empty())
        );
    }
    for (actual, keys) in [
        (
            normalized.input_tokens,
            [
                "llm.token_count.prompt",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.prompt_tokens",
            ],
        ),
        (
            normalized.output_tokens,
            [
                "llm.token_count.completion",
                "gen_ai.usage.output_tokens",
                "output_tokens",
            ],
        ),
    ] {
        if let Some(recorded) = keys.iter().find_map(|key| span.attributes.get(*key)) {
            assert_eq!(
                actual,
                recorded.parse::<u32>().expect("fixture token count")
            );
        }
    }
    if span.attributes.contains_key("input_tokens") {
        let recorded_input = ["input_tokens", "cache_read_tokens", "cache_creation_tokens"]
            .iter()
            .filter_map(|key| span.attributes.get(*key))
            .map(|value| value.parse::<u32>().expect("fixture token count"))
            .sum::<u32>();
        assert_eq!(normalized.input_tokens, recorded_input);
    }
    assert!(normalized.input_preview.chars().count() <= 240);
    if let Ok(Value::Array(messages)) = serde_json::from_str(&normalized.input) {
        let user = messages.iter().rev().find_map(|message| {
            (message.get("role")?.as_str()? == "user")
                .then(|| {
                    message
                        .get("content")?
                        .as_str()
                        .filter(|content| !content.is_empty())
                })
                .flatten()
        });
        if let Some(content) = user {
            assert_eq!(
                normalized.input_preview,
                content.chars().take(240).collect::<String>()
            );
        }
    }
}

fn array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

#[rstest]
#[case::claude_agent_sdk_detailed_export(include_bytes!("fixtures/claude_agent_sdk_detailed_export.json"))]
#[case::claude_agent_sdk_export(include_bytes!("fixtures/claude_agent_sdk_export.json"))]
#[case::claude_agent_sdk_simple(include_bytes!("fixtures/claude_agent_sdk_simple.json"))]
#[case::claude_agent_sdk_swarm(include_bytes!("fixtures/claude_agent_sdk_swarm.json"))]
#[case::claude_missing_id_simple(include_bytes!("fixtures/claude_agent_sdk_missing_request_id_simple.json"))]
#[case::claude_missing_id_swarm(include_bytes!("fixtures/claude_agent_sdk_missing_request_id_swarm.json"))]
#[case::crewai_simple(include_bytes!("fixtures/crewai_simple.json"))]
#[case::crewai_swarm(include_bytes!("fixtures/crewai_swarm.json"))]
#[case::deepagents_simple(include_bytes!("fixtures/deepagents_simple.json"))]
#[case::deepagents_swarm(include_bytes!("fixtures/deepagents_swarm.json"))]
#[case::google_adk_simple(include_bytes!("fixtures/google_adk_simple.json"))]
#[case::google_adk_swarm(include_bytes!("fixtures/google_adk_swarm.json"))]
#[case::langchain_simple(include_bytes!("fixtures/langchain_simple.json"))]
#[case::langchain_swarm(include_bytes!("fixtures/langchain_swarm.json"))]
#[case::langgraph_simple(include_bytes!("fixtures/langgraph_simple.json"))]
#[case::langgraph_swarm(include_bytes!("fixtures/langgraph_swarm.json"))]
#[case::langsmith_deep_agent_export(include_bytes!("fixtures/langsmith_deep_agent_export.json"))]
#[case::llamaindex_simple(include_bytes!("fixtures/llamaindex_simple.json"))]
#[case::llamaindex_swarm(include_bytes!("fixtures/llamaindex_swarm.json"))]
#[case::openai_agents_simple(include_bytes!("fixtures/openai_agents_simple.json"))]
#[case::openai_agents_swarm(include_bytes!("fixtures/openai_agents_swarm.json"))]
#[case::opentelemetry_simple(include_bytes!("fixtures/opentelemetry_simple.json"))]
#[case::opentelemetry_swarm(include_bytes!("fixtures/opentelemetry_swarm.json"))]
#[case::pydantic_ai_simple(include_bytes!("fixtures/pydantic_ai_simple.json"))]
#[case::pydantic_ai_swarm(include_bytes!("fixtures/pydantic_ai_swarm.json"))]
#[case::pydantic_ai_token_limit_swarm(include_bytes!("fixtures/pydantic_ai_token_limit_swarm.json"))]
#[case::query_alternate(include_bytes!("fixtures/query_alternate.json"))]
#[case::query_children(include_bytes!("fixtures/query_children.json"))]
#[case::query_other_team(include_bytes!("fixtures/query_other_team.json"))]
#[case::query_root(include_bytes!("fixtures/query_root.json"))]
#[case::strands_simple(include_bytes!("fixtures/strands_simple.json"))]
#[case::strands_swarm(include_bytes!("fixtures/strands_swarm.json"))]
#[case::vercel_ai_sdk_simple(include_bytes!("fixtures/vercel_ai_sdk_simple.json"))]
#[case::vercel_ai_sdk_swarm(include_bytes!("fixtures/vercel_ai_sdk_swarm.json"))]
#[case::google_adk_stream(include_bytes!("fixtures/google_adk_stream.json"))]
#[case::google_adk_retry(include_bytes!("fixtures/google_adk_retry.json"))]
#[case::google_adk_billed_failure(include_bytes!("fixtures/google_adk_billed_failure.json"))]
#[case::pydantic_ai_stream(include_bytes!("fixtures/pydantic_ai_stream.json"))]
#[case::pydantic_ai_swarm_stream(include_bytes!("fixtures/pydantic_ai_swarm_stream.json"))]
#[case::pydantic_ai_retry(include_bytes!("fixtures/pydantic_ai_retry.json"))]
#[case::pydantic_ai_billed_failure(include_bytes!("fixtures/pydantic_ai_billed_failure.json"))]
#[case::strands_retry(include_bytes!("fixtures/strands_retry.json"))]
#[case::vercel_ai_sdk_stream(include_bytes!("fixtures/vercel_ai_sdk_stream.json"))]
#[case::vercel_ai_sdk_retry(include_bytes!("fixtures/vercel_ai_sdk_retry.json"))]
#[case::vercel_ai_sdk_billed_failure(include_bytes!("fixtures/vercel_ai_sdk_billed_failure.json"))]
#[case::strands_billed_failure(include_bytes!("fixtures/strands_billed_failure.json"))]
#[case::mastra_simple(include_bytes!("fixtures/mastra_simple.json"))]
#[case::mastra_swarm(include_bytes!("fixtures/mastra_swarm.json"))]
#[case::vercel_ai_sdk_py_simple(include_bytes!("fixtures/vercel_ai_sdk_py_simple.json"))]
#[case::vercel_ai_sdk_py_swarm(include_bytes!("fixtures/vercel_ai_sdk_py_swarm.json"))]
fn fixture_normalization(#[case] body: &[u8]) {
    let spans = decode_otlp(body, Some("application/json")).expect("captured OTLP export");
    assert!(!spans.is_empty());
    for span in &spans {
        assert_invariants(span);
    }
    let document: Value = serde_json::from_slice(body).expect("fixture JSON");
    let recorded_count = array(&document, "resourceSpans")
        .iter()
        .flat_map(|resource| array(resource, "scopeSpans"))
        .flat_map(|scope| array(scope, "spans"))
        .count();
    assert_eq!(spans.len(), recorded_count);
}

#[rstest]
#[case::claude_llm(include_bytes!("fixtures/claude_agent_sdk_simple.json"), "claude_code.llm_request", ObservationType::Llm, false)]
#[case::openinference_llm(include_bytes!("fixtures/opentelemetry_simple.json"), "ChatCompletion", ObservationType::Llm, false)]
#[case::langchain_llm(include_bytes!("fixtures/langchain_simple.json"), "ChatOpenAI", ObservationType::Llm, false)]
#[case::llamaindex_llm(include_bytes!("fixtures/llamaindex_simple.json"), "OpenAILike.achat", ObservationType::Llm, false)]
#[case::google_llm(include_bytes!("fixtures/google_adk_simple.json"), "call_llm", ObservationType::Llm, false)]
#[case::openai_llm(include_bytes!("fixtures/openai_agents_simple.json"), "response", ObservationType::Llm, false)]
#[case::strands_llm(include_bytes!("fixtures/strands_simple.json"), "chat", ObservationType::Llm, false)]
#[case::crewai_wrapper(include_bytes!("fixtures/crewai_simple.json"), "research_crew.kickoff", ObservationType::Agent, true)]
#[case::claude_interaction_wrapper(include_bytes!("fixtures/claude_agent_sdk_simple.json"), "claude_code.interaction", ObservationType::Agent, true)]
#[case::claude_delegation_wrapper(include_bytes!("fixtures/claude_agent_sdk_swarm.json"), "ClaudeAgentSDK.Agent", ObservationType::Agent, true)]
#[case::claude_hook(include_bytes!("fixtures/claude_agent_sdk_detailed_export.json"), "claude_code.hook", ObservationType::Framework, false)]
#[case::deepagents_middleware(include_bytes!("fixtures/deepagents_simple.json"), "PatchToolCallsMiddleware.before_agent", ObservationType::Framework, false)]
#[case::langsmith_middleware(include_bytes!("fixtures/langsmith_deep_agent_export.json"), "FilesystemMiddleware.wrap_model_call", ObservationType::Framework, false)]
#[case::google_invocation_wrapper(include_bytes!("fixtures/google_adk_simple.json"), "invocation [research_app]", ObservationType::Agent, true)]
#[case::llamaindex_preparation(include_bytes!("fixtures/llamaindex_simple.json"), "OpenAILike._prepare_chat_with_tools", ObservationType::Chain, false)]
#[case::llamaindex_agent_step(include_bytes!("fixtures/llamaindex_simple.json"), "BaseWorkflowAgent.run_agent_step", ObservationType::Agent, false)]
#[case::llamaindex_run_wrapper(include_bytes!("fixtures/llamaindex_simple.json"), "FunctionAgent.run", ObservationType::Agent, true)]
#[case::openai_agent(include_bytes!("fixtures/openai_agents_simple.json"), "research_agent", ObservationType::Agent, false)]
#[case::pydantic_tool(include_bytes!("fixtures/pydantic_ai_swarm.json"), "execute_tool search", ObservationType::Tool, false)]
#[case::strands_cycle(include_bytes!("fixtures/strands_simple.json"), "execute_event_loop_cycle", ObservationType::Chain, false)]
#[case::vercel_step(include_bytes!("fixtures/vercel_ai_sdk_simple.json"), "step 1", ObservationType::Chain, false)]
#[case::vercel_py_llm(include_bytes!("fixtures/vercel_ai_sdk_py_simple.json"), "chat openai/gpt-6-luna", ObservationType::Llm, false)]
#[case::mastra_llm(include_bytes!("fixtures/mastra_simple.json"), "chat openai/gpt-6-luna", ObservationType::Llm, false)]
#[case::mastra_agent(include_bytes!("fixtures/mastra_simple.json"), "invoke_agent research_agent", ObservationType::Agent, false)]
fn fixture_sdk_roles(
    #[case] body: &[u8],
    #[case] name: &str,
    #[case] expected: ObservationType,
    #[case] wrapper_candidate: bool,
) {
    let spans = decode_otlp(body, Some("application/json")).expect("captured OTLP export");
    let matching: Vec<_> = spans.iter().filter(|span| span.name == name).collect();
    assert!(!matching.is_empty(), "fixture has no {name} span");
    for span in matching {
        assert_eq!(
            span.normalized.observation_type, expected,
            "{}",
            span.span_id
        );
        assert_eq!(
            span.normalized.wrapper_candidate, wrapper_candidate,
            "{}",
            span.span_id
        );
    }
}

#[rstest]
#[case::simple(include_bytes!("fixtures/llamaindex_simple.json"))]
#[case::swarm(include_bytes!("fixtures/llamaindex_swarm.json"))]
fn llamaindex_wrapped_responses_keep_provider_call_keys(#[case] body: &[u8]) {
    let spans = decode_otlp(body, Some("application/json")).unwrap();
    let responses: Vec<_> = spans
        .iter()
        .filter_map(|span| {
            let response: Value =
                serde_json::from_str(span.attributes.get("output.value")?).ok()?;
            let id = response.get("raw")?.get("id")?.as_str()?.to_owned();
            Some((span, id))
        })
        .collect();
    assert!(!responses.is_empty());
    for (span, id) in responses {
        assert!(
            span.normalized
                .calls
                .key_set()
                .unwrap()
                .contains(&litellm_traces::CallKey::ProviderResponse(id))
        );
    }
}

#[rstest]
#[case::request(litellm_traces::CallKey::LiteLlmRequest("request:with:colons".to_owned()))]
#[case::response(litellm_traces::CallKey::ProviderResponse("response:with:colons".to_owned()))]
#[case::transport(litellm_traces::CallKey::Transport)]
#[case::gateway_attempt(litellm_traces::CallKey::GatewayAttempt)]
fn call_keys_round_trip_through_storage(#[case] key: litellm_traces::CallKey) {
    assert_eq!(
        key.to_string().parse::<litellm_traces::CallKey>().unwrap(),
        key
    );
    let encoded = serde_json::to_string(&key).unwrap();
    assert_eq!(
        serde_json::from_str::<litellm_traces::CallKey>(&encoded).unwrap(),
        key
    );
}

#[rstest]
#[case::missing_separator("provider_response")]
#[case::missing_response("provider_response:")]
#[case::missing_request("litellm_request:")]
#[case::transport_id("transport:unexpected")]
#[case::gateway_attempt_separator("gateway_attempt")]
#[case::gateway_attempt_id("gateway_attempt:unexpected")]
#[case::unknown("unknown:id")]
fn malformed_call_keys_are_rejected_at_the_boundary(#[case] encoded: &str) {
    assert!(encoded.parse::<litellm_traces::CallKey>().is_err());
    assert!(serde_json::from_value::<litellm_traces::CallKey>(serde_json::json!(encoded)).is_err());
}
