use litellm_traces::{DecodedSpan, ObservationType, decode_otlp};
use rstest::rstest;
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct Projection<'a> {
    scope: &'a str,
    name: &'a str,
    display_name: &'a str,
    root: bool,
    observation_type: ObservationType,
    wrapper_candidate: bool,
    agent: &'a str,
    framework: &'a str,
    model: &'a str,
    input_tokens: u32,
    output_tokens: u32,
    call_evidence: &'a str,
    call_keys: Vec<String>,
    tool_call_id: &'a str,
    input_preview: &'a str,
}

fn projection<'a>(span: &'a DecodedSpan, name: &'a str) -> Projection<'a> {
    let normalized = &span.normalized;
    Projection {
        scope: span.scope_name.as_ref(),
        name,
        display_name: &span.name,
        root: span.parent_span_id.is_empty(),
        observation_type: normalized.observation_type,
        wrapper_candidate: normalized.wrapper_candidate,
        agent: &normalized.agent_name,
        framework: &normalized.framework,
        model: &normalized.model,
        input_tokens: normalized.input_tokens,
        output_tokens: normalized.output_tokens,
        call_evidence: normalized.call_evidence,
        call_keys: normalized
            .call_keys
            .iter()
            .map(ToString::to_string)
            .collect(),
        tool_call_id: &normalized.tool_call_id,
        input_preview: &normalized.input_preview,
    }
}

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
                .call_keys
                .iter()
                .any(|key| key.to_string() == format!("provider_response:{id}"))
        );
        assert_ne!(normalized.call_evidence, "unknown");
    }
    if normalized.call_evidence == "unknown" {
        assert!(normalized.call_keys.is_empty());
    } else {
        assert!(!normalized.call_keys.is_empty());
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
#[case::claude_agent_sdk_detailed_export("claude_agent_sdk_detailed_export", include_bytes!("fixtures/claude_agent_sdk_detailed_export.json"))]
#[case::claude_agent_sdk_export("claude_agent_sdk_export", include_bytes!("fixtures/claude_agent_sdk_export.json"))]
#[case::claude_agent_sdk_simple("claude_agent_sdk_simple", include_bytes!("fixtures/claude_agent_sdk_simple.json"))]
#[case::claude_agent_sdk_swarm("claude_agent_sdk_swarm", include_bytes!("fixtures/claude_agent_sdk_swarm.json"))]
#[case::crewai_simple("crewai_simple", include_bytes!("fixtures/crewai_simple.json"))]
#[case::crewai_swarm("crewai_swarm", include_bytes!("fixtures/crewai_swarm.json"))]
#[case::deepagents_simple("deepagents_simple", include_bytes!("fixtures/deepagents_simple.json"))]
#[case::deepagents_swarm("deepagents_swarm", include_bytes!("fixtures/deepagents_swarm.json"))]
#[case::deeplite_auth_error("deeplite_auth_error", include_bytes!("fixtures/deeplite_auth_error.json"))]
#[case::deeplite_swarm("deeplite_swarm", include_bytes!("fixtures/deeplite_swarm.json"))]
#[case::google_adk_simple("google_adk_simple", include_bytes!("fixtures/google_adk_simple.json"))]
#[case::google_adk_swarm("google_adk_swarm", include_bytes!("fixtures/google_adk_swarm.json"))]
#[case::langchain_simple("langchain_simple", include_bytes!("fixtures/langchain_simple.json"))]
#[case::langchain_swarm("langchain_swarm", include_bytes!("fixtures/langchain_swarm.json"))]
#[case::langgraph_simple("langgraph_simple", include_bytes!("fixtures/langgraph_simple.json"))]
#[case::langgraph_swarm("langgraph_swarm", include_bytes!("fixtures/langgraph_swarm.json"))]
#[case::langsmith_deep_agent_export("langsmith_deep_agent_export", include_bytes!("fixtures/langsmith_deep_agent_export.json"))]
#[case::llamaindex_simple("llamaindex_simple", include_bytes!("fixtures/llamaindex_simple.json"))]
#[case::llamaindex_swarm("llamaindex_swarm", include_bytes!("fixtures/llamaindex_swarm.json"))]
#[case::openai_agents_simple("openai_agents_simple", include_bytes!("fixtures/openai_agents_simple.json"))]
#[case::openai_agents_swarm("openai_agents_swarm", include_bytes!("fixtures/openai_agents_swarm.json"))]
#[case::opentelemetry_simple("opentelemetry_simple", include_bytes!("fixtures/opentelemetry_simple.json"))]
#[case::opentelemetry_swarm("opentelemetry_swarm", include_bytes!("fixtures/opentelemetry_swarm.json"))]
#[case::pydantic_ai_simple("pydantic_ai_simple", include_bytes!("fixtures/pydantic_ai_simple.json"))]
#[case::pydantic_ai_swarm("pydantic_ai_swarm", include_bytes!("fixtures/pydantic_ai_swarm.json"))]
#[case::query_alternate("query_alternate", include_bytes!("fixtures/query_alternate.json"))]
#[case::query_children("query_children", include_bytes!("fixtures/query_children.json"))]
#[case::query_other_team("query_other_team", include_bytes!("fixtures/query_other_team.json"))]
#[case::query_root("query_root", include_bytes!("fixtures/query_root.json"))]
#[case::strands_simple("strands_simple", include_bytes!("fixtures/strands_simple.json"))]
#[case::strands_swarm("strands_swarm", include_bytes!("fixtures/strands_swarm.json"))]
#[case::vercel_ai_sdk_simple("vercel_ai_sdk_simple", include_bytes!("fixtures/vercel_ai_sdk_simple.json"))]
#[case::vercel_ai_sdk_swarm("vercel_ai_sdk_swarm", include_bytes!("fixtures/vercel_ai_sdk_swarm.json"))]
fn fixture_normalization(#[case] name: &str, #[case] body: &[u8]) {
    let spans = decode_otlp(body, Some("application/json")).expect("captured OTLP export");
    assert!(!spans.is_empty());
    for span in &spans {
        assert_invariants(span);
    }
    let document: Value = serde_json::from_slice(body).expect("fixture JSON");
    let names: Vec<_> = array(&document, "resourceSpans")
        .iter()
        .flat_map(|resource| array(resource, "scopeSpans"))
        .flat_map(|scope| array(scope, "spans"))
        .map(|span| {
            span.get("name")
                .and_then(Value::as_str)
                .expect("fixture span name")
        })
        .collect();
    assert_eq!(spans.len(), names.len());
    let projected: Vec<_> = spans
        .iter()
        .zip(names)
        .map(|(span, name)| projection(span, name))
        .collect();
    insta::with_settings!({prepend_module_to_snapshot => false}, {
        insta::assert_json_snapshot!(format!("normalize__{name}"), projected);
    });
}
