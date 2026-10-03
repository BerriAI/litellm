use std::collections::{BTreeMap, BTreeSet};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE, URL_SAFE_NO_PAD},
};
use litellm_traces::{
    CallEvidence, CallEvidenceKind, CallKey, DecodedSpan, ObservationType, SpanStatus, decode_otlp,
    query::named::{SpendByResponseIdsRow, TraceSpansRow},
    resolve_trace,
};
use rstest::rstest;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
struct CaptureData {
    otlp: &'static [u8],
    spend_logs: &'static str,
}

fn capture_data(name: &str) -> CaptureData {
    match name {
        "claude_agent_sdk_missing_request_id_simple" => CaptureData {
            otlp: include_bytes!("fixtures/claude_agent_sdk_missing_request_id_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/claude_agent_sdk_missing_request_id_simple_spend_logs.jsonl"
            ),
        },
        "claude_agent_sdk_missing_request_id_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/claude_agent_sdk_missing_request_id_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/claude_agent_sdk_missing_request_id_swarm_spend_logs.jsonl"
            ),
        },
        "claude_agent_sdk_simple" => CaptureData {
            otlp: include_bytes!("fixtures/claude_agent_sdk_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/claude_agent_sdk_simple_spend_logs.jsonl"
            ),
        },
        "claude_agent_sdk_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/claude_agent_sdk_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/claude_agent_sdk_swarm_spend_logs.jsonl"
            ),
        },
        "crewai_simple" => CaptureData {
            otlp: include_bytes!("fixtures/crewai_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/crewai_simple_spend_logs.jsonl"
            ),
        },
        "crewai_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/crewai_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/crewai_swarm_spend_logs.jsonl"
            ),
        },
        "deepagents_simple" => CaptureData {
            otlp: include_bytes!("fixtures/deepagents_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/deepagents_simple_spend_logs.jsonl"
            ),
        },
        "deepagents_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/deepagents_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/deepagents_swarm_spend_logs.jsonl"
            ),
        },
        "google_adk_billed_failure" => CaptureData {
            otlp: include_bytes!("fixtures/google_adk_billed_failure.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/google_adk_billed_failure_spend_logs.jsonl"
            ),
        },
        "google_adk_retry" => CaptureData {
            otlp: include_bytes!("fixtures/google_adk_retry.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/google_adk_retry_spend_logs.jsonl"
            ),
        },
        "google_adk_simple" => CaptureData {
            otlp: include_bytes!("fixtures/google_adk_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/google_adk_simple_spend_logs.jsonl"
            ),
        },
        "google_adk_stream" => CaptureData {
            otlp: include_bytes!("fixtures/google_adk_stream.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/google_adk_stream_spend_logs.jsonl"
            ),
        },
        "google_adk_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/google_adk_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/google_adk_swarm_spend_logs.jsonl"
            ),
        },
        "langchain_simple" => CaptureData {
            otlp: include_bytes!("fixtures/langchain_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/langchain_simple_spend_logs.jsonl"
            ),
        },
        "langchain_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/langchain_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/langchain_swarm_spend_logs.jsonl"
            ),
        },
        "langgraph_simple" => CaptureData {
            otlp: include_bytes!("fixtures/langgraph_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/langgraph_simple_spend_logs.jsonl"
            ),
        },
        "langgraph_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/langgraph_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/langgraph_swarm_spend_logs.jsonl"
            ),
        },
        "llamaindex_simple" => CaptureData {
            otlp: include_bytes!("fixtures/llamaindex_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/llamaindex_simple_spend_logs.jsonl"
            ),
        },
        "llamaindex_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/llamaindex_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/llamaindex_swarm_spend_logs.jsonl"
            ),
        },
        "mastra_simple" => CaptureData {
            otlp: include_bytes!("fixtures/mastra_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/mastra_simple_spend_logs.jsonl"
            ),
        },
        "mastra_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/mastra_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/mastra_swarm_spend_logs.jsonl"
            ),
        },
        "openai_agents_simple" => CaptureData {
            otlp: include_bytes!("fixtures/openai_agents_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/openai_agents_simple_spend_logs.jsonl"
            ),
        },
        "openai_agents_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/openai_agents_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/openai_agents_swarm_spend_logs.jsonl"
            ),
        },
        "opentelemetry_simple" => CaptureData {
            otlp: include_bytes!("fixtures/opentelemetry_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/opentelemetry_simple_spend_logs.jsonl"
            ),
        },
        "opentelemetry_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/opentelemetry_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/opentelemetry_swarm_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_billed_failure" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_billed_failure.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_billed_failure_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_retry" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_retry.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_retry_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_simple" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_simple_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_stream" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_stream.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_stream_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_swarm_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_swarm_stream" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_swarm_stream.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_swarm_stream_spend_logs.jsonl"
            ),
        },
        "pydantic_ai_token_limit_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/pydantic_ai_token_limit_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/pydantic_ai_token_limit_swarm_spend_logs.jsonl"
            ),
        },
        "strands_billed_failure" => CaptureData {
            otlp: include_bytes!("fixtures/strands_billed_failure.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/strands_billed_failure_spend_logs.jsonl"
            ),
        },
        "strands_retry" => CaptureData {
            otlp: include_bytes!("fixtures/strands_retry.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/strands_retry_spend_logs.jsonl"
            ),
        },
        "strands_simple" => CaptureData {
            otlp: include_bytes!("fixtures/strands_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/strands_simple_spend_logs.jsonl"
            ),
        },
        "strands_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/strands_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/strands_swarm_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_billed_failure" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_billed_failure.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_billed_failure_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_py_simple" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_py_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_py_simple_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_py_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_py_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_py_swarm_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_retry" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_retry.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_retry_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_simple" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_simple.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_simple_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_stream" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_stream.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_stream_spend_logs.jsonl"
            ),
        },
        "vercel_ai_sdk_swarm" => CaptureData {
            otlp: include_bytes!("fixtures/vercel_ai_sdk_swarm.json"),
            spend_logs: include_str!(
                "../../traces-clickhouse/tests/fixtures/vercel_ai_sdk_swarm_spend_logs.jsonl"
            ),
        },
        _ => panic!("unknown trace capture {name}"),
    }
}

#[derive(Deserialize)]
struct CapturedSpend {
    request_id: String,
    #[serde(default)]
    litellm_call_id: String,
    response_id: String,
    trace_id: String,
    span_id: String,
    team_id: String,
    api_key: String,
    user: String,
    spend: Option<f64>,
    start_time: i64,
    metadata: String,
}

#[derive(Deserialize)]
struct SpendMetadata {
    fixture_capture: FixtureCapture,
}

#[derive(Deserialize)]
struct FixtureCapture {
    name: String,
    trace_id: String,
    spend_linked: bool,
    #[serde(default = "true_value")]
    spend_complete: bool,
}

fn true_value() -> bool {
    true
}

fn upstream_response_id(response_id: &str) -> String {
    let Some(encoded) = response_id.strip_prefix("resp_") else {
        return String::new();
    };
    STANDARD
        .decode(encoded)
        .or_else(|_| URL_SAFE.decode(encoded))
        .or_else(|_| URL_SAFE_NO_PAD.decode(encoded))
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|decoded| {
            let (_, response_id) = decoded.split_once("response_id:")?;
            Some(response_id.split(';').next()?.to_owned())
        })
        .unwrap_or_default()
}

fn captured_spend_rows(spend_logs: &str) -> (FixtureCapture, Vec<SpendByResponseIdsRow>) {
    let records: Vec<CapturedSpend> = spend_logs
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("valid spend fixture row"))
        .collect();
    let metadata: SpendMetadata =
        serde_json::from_str(&records.first().expect("spend fixture rows").metadata)
            .expect("valid spend fixture metadata");
    let spends = records
        .into_iter()
        .map(|record| {
            let upstream_response_id = upstream_response_id(&record.response_id);
            SpendByResponseIdsRow {
                request_id: record.request_id,
                litellm_call_id: record.litellm_call_id,
                response_id: record.response_id,
                upstream_response_id,
                trace_id: record.trace_id,
                span_id: record.span_id,
                team_id: record.team_id,
                api_key: record.api_key,
                user: record.user,
                spend: record.spend,
                start_ms: record.start_time,
            }
        })
        .collect();
    (metadata.fixture_capture, spends)
}

fn status_message(span: &DecodedSpan) -> &str {
    if !span.status_message.is_empty() {
        return &span.status_message;
    }
    span.events
        .iter()
        .find(|event| event.name == "exception")
        .and_then(|event| {
            event
                .attributes
                .get("exception.message")
                .filter(|message| !message.is_empty())
                .or_else(|| event.attributes.get("exception.type"))
        })
        .map(String::as_str)
        .unwrap_or_default()
}

fn trace_span(span: DecodedSpan) -> TraceSpansRow {
    let service = span
        .resource_attributes
        .get("service.name")
        .cloned()
        .unwrap_or_default();
    let message = status_message(&span);
    let error_truncated = message.chars().count() > 128;
    let status_message = message.chars().take(128).collect();
    let normalized = span.normalized;
    let call_keys = normalized
        .calls
        .key_set()
        .into_iter()
        .flatten()
        .cloned()
        .collect();
    let litellm_request_id = normalized
        .calls
        .key_set()
        .into_iter()
        .flatten()
        .find_map(|key| match key {
            CallKey::ProviderResponse(id) => Some(id.clone()),
            CallKey::LiteLlmRequest(_) | CallKey::Transport => None,
        })
        .unwrap_or_default();
    TraceSpansRow {
        trace_id: span.trace_id,
        span_id: span.span_id,
        parent_span_id: span.parent_span_id,
        name: span.name,
        kind: normalized.observation_type,
        wrapper_candidate: normalized.wrapper_candidate,
        agent: normalized.agent_name.unwrap_or_default(),
        framework: normalized
            .framework
            .map(|framework| framework.to_string())
            .unwrap_or_default(),
        status: match span.status_code.as_str() {
            "STATUS_CODE_OK" => SpanStatus::Ok,
            "STATUS_CODE_ERROR" => SpanStatus::Error,
            _ => SpanStatus::Unset,
        },
        status_message,
        error_truncated,
        start_ns: i64::try_from(span.start_ns).expect("valid trace start timestamp"),
        duration_ns: span.end_ns - span.start_ns,
        service,
        input_preview: normalized.input_preview,
        model: normalized.model.unwrap_or_default(),
        input_tokens: normalized.input_tokens,
        output_tokens: normalized.output_tokens,
        litellm_request_id,
        call_keys,
        call_evidence: Some(normalized.calls.kind()),
        tool_call_id: normalized.tool_call_id.unwrap_or_default(),
        team_id: "fixture-team".into(),
        api_key_hash: "fixture-key".into(),
        user_id: "fixture-user".into(),
    }
}

fn trace_rows(otlp: &[u8]) -> Vec<TraceSpansRow> {
    let mut seen = BTreeSet::new();
    decode_otlp(otlp, Some("application/json"))
        .expect("valid OTLP fixture")
        .into_iter()
        .filter_map(|span| seen.insert(span.span_id.clone()).then(|| trace_span(span)))
        .collect()
}

fn fixture(
    name: &str,
) -> (
    CaptureData,
    FixtureCapture,
    Vec<TraceSpansRow>,
    Vec<SpendByResponseIdsRow>,
) {
    let data = capture_data(name);
    let (capture, spends) = captured_spend_rows(data.spend_logs);
    assert_eq!(capture.name, name);
    let rows = trace_rows(data.otlp);
    (data, capture, rows, spends)
}

fn assert_spend_close(actual: Option<f64>, expected: Option<f64>, capture: &str) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert!(
            (actual - expected).abs() <= 1e-12,
            "{capture}: expected {expected}, got {actual}"
        ),
        _ => assert_eq!(actual, expected, "{capture}"),
    }
}

fn agent_spends(trace: &litellm_traces::Trace) -> BTreeMap<String, Option<f64>> {
    trace
        .agents
        .iter()
        .map(|agent| (agent.name.clone(), agent.spend))
        .collect()
}

fn clone_trace_row(row: &TraceSpansRow) -> TraceSpansRow {
    TraceSpansRow {
        trace_id: row.trace_id.clone(),
        span_id: row.span_id.clone(),
        parent_span_id: row.parent_span_id.clone(),
        name: row.name.clone(),
        kind: row.kind,
        wrapper_candidate: row.wrapper_candidate,
        agent: row.agent.clone(),
        framework: row.framework.clone(),
        status: row.status,
        status_message: row.status_message.clone(),
        error_truncated: row.error_truncated,
        start_ns: row.start_ns,
        duration_ns: row.duration_ns,
        service: row.service.clone(),
        input_preview: row.input_preview.clone(),
        model: row.model.clone(),
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        litellm_request_id: row.litellm_request_id.clone(),
        call_keys: row.call_keys.clone(),
        call_evidence: row.call_evidence,
        tool_call_id: row.tool_call_id.clone(),
        team_id: row.team_id.clone(),
        api_key_hash: row.api_key_hash.clone(),
        user_id: row.user_id.clone(),
    }
}

fn unrelated_transport(call: &TraceSpansRow) -> TraceSpansRow {
    let start_ns =
        i64::try_from(i128::from(call.start_ns) + i128::from(call.duration_ns) + 1_000_000)
            .expect("valid unrelated transport timestamp");
    TraceSpansRow {
        trace_id: call.trace_id.clone(),
        span_id: format!("unrelated-transport-{}", call.span_id),
        parent_span_id: call.parent_span_id.clone(),
        name: "unrelated-http".into(),
        kind: ObservationType::Framework,
        wrapper_candidate: false,
        agent: String::new(),
        framework: String::new(),
        status: SpanStatus::Ok,
        status_message: String::new(),
        error_truncated: false,
        start_ns,
        duration_ns: 1_000_000,
        service: call.service.clone(),
        input_preview: String::new(),
        model: String::new(),
        input_tokens: 0,
        output_tokens: 0,
        litellm_request_id: String::new(),
        call_keys: vec![CallKey::Transport],
        call_evidence: Some(CallEvidenceKind::Complete),
        tool_call_id: String::new(),
        team_id: call.team_id.clone(),
        api_key_hash: call.api_key_hash.clone(),
        user_id: call.user_id.clone(),
    }
}

fn append_response_id(document: &mut Value, trace_id: &str, span_id: &str, response_id: &str) {
    let resources = document["resourceSpans"]
        .as_array_mut()
        .expect("OTLP resource spans");
    for resource in resources {
        let scopes = resource["scopeSpans"]
            .as_array_mut()
            .expect("OTLP scope spans");
        for scope in scopes {
            let spans = scope["spans"].as_array_mut().expect("OTLP spans");
            for span in spans {
                let matches = span.get("traceId").and_then(Value::as_str) == Some(trace_id)
                    && span.get("spanId").and_then(Value::as_str) == Some(span_id);
                if !matches {
                    continue;
                }
                let attributes = span
                    .as_object_mut()
                    .expect("OTLP span object")
                    .entry("attributes")
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .expect("OTLP span attributes");
                attributes.push(json!({
                    "key": "gen_ai.response.id",
                    "value": { "stringValue": response_id },
                }));
                return;
            }
        }
    }
    panic!("missing OTLP span {trace_id}/{span_id}");
}

#[rstest]
#[case::claude_agent_sdk_missing_request_id_simple("claude_agent_sdk_missing_request_id_simple")]
#[case::claude_agent_sdk_missing_request_id_swarm("claude_agent_sdk_missing_request_id_swarm")]
#[case::claude_agent_sdk_simple("claude_agent_sdk_simple")]
#[case::claude_agent_sdk_swarm("claude_agent_sdk_swarm")]
#[case::crewai_simple("crewai_simple")]
#[case::crewai_swarm("crewai_swarm")]
#[case::deepagents_simple("deepagents_simple")]
#[case::deepagents_swarm("deepagents_swarm")]
#[case::google_adk_billed_failure("google_adk_billed_failure")]
#[case::google_adk_retry("google_adk_retry")]
#[case::google_adk_simple("google_adk_simple")]
#[case::google_adk_stream("google_adk_stream")]
#[case::google_adk_swarm("google_adk_swarm")]
#[case::langchain_simple("langchain_simple")]
#[case::langchain_swarm("langchain_swarm")]
#[case::langgraph_simple("langgraph_simple")]
#[case::langgraph_swarm("langgraph_swarm")]
#[case::llamaindex_simple("llamaindex_simple")]
#[case::llamaindex_swarm("llamaindex_swarm")]
#[case::mastra_simple("mastra_simple")]
#[case::mastra_swarm("mastra_swarm")]
#[case::openai_agents_simple("openai_agents_simple")]
#[case::openai_agents_swarm("openai_agents_swarm")]
#[case::opentelemetry_simple("opentelemetry_simple")]
#[case::opentelemetry_swarm("opentelemetry_swarm")]
#[case::pydantic_ai_billed_failure("pydantic_ai_billed_failure")]
#[case::pydantic_ai_retry("pydantic_ai_retry")]
#[case::pydantic_ai_simple("pydantic_ai_simple")]
#[case::pydantic_ai_stream("pydantic_ai_stream")]
#[case::pydantic_ai_swarm("pydantic_ai_swarm")]
#[case::pydantic_ai_swarm_stream("pydantic_ai_swarm_stream")]
#[case::pydantic_ai_token_limit_swarm("pydantic_ai_token_limit_swarm")]
#[case::strands_billed_failure("strands_billed_failure")]
#[case::strands_retry("strands_retry")]
#[case::strands_simple("strands_simple")]
#[case::strands_swarm("strands_swarm")]
#[case::vercel_ai_sdk_billed_failure("vercel_ai_sdk_billed_failure")]
#[case::vercel_ai_sdk_py_simple("vercel_ai_sdk_py_simple")]
#[case::vercel_ai_sdk_py_swarm("vercel_ai_sdk_py_swarm")]
#[case::vercel_ai_sdk_retry("vercel_ai_sdk_retry")]
#[case::vercel_ai_sdk_simple("vercel_ai_sdk_simple")]
#[case::vercel_ai_sdk_stream("vercel_ai_sdk_stream")]
#[case::vercel_ai_sdk_swarm("vercel_ai_sdk_swarm")]
fn captured_trace_cost_matches_spend_logs(#[case] name: &str) {
    let (_, capture, rows, spends) = fixture(name);
    let trace = resolve_trace(&capture.trace_id, "", &rows, &spends).expect("captured trace");
    let expected = if capture.spend_linked && capture.spend_complete {
        Some(spends.iter().map(|row| row.spend.unwrap_or(0.0)).sum())
    } else {
        None
    };
    assert_spend_close(trace.summary.spend, expected, name);
}

#[rstest]
#[case::claude_agent_sdk_missing_request_id_simple("claude_agent_sdk_missing_request_id_simple")]
#[case::claude_agent_sdk_missing_request_id_swarm("claude_agent_sdk_missing_request_id_swarm")]
#[case::claude_agent_sdk_simple("claude_agent_sdk_simple")]
#[case::claude_agent_sdk_swarm("claude_agent_sdk_swarm")]
#[case::crewai_simple("crewai_simple")]
#[case::crewai_swarm("crewai_swarm")]
#[case::deepagents_simple("deepagents_simple")]
#[case::deepagents_swarm("deepagents_swarm")]
#[case::google_adk_billed_failure("google_adk_billed_failure")]
#[case::google_adk_retry("google_adk_retry")]
#[case::google_adk_simple("google_adk_simple")]
#[case::google_adk_stream("google_adk_stream")]
#[case::google_adk_swarm("google_adk_swarm")]
#[case::langchain_simple("langchain_simple")]
#[case::langchain_swarm("langchain_swarm")]
#[case::langgraph_simple("langgraph_simple")]
#[case::langgraph_swarm("langgraph_swarm")]
#[case::llamaindex_simple("llamaindex_simple")]
#[case::llamaindex_swarm("llamaindex_swarm")]
#[case::mastra_simple("mastra_simple")]
#[case::mastra_swarm("mastra_swarm")]
#[case::openai_agents_simple("openai_agents_simple")]
#[case::openai_agents_swarm("openai_agents_swarm")]
#[case::opentelemetry_simple("opentelemetry_simple")]
#[case::opentelemetry_swarm("opentelemetry_swarm")]
#[case::pydantic_ai_billed_failure("pydantic_ai_billed_failure")]
#[case::pydantic_ai_retry("pydantic_ai_retry")]
#[case::pydantic_ai_simple("pydantic_ai_simple")]
#[case::pydantic_ai_stream("pydantic_ai_stream")]
#[case::pydantic_ai_swarm("pydantic_ai_swarm")]
#[case::pydantic_ai_swarm_stream("pydantic_ai_swarm_stream")]
#[case::pydantic_ai_token_limit_swarm("pydantic_ai_token_limit_swarm")]
#[case::strands_billed_failure("strands_billed_failure")]
#[case::strands_retry("strands_retry")]
#[case::strands_simple("strands_simple")]
#[case::strands_swarm("strands_swarm")]
#[case::vercel_ai_sdk_billed_failure("vercel_ai_sdk_billed_failure")]
#[case::vercel_ai_sdk_py_simple("vercel_ai_sdk_py_simple")]
#[case::vercel_ai_sdk_py_swarm("vercel_ai_sdk_py_swarm")]
#[case::vercel_ai_sdk_retry("vercel_ai_sdk_retry")]
#[case::vercel_ai_sdk_simple("vercel_ai_sdk_simple")]
#[case::vercel_ai_sdk_stream("vercel_ai_sdk_stream")]
#[case::vercel_ai_sdk_swarm("vercel_ai_sdk_swarm")]
fn unrelated_sibling_transport_leaves_cost_unchanged(#[case] name: &str) {
    let (_, capture, rows, spends) = fixture(name);
    let baseline = resolve_trace(&capture.trace_id, "", &rows, &spends).expect("captured trace");
    let baseline_spend = baseline.summary.spend;
    let baseline_agent_spends = agent_spends(&baseline);
    let calls: Vec<_> = rows
        .iter()
        .filter(|row| {
            row.kind == ObservationType::Llm && !row.call_keys.contains(&CallKey::Transport)
        })
        .map(clone_trace_row)
        .collect();
    assert!(!calls.is_empty(), "{name} has no model call rows");
    for call in calls {
        let augmented_rows = rows
            .iter()
            .map(clone_trace_row)
            .chain([unrelated_transport(&call)])
            .collect::<Vec<_>>();
        let augmented =
            resolve_trace(&capture.trace_id, "", &augmented_rows, &spends).expect("captured trace");
        assert_eq!(
            augmented.summary.spend, baseline_spend,
            "{name}, model span {}",
            call.span_id
        );
        assert_eq!(
            agent_spends(&augmented),
            baseline_agent_spends,
            "{name}, model span {}",
            call.span_id
        );
    }
}

#[rstest]
#[case::claude_agent_sdk_missing_request_id_simple("claude_agent_sdk_missing_request_id_simple")]
#[case::claude_agent_sdk_missing_request_id_swarm("claude_agent_sdk_missing_request_id_swarm")]
#[case::claude_agent_sdk_simple("claude_agent_sdk_simple")]
#[case::claude_agent_sdk_swarm("claude_agent_sdk_swarm")]
#[case::crewai_simple("crewai_simple")]
#[case::crewai_swarm("crewai_swarm")]
#[case::deepagents_simple("deepagents_simple")]
#[case::deepagents_swarm("deepagents_swarm")]
#[case::google_adk_billed_failure("google_adk_billed_failure")]
#[case::google_adk_retry("google_adk_retry")]
#[case::google_adk_simple("google_adk_simple")]
#[case::google_adk_stream("google_adk_stream")]
#[case::google_adk_swarm("google_adk_swarm")]
#[case::langchain_simple("langchain_simple")]
#[case::langchain_swarm("langchain_swarm")]
#[case::langgraph_simple("langgraph_simple")]
#[case::langgraph_swarm("langgraph_swarm")]
#[case::llamaindex_simple("llamaindex_simple")]
#[case::llamaindex_swarm("llamaindex_swarm")]
#[case::mastra_simple("mastra_simple")]
#[case::mastra_swarm("mastra_swarm")]
#[case::openai_agents_simple("openai_agents_simple")]
#[case::openai_agents_swarm("openai_agents_swarm")]
#[case::opentelemetry_simple("opentelemetry_simple")]
#[case::opentelemetry_swarm("opentelemetry_swarm")]
#[case::pydantic_ai_billed_failure("pydantic_ai_billed_failure")]
#[case::pydantic_ai_retry("pydantic_ai_retry")]
#[case::pydantic_ai_simple("pydantic_ai_simple")]
#[case::pydantic_ai_stream("pydantic_ai_stream")]
#[case::pydantic_ai_swarm("pydantic_ai_swarm")]
#[case::pydantic_ai_swarm_stream("pydantic_ai_swarm_stream")]
#[case::pydantic_ai_token_limit_swarm("pydantic_ai_token_limit_swarm")]
#[case::strands_billed_failure("strands_billed_failure")]
#[case::strands_retry("strands_retry")]
#[case::strands_simple("strands_simple")]
#[case::strands_swarm("strands_swarm")]
#[case::vercel_ai_sdk_billed_failure("vercel_ai_sdk_billed_failure")]
#[case::vercel_ai_sdk_py_simple("vercel_ai_sdk_py_simple")]
#[case::vercel_ai_sdk_py_swarm("vercel_ai_sdk_py_swarm")]
#[case::vercel_ai_sdk_retry("vercel_ai_sdk_retry")]
#[case::vercel_ai_sdk_simple("vercel_ai_sdk_simple")]
#[case::vercel_ai_sdk_stream("vercel_ai_sdk_stream")]
#[case::vercel_ai_sdk_swarm("vercel_ai_sdk_swarm")]
fn redundant_genai_response_id_keeps_call_evidence(#[case] name: &str) {
    let data = capture_data(name);
    let decoded = decode_otlp(data.otlp, Some("application/json")).expect("valid OTLP fixture");
    let targets: Vec<_> = decoded
        .iter()
        .filter_map(|span| {
            let CallEvidence::Complete(keys) = &span.normalized.calls else {
                return None;
            };
            if span.attributes.contains_key("gen_ai.response.id") {
                return None;
            }
            let response_ids: Vec<_> = keys
                .iter()
                .filter_map(|key| match key {
                    CallKey::ProviderResponse(id) => Some(id.clone()),
                    CallKey::LiteLlmRequest(_) | CallKey::Transport => None,
                })
                .collect();
            (!response_ids.is_empty()).then(|| {
                (
                    span.trace_id.clone(),
                    span.span_id.clone(),
                    span.normalized.calls.clone(),
                    response_ids,
                )
            })
        })
        .collect();
    for (trace_id, span_id, expected, response_ids) in targets {
        for response_id in response_ids {
            let mut document: Value =
                serde_json::from_slice(data.otlp).expect("valid OTLP JSON fixture");
            append_response_id(&mut document, &trace_id, &span_id, &response_id);
            let modified = serde_json::to_vec(&document).expect("serializable OTLP JSON");
            let spans = decode_otlp(&modified, Some("application/json"))
                .expect("OTLP with redundant response ID");
            let actual = spans
                .iter()
                .find(|span| span.trace_id == trace_id && span.span_id == span_id)
                .expect("modified span")
                .normalized
                .calls
                .clone();
            assert_eq!(
                actual, expected,
                "{name}, span {span_id}, response id {response_id}"
            );
        }
    }
}
