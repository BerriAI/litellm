use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use litellm_traces::{
    CallEvidence, CallEvidenceKind, CallKey, DecodedSpan, ObservationType, SpanStatus, decode_otlp,
    query::named::{SpendByResponseIdsRow, TraceSpansRow},
    resolve_trace,
};
use rstest::rstest;
use serde::Deserialize;
use serde_json::{Value, json};

struct CaptureData {
    otlp: Vec<u8>,
}

fn capture_name(spend_log_path: &Path) -> &str {
    spend_log_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_suffix("_spend_logs"))
        .unwrap_or_else(|| {
            panic!(
                "spend log filename must end with _spend_logs: {}",
                spend_log_path.display()
            )
        })
}

fn manifest_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

fn capture_data(spend_log_path: &Path) -> CaptureData {
    let name = capture_name(spend_log_path);
    let otlp_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.json"));
    let otlp = std::fs::read(&otlp_path).unwrap_or_else(|error| {
        panic!(
            "missing OTLP export for spend capture {name} at {}: {error}",
            otlp_path.display()
        )
    });
    CaptureData { otlp }
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
                provider_request_id: String::new(),
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
            CallKey::ProviderResponse(id) | CallKey::ProviderRequest(id) => Some(id.clone()),
            CallKey::LiteLlmRequest(_) | CallKey::Transport | CallKey::GatewayAttempt => None,
        })
        .unwrap_or_default();
    TraceSpansRow {
        trace_id: span.trace_id,
        original_trace_id: String::new(),
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
        source_type: String::new(),
        source_url: String::new(),
        source_title: String::new(),
        source_user: String::new(),
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
    spend_log_path: &Path,
) -> (
    CaptureData,
    FixtureCapture,
    Vec<TraceSpansRow>,
    Vec<SpendByResponseIdsRow>,
) {
    let spend_log_path = manifest_path(spend_log_path);
    let name = capture_name(&spend_log_path);
    let spend_log_contents = std::fs::read_to_string(&spend_log_path).unwrap_or_else(|error| {
        panic!(
            "unable to read spend log fixture {}: {error}",
            spend_log_path.display()
        )
    });
    let (capture, spends) = captured_spend_rows(&spend_log_contents);
    assert_eq!(capture.name, name);
    let data = capture_data(&spend_log_path);
    let rows = trace_rows(&data.otlp);
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

fn unrelated_transport(call: &TraceSpansRow) -> TraceSpansRow {
    let start_ns =
        i64::try_from(i128::from(call.start_ns) + i128::from(call.duration_ns) + 1_000_000)
            .expect("valid unrelated transport timestamp");
    TraceSpansRow {
        trace_id: call.trace_id.clone(),
        original_trace_id: call.original_trace_id.clone(),
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
        source_type: String::new(),
        source_url: String::new(),
        source_title: String::new(),
        source_user: String::new(),
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
fn captured_trace_cost_matches_spend_logs(
    #[files("../traces-clickhouse/tests/fixtures/*_spend_logs.jsonl")] spend_logs: PathBuf,
) {
    let name = capture_name(&spend_logs);
    let (_, capture, rows, spends) = fixture(&spend_logs);
    let trace = resolve_trace(&capture.trace_id, "", &rows, &spends).expect("captured trace");
    let expected = if capture.spend_linked && capture.spend_complete {
        Some(spends.iter().map(|row| row.spend.unwrap_or(0.0)).sum())
    } else {
        None
    };
    assert_spend_close(trace.summary.spend, expected, name);
}

#[rstest]
fn unrelated_sibling_transport_leaves_cost_unchanged(
    #[files("../traces-clickhouse/tests/fixtures/*_spend_logs.jsonl")] spend_logs: PathBuf,
) {
    let name = capture_name(&spend_logs);
    let (_, capture, rows, spends) = fixture(&spend_logs);
    let baseline = resolve_trace(&capture.trace_id, "", &rows, &spends).expect("captured trace");
    let baseline_spend = baseline.summary.spend;
    let baseline_agent_spends = agent_spends(&baseline);
    let calls: Vec<_> = rows
        .iter()
        .filter(|row| {
            row.kind == ObservationType::Llm
                && !row
                    .call_keys
                    .iter()
                    .any(|key| matches!(key, CallKey::Transport | CallKey::GatewayAttempt))
        })
        .cloned()
        .collect();
    assert!(!calls.is_empty(), "{name} has no model call rows");
    for call in calls {
        let augmented_rows = rows
            .iter()
            .cloned()
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
fn redundant_genai_response_id_keeps_call_evidence(
    #[files("../traces-clickhouse/tests/fixtures/*_spend_logs.jsonl")] spend_logs: PathBuf,
) {
    let (data, capture, _, _) = fixture(&spend_logs);
    let name = capture.name;
    let decoded = decode_otlp(&data.otlp, Some("application/json")).expect("valid OTLP fixture");
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
                    CallKey::ProviderResponse(id) | CallKey::ProviderRequest(id) => {
                        Some(id.clone())
                    }
                    CallKey::LiteLlmRequest(_) | CallKey::Transport | CallKey::GatewayAttempt => {
                        None
                    }
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
                serde_json::from_slice(&data.otlp).expect("valid OTLP JSON fixture");
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
