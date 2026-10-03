use litellm_traces::query::named::*;
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T: DeserializeOwned + Serialize>(wire: Value) {
    let contract: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(contract).unwrap(), wire);
}

#[rstest]
#[case::admin(1, "", vec![])]
#[case::own_user(0, "user", vec![])]
#[case::multiple_teams(0, "user", vec!["team-a", "team-b"])]
#[case::no_identity(0, "", vec![])]
fn named_requests_preserve_all_access_cases(
    #[case] all_teams: u8,
    #[case] user: &str,
    #[case] teams: Vec<&str>,
) {
    let access = json!({"all_teams": all_teams, "user_id": user, "team_ids": teams});
    round_trip::<ReadAccessParams>(access.clone());
    let request = |specific: Value| {
        Value::Object(
            access
                .as_object()
                .unwrap()
                .iter()
                .chain(specific.as_object().unwrap())
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        )
    };
    round_trip::<ListTracesParams>(request(
        json!({"start_ms": -1, "end_ms": 10, "cursor_ms": 0, "cursor_trace_id": "", "limit": 100}),
    ));
    round_trip::<TraceIdentityParams>(request(json!({"trace_id": "trace"})));
    round_trip::<TraceSpansParams>(request(json!({"trace_id": "trace", "trace_ref": "ref"})));
    round_trip::<SpanDetailParams>(request(
        json!({"trace_id": "trace", "trace_ref": "ref", "span_id": "span"}),
    ));
    round_trip::<SpanErrorParams>(request(
        json!({"trace_id": "trace", "trace_ref": "ref", "span_id": "span", "error_offset": u64::MAX, "error_version": "version"}),
    ));
    round_trip::<SpendByResponseIdsParams>(request(
        json!({"response_ids": ["response"], "request_ids": ["request"], "trace_ids": ["trace"], "start_ms": -1, "end_ms": 10}),
    ));
}

#[rstest]
fn result_contracts_preserve_public_field_names() {
    round_trip::<ListTracesRow>(
        json!({"trace_id": "trace", "trace_ref": "ref", "team_id": "team", "api_key_hash": "key", "user_id": "user", "name": "agent", "service": "service", "input_preview": "input", "status": "STATUS_CODE_OK", "start_ms": -1, "duration_ms": 20, "span_count": u64::MAX, "agent_count": 1, "agent_invocations": 2, "agent_names": ["agent"], "frameworks": ["framework"], "llm_calls": 3, "tool_calls": 4, "input_tokens": 5, "output_tokens": 6, "models": ["model"], "error_count": 0, "request_ids": ["request"]}),
    );
    round_trip::<TraceSpansRow>(
        json!({"trace_id": "trace", "span_id": "span", "parent_span_id": "parent", "name": "agent", "type": "agent", "wrapper_candidate": 1, "agent": "agent", "framework": "framework", "status": "STATUS_CODE_ERROR", "status_message": "error", "error_truncated": 1, "start_ns": -1, "duration_ns": u64::MAX, "service": "service", "input_preview": "input", "model": "model", "input_tokens": u32::MAX, "output_tokens": 6, "litellm_request_id": "request", "call_keys": ["provider_response:request"], "call_evidence": "complete", "tool_call_id": "call", "team_id": "team", "api_key_hash": "key", "user_id": "user"}),
    );
    round_trip::<SpanDetailRow>(
        json!({"span_id": "span", "input": "input", "output": "output", "attributes": {"count": "42"}}),
    );
    round_trip::<SpanErrorRow>(
        json!({"span_id": "span", "message": "error", "total_chars": u64::MAX, "version": "version"}),
    );
    round_trip::<SpendByResponseIdsRow>(
        json!({"request_id": "request", "response_id": "response", "upstream_response_id": "upstream", "trace_id": "trace", "span_id": "span", "team_id": "team", "api_key": "key", "user": "user", "spend": 0.125, "start_ms": -1}),
    );
}
