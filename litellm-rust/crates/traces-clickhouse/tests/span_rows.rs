use litellm_traces::{Shared, Tenant, decode_otlp};
use litellm_traces_clickhouse::{NORMALIZED_FIELD_DEFINITIONS, span_rows};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

const MAX_VALUE_BYTES: usize = 64 * 1024;

#[fixture]
fn tenant() -> Tenant {
    Tenant {
        team_id: "team-a".into(),
        api_key_hash: "key-a".into(),
        org_id: "org-a".into(),
        user_id: "user-a".into(),
    }
}

fn attribute(key: &str, value: &str) -> Value {
    json!({"key": key, "value": {"stringValue": value}})
}

fn span(span_id: &str, attributes: Vec<Value>, extra: Value) -> Value {
    let mut span = json!({
        "traceId": "01".repeat(16),
        "spanId": span_id,
        "name": "operation",
        "startTimeUnixNano": "1000",
        "endTimeUnixNano": "5000",
        "attributes": attributes,
    });
    span.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    span
}

fn export(resources: Vec<(Vec<Value>, Vec<Value>)>) -> Vec<u8> {
    let resource_spans: Vec<Value> = resources
        .into_iter()
        .map(|(attributes, spans)| {
            json!({
                "resource": {"attributes": attributes},
                "scopeSpans": [{"scope": {"name": "scope", "version": "1"}, "spans": spans}],
            })
        })
        .collect();
    json!({"resourceSpans": resource_spans})
        .to_string()
        .into_bytes()
}

fn rows(body: &[u8], tenant: &Tenant, max_value_bytes: usize) -> Vec<Value> {
    let spans = decode_otlp(body, Some("application/json")).unwrap();
    span_rows(spans, tenant, max_value_bytes)
        .iter()
        .map(|row| serde_json::to_value(row).unwrap())
        .collect()
}

#[rstest]
fn tenant_overwrites_claimed_identity_and_resources_stay_shared_per_group(tenant: Tenant) {
    let spoofed = vec![
        attribute("service.name", "svc"),
        attribute("litellm.team_id", "spoofed-team"),
        attribute("litellm.user_id", "spoofed-user"),
    ];
    let body = export(vec![
        (
            spoofed.clone(),
            vec![
                span(&"02".repeat(8), vec![], json!({})),
                span(&"03".repeat(8), vec![], json!({})),
            ],
        ),
        (spoofed, vec![span(&"04".repeat(8), vec![], json!({}))]),
    ]);
    let spans = decode_otlp(&body, Some("application/json")).unwrap();
    let stored = span_rows(spans, &tenant, MAX_VALUE_BYTES);
    let resource = |index: usize| &stored[index]["ResourceAttributes"];

    assert!(Shared::shares_storage_with(resource(0), resource(1)));
    assert!(!Shared::shares_storage_with(resource(0), resource(2)));
    assert_eq!(resource(0), resource(2));
    assert_eq!(
        **resource(0),
        json!({
            "service.name": "svc",
            "litellm.team_id": "team-a",
            "litellm.user_id": "user-a",
            "litellm.api_key_hash": "key-a",
            "litellm.org_id": "org-a",
        })
    );
    for row in &stored {
        assert_eq!(
            (&*row["TeamId"], &*row["ApiKeyHash"], &*row["UserId"]),
            (&json!("team-a"), &json!("key-a"), &json!("user-a"))
        );
        assert_eq!(*row["ServiceName"], json!("svc"));
    }
}

#[rstest]
#[case::exception_event("", json!("customer acme-404 not found"))]
#[case::status_message_wins("boom", json!("boom"))]
fn status_message_falls_back_to_the_exception_event(
    tenant: Tenant,
    #[case] status_message: &str,
    #[case] expected: Value,
) {
    let exported = span(
        &"02".repeat(8),
        vec![],
        json!({
            "status": {"code": 2, "message": status_message},
            "events": [{"name": "exception", "timeUnixNano": "2000", "attributes": [
                attribute("exception.type", "KeyError"),
                attribute("exception.message", "customer acme-404 not found"),
            ]}],
        }),
    );
    let row = &rows(
        &export(vec![(vec![], vec![exported])]),
        &tenant,
        MAX_VALUE_BYTES,
    )[0];
    assert_eq!(row["StatusCode"], "STATUS_CODE_ERROR");
    assert_eq!(row["StatusMessage"], expected);
}

#[rstest]
fn consumed_payloads_leave_span_attributes_and_long_values_are_capped(tenant: Tenant) {
    let messages = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "x".repeat(300)},
        {"role": "user", "content": "latest question"},
    ]);
    let exported = span(
        &"02".repeat(8),
        vec![
            attribute("gen_ai.operation.name", "chat"),
            attribute("gen_ai.input.messages", &messages.to_string()),
            attribute(
                "gen_ai.output.messages",
                &json!([{"role": "assistant", "content": "y".repeat(300)}]).to_string(),
            ),
            attribute("custom.blob", &"z".repeat(300)),
        ],
        json!({}),
    );
    let row = &rows(&export(vec![(vec![], vec![exported])]), &tenant, 200)[0];
    let attributes = row["SpanAttributes"].as_object().unwrap();
    assert!(!attributes.contains_key("gen_ai.input.messages"));
    assert!(!attributes.contains_key("gen_ai.output.messages"));
    assert_eq!(
        attributes["custom.blob"],
        format!("{}…[truncated 100 bytes]", "z".repeat(200))
    );
    let input = row["Input"].as_str().unwrap();
    let kept: Vec<Value> = serde_json::from_str(input).unwrap();
    assert!(input.len() <= 200);
    assert_eq!(kept[0]["content"], "be brief");
    assert_eq!(kept.last().unwrap()["content"], "latest question");
    assert!(row["Output"].as_str().unwrap().contains("…[truncated "));
    assert_eq!(row["ObservationType"], "llm");
}

#[rstest]
fn rows_carry_every_normalized_column(tenant: Tenant) {
    let row = &rows(
        &export(vec![(
            vec![],
            vec![span(&"02".repeat(8), vec![], json!({}))],
        )]),
        &tenant,
        MAX_VALUE_BYTES,
    )[0];
    for field in NORMALIZED_FIELD_DEFINITIONS {
        assert!(
            row.get(field.clickhouse_column).is_some(),
            "{}",
            field.clickhouse_column
        );
    }
    assert_eq!(row["Duration"], 4000);
    assert_eq!(row["AgentMetadata"], "{}");
}

#[rstest]
fn absent_identity_fields_are_empty_only_in_storage(tenant: Tenant) {
    let body = export(vec![(
        vec![],
        vec![span(&"02".repeat(8), vec![], json!({}))],
    )]);
    let decoded = decode_otlp(&body, Some("application/json")).unwrap();
    let normalized = &decoded[0].normalized;
    assert_eq!(normalized.agent_name, None);
    assert_eq!(normalized.framework, None);
    assert_eq!(normalized.model, None);
    assert_eq!(normalized.tool_call_id, None);
    let stored = span_rows(decoded, &tenant, MAX_VALUE_BYTES);
    let row = serde_json::to_value(&stored[0]).unwrap();
    assert_eq!(
        [
            "AgentName",
            "Framework",
            "Model",
            "ToolCallId",
            "LiteLLMRequestId"
        ]
        .map(|column| row[column].clone()),
        [""; 5].map(|value| json!(value)),
    );
    assert_eq!(row["CallKeys"], json!([]));
    assert_eq!(row["CallEvidence"], "unknown");
}

#[rstest]
fn compatibility_id_keeps_provider_semantics_with_gateway_keys(tenant: Tenant) {
    let body = export(vec![(
        vec![],
        vec![span(
            &"02".repeat(8),
            vec![
                attribute("gen_ai.response.id", "response"),
                attribute("litellm.call_id", "gateway"),
            ],
            json!({}),
        )],
    )]);
    let stored = rows(&body, &tenant, MAX_VALUE_BYTES);
    assert_eq!(stored[0]["LiteLLMRequestId"], "response");
    assert_eq!(stored[0]["CallEvidence"], "partial");
    assert_eq!(
        stored[0]["CallKeys"],
        json!(["litellm_request:gateway", "provider_response:response"])
    );
}
