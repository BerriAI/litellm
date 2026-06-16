use std::sync::Arc;
use std::time::Duration;

use guardrails::HttpClient;
use guardrails::{GuardrailInput, InputType, ProviderConfig, RequestContext, Verdict};

fn build(config_json: serde_json::Value) -> Box<dyn guardrails::Guardrail> {
    let config: ProviderConfig = serde_json::from_value(config_json).unwrap();
    let http = Arc::new(HttpClient::new(Duration::from_secs(5)));
    guardrails::build(config, &http).unwrap()
}

fn input(text: &str) -> GuardrailInput {
    GuardrailInput {
        texts: vec![text.to_owned()],
        ..Default::default()
    }
}

async fn verdict(config_json: serde_json::Value, text: &str) -> Verdict {
    build(config_json)
        .apply(&input(text), InputType::Request, &RequestContext::default())
        .await
        .unwrap()
        .verdict
}

#[tokio::test]
async fn masks_pii_with_zero_network_calls() {
    let result = verdict(
        serde_json::json!({"guardrail": "local_pii"}),
        "ping me at dev@acme.io or 4111 1111 1111 1111",
    )
    .await;

    match result {
        Verdict::Mask {
            texts,
            masked_entity_count,
            ..
        } => {
            assert_eq!(texts[0], "ping me at <EMAIL_ADDRESS> or <CREDIT_CARD>");
            assert_eq!(masked_entity_count.get("EMAIL_ADDRESS"), Some(&1));
            assert_eq!(masked_entity_count.get("CREDIT_CARD"), Some(&1));
        }
        other => panic!("expected mask, got {other:?}"),
    }
}

#[tokio::test]
async fn blocks_when_entity_action_is_block() {
    let result = verdict(
        serde_json::json!({
            "guardrail": "local_pii",
            "pii_entities_config": {"EMAIL_ADDRESS": "BLOCK"},
        }),
        "contact root@corp.internal",
    )
    .await;

    assert!(matches!(result, Verdict::Block { .. }));
}

#[tokio::test]
async fn allow_list_short_circuits_to_pass() {
    let result = verdict(
        serde_json::json!({
            "guardrail": "local_pii",
            "allow_list": ["help@acme.io"],
        }),
        "write to help@acme.io",
    )
    .await;

    assert_eq!(result, Verdict::Pass);
}
