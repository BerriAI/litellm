//! The gateway's model table: booted on the bundled cost map, republished from the remote one.

use std::sync::Arc;

use litellm_config::Config;
use litellm_gateway::Refreshed;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn config(url: &str, extra: &str) -> Config {
    Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-native-9\nenvironment_variables:\n  LITELLM_MODEL_COST_MAP_URL: {url}\n  MODEL_COST_MAP_MIN_MODEL_COUNT: \"0\"\n  MODEL_COST_MAP_MAX_SHRINK_RATIO: \"0\"\n{extra}"
    ))
    .unwrap()
}

fn reasons(gateway: &litellm_gateway::Gateway) -> bool {
    gateway
        .models()
        .shaping
        .values()
        .next()
        .unwrap()
        .capabilities
        .supports_reasoning
}

async fn remote() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/cost.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "claude-native-9": {"litellm_provider": "anthropic", "mode": "chat", "supports_reasoning": false},
        })))
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn a_remote_cost_map_republishes_the_table_once_until_it_changes() {
    let server = remote().await;
    let (gateway, refresh) =
        litellm_gateway::build_serving(&config(&format!("{}/cost.json", server.uri()), ""))
            .await
            .unwrap();
    let booted = gateway.models();

    let first = refresh.refresh(&gateway).await;
    let published = gateway.models();
    let second = refresh.refresh(&gateway).await;

    assert_eq!(
        (
            first,
            second,
            reasons(&gateway),
            Arc::ptr_eq(&published, &gateway.models()),
            booted
                .shaping
                .values()
                .next()
                .unwrap()
                .capabilities
                .supports_reasoning,
        ),
        (
            Refreshed::Published,
            Refreshed::Unchanged,
            false,
            true,
            true
        )
    );
}

#[rstest::rstest]
#[case::local_map("  LITELLM_LOCAL_MODEL_COST_MAP: \"True\"\n", "/cost.json")]
#[case::offline("  LITELLM_OFFLINE_MODE: \"true\"\n", "/cost.json")]
#[case::unreachable("", "/missing.json")]
#[tokio::test]
async fn a_map_that_is_not_fetched_keeps_the_bundled_table(
    #[case] extra: &str,
    #[case] remote_path: &str,
) {
    let server = remote().await;
    let config = config(&format!("{}{remote_path}", server.uri()), extra);
    let (gateway, refresh) = litellm_gateway::build_serving(&config).await.unwrap();

    assert_eq!(
        (refresh.refresh(&gateway).await, reasons(&gateway)),
        (Refreshed::Kept, true)
    );
}
