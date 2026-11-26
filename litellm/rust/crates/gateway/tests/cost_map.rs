//! The native app boots on the bundled cost map and reads the remote one once a refresh installs
//! it.

use std::sync::Arc;

use axum::{body::Body, http::Request};
use litellm_config::{Config, Environment, Flag};
use litellm_gateway::{Admit, AppState, CostMapRefresh, Resources, StoredConfig};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

const REMOTE_COST_MAP: &str = r#"{
    "claude-refresh-test-9": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": false},
    "claude-refresh-test-latest": {"alias_of": "claude-refresh-test-9"}
}"#;

struct Native {
    config: Config,
    state: AppState,
    admit: Arc<Admit>,
    provider: MockServer,
}

impl Native {
    /// Serves two deployments and fetches its remote cost map from `registry`, named by
    /// `LITELLM_MODEL_COST_MAP_URL` as Python's import reads it.
    async fn start(registry: &MockServer) -> Self {
        litellm_gateway::install_crypto_provider();
        let provider = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": "claude-refresh-test-9", "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .mount(&provider)
            .await;
        let config = Config::from_yaml(&format!(
            "model_list:
  - model_name: sampled
    litellm_params:
      model: anthropic/claude-refresh-test-9
      api_base: {base}
      api_key: provider-key
      drop_params: true
  - model_name: aliased
    litellm_params:
      model: anthropic/claude-refresh-test-latest
      api_base: {base}
      api_key: provider-key
general_settings:
  master_key: gateway-key
",
            base = provider.uri()
        ))
        .unwrap();
        let url = format!("{}/model_prices.json", registry.uri());
        let config = config.with_environment(&move |name: &str| {
            (name == "LITELLM_MODEL_COST_MAP_URL").then(|| url.clone())
        });
        let resources = Resources::over(&config, Arc::new(|_: &str| None::<String>)).unwrap();
        let state = AppState::over(&config, &Environment::default(), &resources)
            .await
            .unwrap();
        let admit = Arc::new(Admit::new(&config, &state, StoredConfig::Unused));
        Self {
            config,
            state,
            admit,
            provider,
        }
    }

    fn request(model: &str) -> Request<Body> {
        Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model,
                    "max_tokens": 8,
                    "temperature": 0.5,
                    "messages": [{"role": "user", "content": "hi"}],
                })
                .to_string(),
            ))
            .unwrap()
    }

    /// Who answers a request for `model`, and the reason it goes to Python.
    async fn admitted(&self, model: &str) -> Result<(), String> {
        self.admit
            .clone()
            .admit(Self::request(model))
            .await
            .map(drop)
            .map_err(|(_, reason)| String::from(reason.over))
    }

    /// The temperature the provider received for the last call to `model`.
    async fn sent_temperature(&self, model: &str) -> Option<Value> {
        let response = litellm_gateway::router(self.state.clone(), &self.config)
            .oneshot(Self::request(model))
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let received = self.provider.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&received.last().unwrap().body).unwrap();
        body.get("temperature").cloned()
    }
}

#[rstest]
#[tokio::test]
async fn the_native_app_reads_the_remote_cost_map_once_a_refresh_installs_it() {
    let registry = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model_prices.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(REMOTE_COST_MAP))
        .mount(&registry)
        .await;
    let native = Native::start(&registry).await;
    let environment = Environment {
        model_cost_map_min_model_count: Some(0),
        model_cost_map_max_shrink_ratio: Some(0.0),
        ..Environment::default()
    };
    let bundled = (
        native.admitted("aliased").await,
        native.sent_temperature("sampled").await,
    );

    CostMapRefresh::new(&native.config, &environment, &native.state)
        .unwrap()
        .unwrap()
        .run(native.state.clone())
        .await;

    assert_eq!(bundled, (Ok(()), Some(json!(0.5))));
    assert_eq!(
        (
            native.admitted("aliased").await,
            native.sent_temperature("sampled").await,
        ),
        (
            Err("model_list.litellm_params.model alias claude-refresh-test-latest".to_owned()),
            None
        )
    );
    native.state.close().await;
}

#[rstest]
#[tokio::test]
async fn a_refused_remote_cost_map_leaves_the_bundled_one_in_use() {
    let registry = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model_prices.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(REMOTE_COST_MAP))
        .mount(&registry)
        .await;
    let native = Native::start(&registry).await;
    let environment = Environment::default();

    CostMapRefresh::new(&native.config, &environment, &native.state)
        .unwrap()
        .unwrap()
        .run(native.state.clone())
        .await;

    assert_eq!(
        (
            native.admitted("aliased").await,
            native.sent_temperature("sampled").await,
        ),
        (Ok(()), Some(json!(0.5)))
    );
    native.state.close().await;
}

#[rstest]
#[case::local_map(Environment { litellm_local_model_cost_map: Flag(true), ..Environment::default() })]
#[case::offline(Environment { litellm_offline_mode: Flag(true), ..Environment::default() })]
#[tokio::test]
async fn a_bundled_only_setting_starts_no_refresh(#[case] environment: Environment) {
    let native = Native::start(&MockServer::start().await).await;

    assert!(matches!(
        CostMapRefresh::new(&native.config, &environment, &native.state),
        Ok(None)
    ));
    native.state.close().await;
}
