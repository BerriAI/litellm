//! A key's own budget, served natively while `general_settings.disable_budget_reservation` is on:
//! `admit` admits the key, and the read-time check refuses a call over its budget with Python's
//! budget error before anything reaches the provider. A served call charges the key the cost its
//! route settled, so the key's own calls spend it past its budget, and a call the route cannot price
//! goes to Python, which prices it. A key without a budget is served such a call natively.

mod keys;

use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use litellm_config::{Config, Environment};
use litellm_gateway::{Admit, AppState, StoredConfig};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const KEY: &str = "sk-budgeted";
const PRICED: &str = "anthropic/claude-haiku-5-5";
const UNKNOWN: &str = "anthropic/claude-unlisted-9";

/// Who answered a request: Python, over the reason `admit` gave, or the native app.
#[derive(Debug, PartialEq)]
enum Answered {
    Python(String),
    Native(u16, String),
}

async fn provider(usage: Value) -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": usage
        })))
        .mount(&upstream)
        .await;
    upstream
}

/// A gateway in front of Python, with [`KEY`] holding `columns` in its database, serving the
/// model group named for `model` from a provider that reports `usage` on every answer.
struct Gateway {
    _held: keys::Held,
    upstream: MockServer,
    config: Config,
    state: AppState,
    admit: Arc<Admit>,
    group: String,
}

impl Gateway {
    async fn start(columns: Value, model: &str, usage: Value, reservation_disabled: bool) -> Self {
        litellm_gateway::install_crypto_provider();
        let held = keys::Held::holding(KEY, columns).await;
        let upstream = provider(usage).await;
        let group = model.rsplit_once('/').map_or(model, |(_, name)| name);
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: {group}\n    litellm_params:\n      model: {model}\n      api_base: {}\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n  database_url: {}\n  disable_budget_reservation: {reservation_disabled}\n",
            upstream.uri(),
            held.url(),
        ))
        .unwrap();
        let state = AppState::build(&config, &Environment::default())
            .await
            .unwrap();
        let admit = Arc::new(Admit::new(&config, &state, StoredConfig::Unused));
        Self {
            _held: held,
            upstream,
            config,
            state,
            admit,
            group: group.to_owned(),
        }
    }

    /// What one call from [`KEY`] gets.
    async fn send(&self) -> Answered {
        self.send_with(json!({})).await
    }

    /// What one call from [`KEY`] gets when its body also carries each field of `extra`.
    async fn send_with(&self, extra: Value) -> Answered {
        let body = json!({"model": self.group, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
            .as_object()
            .into_iter()
            .flatten()
            .chain(extra.as_object().into_iter().flatten())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<serde_json::Map<_, _>>();
        let request = Request::post("/v1/messages")
            .header("authorization", format!("Bearer {KEY}"))
            .header("content-type", "application/json")
            .body(Body::from(Value::Object(body).to_string()))
            .unwrap();
        match self.admit.clone().admit(request).await {
            Err((_, reason)) => Answered::Python(<&str>::from(reason.why).to_owned()),
            Ok(admitted) => {
                let response = litellm_gateway::router(self.state.clone(), &self.config)
                    .oneshot(admitted)
                    .await
                    .unwrap();
                let status = response.status().as_u16();
                let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
                Answered::Native(status, String::from_utf8(body.to_vec()).unwrap())
            }
        }
    }

    /// How many calls reached the provider, once the gateway is closed.
    async fn close(self) -> usize {
        self.state.close().await;
        self.upstream
            .received_requests()
            .await
            .unwrap_or_default()
            .len()
    }
}

/// What a call from [`KEY`], holding `columns`, to the model group named for `model` gets from a
/// gateway in front of Python, and how many calls reached the provider.
async fn answered(columns: Value, model: &str, reservation_disabled: bool) -> (Answered, usize) {
    answered_with(columns, model, reservation_disabled, json!({})).await
}

/// [`answered`] for a call whose body also carries each field of `extra`.
async fn answered_with(
    columns: Value,
    model: &str,
    reservation_disabled: bool,
    extra: Value,
) -> (Answered, usize) {
    let gateway = Gateway::start(
        columns,
        model,
        json!({"input_tokens": 1, "output_tokens": 1}),
        reservation_disabled,
    )
    .await;
    let answered = gateway.send_with(extra).await;
    (answered, gateway.close().await)
}

/// The status and `litellm_error_type` of a native answer, or `None` for an answer from Python.
fn refusal(answered: &Answered) -> Option<(u16, Option<String>)> {
    let Answered::Native(status, body) = answered else {
        return None;
    };
    let error_type = serde_json::from_str::<Value>(body).ok().and_then(|body| {
        body["error"]["litellm_error_type"]
            .as_str()
            .map(str::to_owned)
    });
    Some((*status, error_type))
}

/// The bytes of Python's `BudgetExceededError` for a key over its budget, in the error envelope
/// Python's exception handler answers `/v1/messages` with.
fn over_budget(spend: &str) -> String {
    format!(
        r#"{{"type":"error","error":{{"type":"invalid_request_error","message":"Budget has been exceeded! Key=key (sk-...-key) Current cost: {spend}, Max budget: 10.0","litellm_error_type":"budget_exceeded"}}}}"#
    )
}

#[rstest]
#[case::at_its_budget(json!({"max_budget": 10.0, "spend": 10.0}), PRICED, "10.0")]
#[case::over_its_budget(json!({"max_budget": 10.0, "spend": 12.5}), PRICED, "12.5")]
#[case::over_a_budget_that_resets(
    json!({"max_budget": 10.0, "spend": 12.5, "budget_duration": "30d", "budget_reset_at": "2100-01-01T00:00:00"}),
    PRICED,
    "12.5"
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_at_or_over_its_budget_is_refused_natively_with_pythons_error(
    #[case] columns: Value,
    #[case] model: &str,
    #[case] spend: &str,
) {
    assert_eq!(
        answered(columns, model, true).await,
        (Answered::Native(422, over_budget(spend)), 0)
    );
}

fn web_search() -> Value {
    json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})
}

#[rstest]
#[case::with_a_max_budget(json!({"max_budget": 10.0, "spend": 9.5}), PRICED, json!({}))]
#[case::with_a_budget_that_resets(
    json!({"max_budget": 10.0, "spend": 9.5, "budget_duration": "30d", "budget_reset_at": "2100-01-01T00:00:00"}),
    PRICED,
    json!({})
)]
#[case::with_a_tool_it_runs_itself(
    json!({"max_budget": 10.0, "spend": 9.5}),
    PRICED,
    json!({"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]})
)]
#[case::without_a_budget_on_a_model_the_cost_map_does_not_name(json!({}), UNKNOWN, json!({}))]
#[case::without_a_budget_asking_for_web_search(json!({}), PRICED, web_search())]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_under_its_budget_is_served_natively(
    #[case] columns: Value,
    #[case] model: &str,
    #[case] extra: Value,
) {
    let (answered, sent) = answered_with(columns, model, true, extra).await;

    assert_eq!(
        (matches!(answered, Answered::Native(200, _)), sent),
        (true, 1),
        "{answered:?}"
    );
}

/// A budgeted key's call the native route would leave unpriced would charge it nothing, so it goes
/// to Python before anything reaches the provider, whether or not the key is over its budget yet.
#[rstest]
#[case::asking_for_web_search(json!({"max_budget": 10.0, "spend": 9.5}), PRICED, web_search(), "body")]
#[case::streaming_with_web_search(
    json!({"max_budget": 10.0, "spend": 9.5}),
    PRICED,
    json!({"stream": true, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
    "body"
)]
#[case::asking_for_thinking(
    json!({"max_budget": 10.0, "spend": 9.5}),
    PRICED,
    json!({"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
    "body"
)]
#[case::on_a_model_the_cost_map_does_not_name(json!({"max_budget": 10.0, "spend": 9.5}), UNKNOWN, json!({}), "model")]
#[case::over_its_budget_on_a_model_the_cost_map_does_not_name(
    json!({"max_budget": 10.0, "spend": 12.5}),
    UNKNOWN,
    json!({}),
    "model"
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_budgeted_keys_call_the_gateway_cannot_price_goes_to_python(
    #[case] columns: Value,
    #[case] model: &str,
    #[case] extra: Value,
    #[case] reason: &str,
) {
    assert_eq!(
        answered_with(columns, model, true, extra).await,
        (Answered::Python(reason.to_owned()), 0)
    );
}

#[rstest]
#[case::under_its_budget(json!({"max_budget": 10.0, "spend": 9.5}))]
#[case::over_its_budget(json!({"max_budget": 10.0, "spend": 12.5}))]
#[tokio::test(flavor = "multi_thread")]
async fn with_the_reservation_on_a_key_with_a_budget_goes_to_python(#[case] columns: Value) {
    assert_eq!(
        answered(columns, PRICED, false).await,
        (Answered::Python("credential".to_owned()), 0)
    );
}

/// The calls a gateway serves before it answers one any other way, and that answer. The spend log
/// charges a served call after its answer is sent, so a call right behind it can still be served.
async fn served_until_refused(gateway: &Gateway) -> (usize, Option<Answered>) {
    for served in 0..100 {
        let answered = gateway.send().await;
        if !matches!(answered, Answered::Native(200, _)) {
            return (served, Some(answered));
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    (100, None)
}

/// A key's own calls spend it past its budget: the spend log charges what each served call cost,
/// and a later call reads that charge and is refused before it reaches the provider.
#[tokio::test(flavor = "multi_thread")]
async fn a_key_under_its_budget_is_refused_once_its_native_calls_spend_past_it() {
    let gateway = Gateway::start(
        json!({"max_budget": 0.000001, "spend": 0.0}),
        PRICED,
        json!({"input_tokens": 100_000, "output_tokens": 100_000}),
        true,
    )
    .await;

    let (served, refused) = served_until_refused(&gateway).await;
    let sent = gateway.close().await;

    assert_eq!(
        (
            served >= 1,
            refused.as_ref().and_then(refusal),
            sent == served
        ),
        (true, Some((422, Some("budget_exceeded".to_owned()))), true),
        "served {served}, sent {sent}"
    );
}
