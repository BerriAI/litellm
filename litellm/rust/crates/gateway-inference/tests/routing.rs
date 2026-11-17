mod support;

use axum::Router;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

struct Upstreams {
    primary: MockServer,
    backup: MockServer,
}

#[fixture]
async fn upstreams() -> Upstreams {
    Upstreams {
        primary: MockServer::start().await,
        backup: MockServer::start().await,
    }
}

async fn answer(server: &MockServer, response: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(response)
        .mount(server)
        .await;
}

fn message() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }))
}

fn provider_error(status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(json!({
        "type": "error", "error": {"type": "api_error", "message": "upstream failed"},
    }))
}

fn deployment(name: &str, id: &str, api_base: &str) -> Value {
    json!({
        "model_name": name,
        "litellm_params": {"model": "anthropic/test-model", "api_base": api_base, "api_key": "test-key"},
        "model_info": {"id": id},
    })
}

fn app(upstreams: &Upstreams, settings: Value) -> Router {
    support::routed_app(
        settings,
        json!([
            deployment("primary", "primary-1", &upstreams.primary.uri()),
            deployment("backup", "backup-1", &upstreams.backup.uri()),
        ]),
    )
}

async fn send(app: Router) -> axum::response::Response {
    support::post(
        app,
        "/v1/messages",
        json!({"model": "primary", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await
}

async fn calls(server: &MockServer) -> usize {
    server.received_requests().await.unwrap().len()
}

#[rstest]
#[tokio::test]
async fn a_failing_group_is_retried_then_falls_back(#[future(awt)] upstreams: Upstreams) {
    answer(&upstreams.primary, provider_error(500)).await;
    answer(&upstreams.backup, message()).await;

    let response = send(app(
        &upstreams,
        json!({"num_retries": 1, "fallbacks": [{"primary": ["backup"]}]}),
    ))
    .await;

    assert_eq!(response.status(), 200);
    let headers = response.headers();
    assert_eq!(
        (
            &headers["x-litellm-model-id"],
            &headers["x-litellm-model-group"],
            &headers["x-litellm-attempted-fallbacks"],
        ),
        (
            &"backup-1".parse().unwrap(),
            &"backup".parse().unwrap(),
            &"1".parse().unwrap()
        )
    );
    assert_eq!(
        (
            calls(&upstreams.primary).await,
            calls(&upstreams.backup).await
        ),
        (2, 1)
    );
}

#[rstest]
#[case::falls_back_by_default(Value::Null, 200, (1, 1))]
#[case::stays_when_unsupported_params_do_not_fall_back(json!(false), 400, (1, 0))]
#[tokio::test]
async fn a_provider_bad_request_is_never_retried(
    #[future(awt)] upstreams: Upstreams,
    #[case] fallback_on_unsupported_params: Value,
    #[case] status: u16,
    #[case] expected_calls: (usize, usize),
) {
    answer(&upstreams.primary, provider_error(400)).await;
    answer(&upstreams.backup, message()).await;
    let app = app(
        &upstreams,
        json!({
            "num_retries": 1,
            "fallbacks": [{"primary": ["backup"]}],
            "fallback_on_unsupported_params": fallback_on_unsupported_params,
        }),
    );

    let response = send(app).await;

    assert_eq!(response.status(), status);
    assert_eq!(
        (
            calls(&upstreams.primary).await,
            calls(&upstreams.backup).await
        ),
        expected_calls
    );
}

fn weighted(id: &str, api_base: &str, weight: u32) -> Value {
    let mut deployment = deployment("primary", id, api_base);
    deployment["litellm_params"]["weight"] = json!(weight);
    deployment
}

#[rstest]
#[tokio::test]
async fn a_name_picks_among_all_its_deployments(#[future(awt)] upstreams: Upstreams) {
    answer(&upstreams.primary, message()).await;
    answer(&upstreams.backup, message()).await;
    let app = support::routed_app(
        json!({}),
        json!([
            weighted("weighted", &upstreams.primary.uri(), 1),
            weighted("unweighted", &upstreams.backup.uri(), 0),
        ]),
    );

    let response = send(app).await;

    assert_eq!(response.headers()["x-litellm-model-id"], "weighted");
    assert_eq!(
        (
            calls(&upstreams.primary).await,
            calls(&upstreams.backup).await
        ),
        (1, 0)
    );
}
