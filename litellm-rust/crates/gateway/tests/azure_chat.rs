//! Azure OpenAI chat completions through the gateway: the deployment URL, `api-version` and
//! `api-key` the upstream sees, and the usage and cost the request is recorded with.

use std::{
    io::Write,
    sync::{Arc, Mutex, Once},
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_gateway_hooks::{Facts, Observer, Record};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

const RESPONSE_MODEL: &str = "gpt-4o-2024-11-20";
const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o-2024-11-20","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#;
const COMPLETION_STREAM: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-11-20\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-11-20\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n",
    "data: [DONE]\n\n",
);

/// Prices the cost map gives the model a response names, and the `base_model` one deployment
/// declares, so the two pricing routes give different costs.
const RESPONSE_MODEL_RATES: (f64, f64) = (1.0, 2.0);
const BASE_MODEL_RATES: (f64, f64) = (10.0, 20.0);

/// The proxy reads its cost map from this variable, and every test sets it before building a
/// gateway, so no test reads the environment while it changes.
fn use_local_cost_map() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let map = json!({
            format!("azure/{RESPONSE_MODEL}"): {
                "litellm_provider": "azure",
                "mode": "chat",
                "input_cost_per_token": RESPONSE_MODEL_RATES.0,
                "output_cost_per_token": RESPONSE_MODEL_RATES.1,
            },
            "azure/gpt-4o": {
                "litellm_provider": "azure",
                "mode": "chat",
                "input_cost_per_token": BASE_MODEL_RATES.0,
                "output_cost_per_token": BASE_MODEL_RATES.1,
            },
        });
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(map.to_string().as_bytes()).unwrap();
        let (_, path) = file.keep().unwrap();
        // SAFETY: set once, before any test builds a gateway or spawns a thread that reads the
        // environment.
        unsafe { std::env::set_var("LITELLM_MODEL_COST_MAP_PATH", path) };
    });
}

#[derive(Clone, Debug)]
struct Sent {
    path: String,
    query: Option<String>,
    api_key: Option<String>,
    authorization: Option<String>,
    body: Value,
}

type Calls = Arc<Mutex<Vec<Sent>>>;

async fn upstream(calls: Calls) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let calls = calls.clone();
        async move {
            let (parts, body) = request.into_parts();
            let header = |name: &str| {
                parts
                    .headers
                    .get(name)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned)
            };
            let (api_key, authorization) = (header("api-key"), header("authorization"));
            let path = parts.uri.path().to_owned();
            let query = parts.uri.query().map(str::to_owned);
            let body: Value =
                serde_json::from_slice(&to_bytes(body, 1 << 20).await.unwrap()).unwrap();
            let stream = body["stream"].as_bool().unwrap_or(false);
            calls.lock().unwrap().push(Sent {
                path,
                query,
                api_key,
                authorization,
                body,
            });
            match stream {
                true => ([(CONTENT_TYPE, "text/event-stream")], COMPLETION_STREAM).into_response(),
                false => ([(CONTENT_TYPE, "application/json")], COMPLETION).into_response(),
            }
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

#[derive(Debug)]
struct Seen {
    deployment: Option<String>,
    usage: Option<(u64, u64)>,
    cost: Option<f64>,
}

struct Collect(Mutex<Vec<Seen>>);

impl Observer for Collect {
    fn name(&self) -> &'static str {
        "collect"
    }

    fn reads(&self) -> Facts {
        Facts::DEPLOYMENT | Facts::USAGE | Facts::COST
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.0.lock().unwrap().push(Seen {
                deployment: record
                    .outcome
                    .deployment
                    .as_ref()
                    .map(|deployment| deployment.id.to_string()),
                usage: record
                    .outcome
                    .usage
                    .map(|usage| (usage.prompt_tokens(), usage.output_tokens)),
                cost: record.outcome.cost,
            });
        })
    }
}

struct Harness {
    app: Router,
    inference: Arc<litellm_gateway_inference::Gateway>,
    calls: Calls,
    seen: Arc<Collect>,
}

async fn harness() -> Harness {
    use_local_cost_map();
    let calls = Calls::default();
    let upstream = upstream(calls.clone()).await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: chat
    litellm_params:
      model: azure/chat-prod
      api_base: {upstream}
      api_key: sk-azure
      api_version: 2024-10-21
    model_info:
      id: azure-1
  - model_name: chat-base
    litellm_params:
      model: azure/chat-base
      api_base: {upstream}/
      api_key: sk-azure
      api_version: 2024-10-21
      base_model: gpt-4o
    model_info:
      id: azure-2
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let seen = Arc::new(Collect(Mutex::new(Vec::new())));
    litellm_gateway::hooks::install(&inference, &config, Vec::new(), vec![seen.clone()])
        .await
        .unwrap();
    let app = litellm_gateway::router(inference.clone(), &config, None, None);
    Harness {
        app,
        inference,
        calls,
        seen,
    }
}

async fn send(app: &Router, body: Value) -> (u16, String) {
    send_to(app, "/v1/chat/completions", body).await
}

async fn send_to(app: &Router, path: &str, body: Value) -> (u16, String) {
    let request = Request::post(path)
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-master")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

fn cost((input, output): (u64, u64), (input_rate, output_rate): (f64, f64)) -> f64 {
    input as f64 * input_rate + output as f64 * output_rate
}

#[rstest]
#[case::unary_by_response_model("chat", false, "chat-prod", "azure-1", (5, 6), RESPONSE_MODEL_RATES)]
#[case::stream_by_response_model("chat", true, "chat-prod", "azure-1", (7, 3), RESPONSE_MODEL_RATES)]
#[case::unary_by_base_model("chat-base", false, "chat-base", "azure-2", (5, 6), BASE_MODEL_RATES)]
#[case::stream_by_base_model("chat-base", true, "chat-base", "azure-2", (7, 3), BASE_MODEL_RATES)]
#[tokio::test]
async fn azure_chat_reaches_the_deployment_and_is_priced(
    #[case] model: &str,
    #[case] stream: bool,
    #[case] deployment: &str,
    #[case] id: &str,
    #[case] usage: (u64, u64),
    #[case] rates: (f64, f64),
) {
    let harness = harness().await;
    let mut request = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "user": "end-user-1",
    });
    if stream {
        request["stream"] = json!(true);
        request["stream_options"] = json!({"include_usage": true});
    }
    let (status, body) = send(&harness.app, request).await;
    assert_eq!(status, 200, "{body}");
    match stream {
        true => assert!(body.contains("\"content\":\"hi\""), "{body}"),
        false => assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["model"],
            RESPONSE_MODEL
        ),
    }

    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(
        sent.path,
        format!("/openai/deployments/{deployment}/chat/completions")
    );
    assert_eq!(sent.query.as_deref(), Some("api-version=2024-10-21"));
    assert_eq!(sent.api_key.as_deref(), Some("sk-azure"));
    assert_eq!(sent.authorization, None);
    assert_eq!(sent.body["model"], deployment);
    assert_eq!(sent.body["max_tokens"], 8);
    assert_eq!(sent.body["user"], "end-user-1");
    assert_eq!(sent.body["stream"].as_bool().unwrap_or(false), stream);

    litellm_gateway::hooks::close(&harness.inference).await;
    let seen = harness.seen.0.lock().unwrap();
    let [seen] = seen.as_slice() else {
        panic!("{seen:?}")
    };
    assert_eq!(seen.deployment.as_deref(), Some(id));
    assert_eq!(seen.usage, Some(usage));
    assert!(close(seen.cost.unwrap(), cost(usage, rates)), "{seen:?}");
}

#[rstest]
#[tokio::test]
async fn an_unsupported_param_is_refused_before_the_call() {
    let harness = harness().await;
    let (status, body) = send(
        &harness.app,
        json!({
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("reasoning_effort"), "{body}");
    assert!(harness.calls.lock().unwrap().is_empty());
}

#[rstest]
#[case::unary(false, (5, 6))]
#[case::stream(true, (7, 3))]
#[tokio::test]
async fn messages_for_an_azure_deployment_are_priced_by_the_response_model(
    #[case] stream: bool,
    #[case] usage: (u64, u64),
) {
    let harness = harness().await;
    let (status, body) = send_to(
        &harness.app,
        "/v1/messages",
        json!({"model": "chat", "max_tokens": 8, "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    match stream {
        true => assert!(body.contains("event: message_stop"), "{body}"),
        false => assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["model"],
            RESPONSE_MODEL
        ),
    }
    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(sent.path, "/openai/deployments/chat-prod/chat/completions");

    litellm_gateway::hooks::close(&harness.inference).await;
    let seen = harness.seen.0.lock().unwrap();
    let [seen] = seen.as_slice() else {
        panic!("{seen:?}")
    };
    assert_eq!(seen.deployment.as_deref(), Some("azure-1"));
    assert_eq!(seen.usage, Some(usage));
    assert!(
        close(seen.cost.unwrap(), cost(usage, RESPONSE_MODEL_RATES)),
        "{seen:?}"
    );
}
