//! Chat completions for the OpenAI-compatible providers through the gateway: the URL, auth
//! and exact body each provider's upstream sees, and the usage and cost a stream is recorded
//! with whether or not the client asked to see the usage.

use std::{
    collections::BTreeMap,
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

const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#;
const CONTENT_CHUNK: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n";
const USAGE_CHUNK: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n";
/// xAI counts reasoning apart from `completion_tokens` but inside `total_tokens`.
const XAI_USAGE_CHUNK: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":14,\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\n";
/// Mistral ends its stream with the usage on the last content chunk, unasked.
const MISTRAL_LAST_CHUNK: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n";
const DONE: &str = "data: [DONE]\n\n";

const RATES: (f64, f64) = (1.0, 2.0);

fn use_local_cost_map() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let price = json!({
            "mode": "chat",
            "input_cost_per_token": RATES.0,
            "output_cost_per_token": RATES.1,
        });
        let map: serde_json::Map<String, Value> = [
            ("groq/llama-3.1-8b-instant", "groq"),
            ("mistral/mistral-small-latest", "mistral"),
            ("xai/grok-3", "xai"),
            (
                "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
                "hosted_vllm",
            ),
        ]
        .into_iter()
        .map(|(model, provider)| {
            let mut entry = price.clone();
            entry["litellm_provider"] = json!(provider);
            (model.to_owned(), entry)
        })
        .collect();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(Value::Object(map).to_string().as_bytes())
            .unwrap();
        let (_, path) = file.keep().unwrap();
        // SAFETY: set once, before any test builds a gateway or spawns a thread that reads the
        // environment.
        unsafe { std::env::set_var("LITELLM_MODEL_COST_MAP_PATH", path) };
    });
}

#[derive(Clone, Debug)]
struct Sent {
    path: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

type Calls = Arc<Mutex<Vec<Sent>>>;

fn stream_answer(path: &str) -> String {
    match path.split('/').nth(1) {
        Some("xai") => [CONTENT_CHUNK, XAI_USAGE_CHUNK, DONE].concat(),
        Some("mistral") => [CONTENT_CHUNK, MISTRAL_LAST_CHUNK, DONE].concat(),
        Some("vllm") => [
            "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"hmm\"}}]}\n\n",
            CONTENT_CHUNK,
            USAGE_CHUNK,
            DONE,
        ]
        .concat(),
        _ => [CONTENT_CHUNK, USAGE_CHUNK, DONE].concat(),
    }
}

async fn upstream(calls: Calls) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let calls = calls.clone();
        async move {
            let (parts, body) = request.into_parts();
            let headers = parts
                .headers
                .iter()
                .filter_map(|(name, value)| {
                    Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
                })
                .collect();
            let path = parts.uri.path().to_owned();
            let body: Value =
                serde_json::from_slice(&to_bytes(body, 1 << 20).await.unwrap()).unwrap();
            let stream = body["stream"].as_bool().unwrap_or(false);
            calls.lock().unwrap().push(Sent {
                path: path.clone(),
                headers,
                body,
            });
            match stream {
                true => {
                    ([(CONTENT_TYPE, "text/event-stream")], stream_answer(&path)).into_response()
                }
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

/// One deployment per provider, each at its own path on the mock upstream.
const DEPLOYMENTS: &[(&str, &str, &str, Option<&str>)] = &[
    (
        "vllm",
        "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        "/vllm/v1",
        None,
    ),
    (
        "nim",
        "nvidia_nim/meta/llama3-70b-instruct",
        "/nim/v1",
        Some("sk-nim"),
    ),
    (
        "groq",
        "groq/llama-3.1-8b-instant",
        "/groq/openai/v1",
        Some("sk-groq"),
    ),
    (
        "together",
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "/together/v1",
        Some("sk-together"),
    ),
    (
        "fireworks",
        "fireworks_ai/llama-v3p1-8b-instruct",
        "/fireworks/inference/v1",
        Some("sk-fireworks"),
    ),
    ("xai", "xai/grok-3", "/xai/v1", Some("sk-xai")),
    (
        "mistral",
        "mistral/mistral-small-latest",
        "/mistral",
        Some("sk-mistral"),
    ),
    (
        "deepseek",
        "deepseek/deepseek-chat",
        "/deepseek/beta",
        Some("sk-deepseek"),
    ),
    (
        "openrouter",
        "openrouter/openai/gpt-4o",
        "/openrouter/api/v1",
        Some("sk-openrouter"),
    ),
    (
        "cerebras",
        "cerebras/llama3.1-8b",
        "/cerebras/v1",
        Some("sk-cerebras"),
    ),
    (
        "perplexity",
        "perplexity/sonar",
        "/perplexity",
        Some("sk-perplexity"),
    ),
    (
        "custom-openai",
        "openai/my-served-model",
        "/custom/v1",
        Some("sk-custom"),
    ),
];

async fn harness() -> Harness {
    use_local_cost_map();
    let calls = Calls::default();
    let upstream = upstream(calls.clone()).await;
    let models: String = DEPLOYMENTS
        .iter()
        .map(|(name, model, base, key)| {
            let key = key.map_or_else(String::new, |key| format!("\n      api_key: {key}"));
            format!(
                "  - model_name: {name}
    litellm_params:
      model: {model}
      api_base: {upstream}{base}{key}
    model_info:
      id: {name}-1
"
            )
        })
        .collect();
    let config = Config::from_yaml(&format!(
        "model_list:
{models}router_settings:
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
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-master")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

async fn seen(harness: &Harness) -> Vec<Seen> {
    litellm_gateway::hooks::close(&harness.inference).await;
    std::mem::take(&mut *harness.seen.0.lock().unwrap())
}

fn hi(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 8,
        "temperature": 0.5,
        "metadata": {"trace": "t"},
    })
}

fn messages() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

#[rstest]
#[case::hosted_vllm("vllm", "/vllm/v1/chat/completions", "Bearer fake-api-key", None,
    json!({"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": messages(), "max_completion_tokens": 8, "temperature": 0.5}))]
#[case::nvidia_nim("nim", "/nim/v1/chat/completions", "Bearer sk-nim", None,
    json!({"model": "meta/llama3-70b-instruct", "messages": messages(), "max_tokens": 8, "temperature": 0.5}))]
#[case::groq("groq", "/groq/openai/v1/chat/completions", "Bearer sk-groq", None,
    json!({"model": "llama-3.1-8b-instant", "messages": messages(), "max_tokens": 8, "temperature": 0.5}))]
#[case::together_ai("together", "/together/v1/chat/completions", "Bearer sk-together", None,
    json!({"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "messages": messages(), "max_completion_tokens": 8, "temperature": 0.5}))]
#[case::fireworks_ai("fireworks", "/fireworks/inference/v1/chat/completions", "Bearer sk-fireworks", None,
    json!({"model": "accounts/fireworks/models/llama-v3p1-8b-instruct", "messages": messages(), "max_tokens": 8, "temperature": 0.5}))]
#[case::xai("xai", "/xai/v1/chat/completions", "Bearer sk-xai", None,
    json!({"model": "grok-3", "messages": messages(), "max_tokens": 8, "temperature": 0.5}))]
#[case::mistral("mistral", "/mistral/v1/chat/completions", "Bearer sk-mistral", None,
    json!({"model": "mistral-small-latest", "messages": messages(), "max_tokens": 8, "temperature": 0.5}))]
#[case::deepseek("deepseek", "/deepseek/beta/chat/completions", "Bearer sk-deepseek", None,
    json!({"model": "deepseek-chat", "messages": messages(), "max_completion_tokens": 8, "temperature": 0.5}))]
#[case::openrouter("openrouter", "/openrouter/api/v1/chat/completions", "Bearer sk-openrouter",
    Some(("x-title", "liteLLM")),
    json!({"model": "openai/gpt-4o", "messages": messages(), "max_completion_tokens": 8, "temperature": 0.5, "usage": {"include": true}}))]
#[case::cerebras("cerebras", "/cerebras/v1/chat/completions", "Bearer sk-cerebras", None,
    json!({"model": "llama3.1-8b", "messages": messages(), "max_tokens": 8, "temperature": 0.5}))]
#[case::perplexity("perplexity", "/perplexity/chat/completions", "Bearer sk-perplexity",
    Some(("x-pplx-integration", "litellm")),
    json!({"model": "sonar", "messages": messages(), "max_completion_tokens": 8, "temperature": 0.5}))]
#[case::openai_custom_base("custom-openai", "/custom/v1/chat/completions", "Bearer sk-custom", None,
    json!({"model": "my-served-model", "messages": messages(), "max_completion_tokens": 8, "temperature": 0.5}))]
#[tokio::test]
async fn each_provider_gets_its_url_auth_and_body(
    #[case] model: &str,
    #[case] path: &str,
    #[case] authorization: &str,
    #[case] header: Option<(&str, &str)>,
    #[case] body: Value,
) {
    let harness = harness().await;
    let (status, answer) = send(&harness.app, hi(model)).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(
        serde_json::from_str::<Value>(&answer).unwrap()["choices"][0]["message"]["content"],
        "hi"
    );

    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(sent.path, path);
    assert_eq!(
        sent.headers.get("authorization").map(String::as_str),
        Some(authorization)
    );
    if let Some((name, value)) = header {
        assert_eq!(sent.headers.get(name).map(String::as_str), Some(value));
    }
    assert_eq!(sent.body, body);

    let seen = seen(&harness).await;
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].deployment, Some(format!("{model}-1")));
    assert_eq!(seen[0].usage, Some((5, 6)));
}

#[rstest]
#[case::groq_asks_on_the_clients_behalf("groq", false, Some(json!({"include_usage": true})), (7, 3))]
#[case::groq_client_asked("groq", true, Some(json!({"include_usage": true})), (7, 3))]
#[case::mistral_reports_unasked("mistral", false, None, (7, 3))]
#[case::xai_folds_reasoning_into_completion("xai", false, Some(json!({"include_usage": true})), (7, 7))]
#[case::hosted_vllm("vllm", false, Some(json!({"include_usage": true})), (7, 3))]
#[tokio::test]
async fn a_stream_is_metered_and_priced(
    #[case] model: &str,
    #[case] client_usage: bool,
    #[case] stream_options: Option<Value>,
    #[case] usage: (u64, u64),
) {
    let harness = harness().await;
    let mut request = hi(model);
    request["stream"] = json!(true);
    if client_usage {
        request["stream_options"] = json!({"include_usage": true});
    }
    let (status, body) = send(&harness.app, request).await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("\"content\":\"hi\""), "{body}");
    assert!(body.ends_with(DONE), "{body}");
    assert_eq!(body.contains("\"usage\":{"), client_usage, "{body}");

    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(sent.body.get("stream_options").cloned(), stream_options);

    let seen = seen(&harness).await;
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].usage, Some(usage));
    let expected = usage.0 as f64 * RATES.0 + usage.1 as f64 * RATES.1;
    assert!((seen[0].cost.unwrap() - expected).abs() < 1e-9, "{seen:?}");
}

#[rstest]
#[tokio::test]
async fn a_vllm_stream_names_its_reasoning_as_litellm_does() {
    let harness = harness().await;
    let mut request = hi("vllm");
    request["stream"] = json!(true);
    let (status, body) = send(&harness.app, request).await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("\"reasoning_content\":\"hmm\""), "{body}");
    assert!(!body.contains("\"reasoning\":"), "{body}");
}

#[rstest]
#[case::groq_streamed_response_format("groq", json!({"stream": true, "response_format": {"type": "json_object"}}), 501)]
#[case::param_the_model_lacks("cerebras", json!({"frequency_penalty": 1}), 400)]
#[case::fireworks_file_part("fireworks", json!({"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": "f"}}]}]}), 400)]
#[tokio::test]
async fn requests_a_provider_cannot_take_are_refused_before_the_call(
    #[case] model: &str,
    #[case] extra: Value,
    #[case] status: u16,
) {
    let harness = harness().await;
    let mut request = hi(model);
    let (Value::Object(request_fields), Value::Object(extra)) = (&mut request, extra) else {
        unreachable!()
    };
    request_fields.extend(extra);
    let (sent_status, body) = send(&harness.app, request).await;
    assert_eq!(sent_status, status, "{body}");
    assert!(harness.calls.lock().unwrap().is_empty());
}
