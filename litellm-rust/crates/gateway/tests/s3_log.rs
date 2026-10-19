//! The `s3_v2` observer against a mock S3: a request ends up as a signed PUT under the proxy's
//! key layout, a 503 is retried, and an object S3 rejects is dropped instead of kept.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use base64::Engine;
use litellm_auth_aws::{Credentials, sign_s3};
use litellm_config::Config;
use litellm_gateway::s3_log::S3Log;
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyHash, KeyLookup, KeyModels, KeyProfile, KeyStatus, LookupFuture,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path_regex},
};

const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#;
const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":4}}"#;
const OBJECT_KEY: &str =
    r"^/logs/audit/research/\d{4}-\d{2}-\d{2}/time-\d{2}-\d{2}-\d{2}-\d{6}_chatcmpl-1\.json$";

/// Answers Anthropic and OpenAI calls.
async fn upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(|request: Request<Body>| async move {
        let body = match request.uri().path().ends_with("/chat/completions") {
            true => COMPLETION,
            false => MESSAGE,
        };
        ([(CONTENT_TYPE, "application/json")], body).into_response()
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

struct Keys(HashMap<KeyHash, KeyStatus>);

impl KeyLookup for Keys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { Ok(self.0.get(hash).cloned()) })
    }
}

fn keys() -> Arc<dyn KeyLookup> {
    let status = KeyStatus::Active {
        expires_at: None,
        models: KeyModels::default(),
        allowed_routes: AllowedRoutes::default(),
        team_id: Some("team-1".into()),
        profile: Arc::new(KeyProfile {
            alias: Some("team-key".into()),
            user_id: Some("user-1".into()),
            team_alias: Some("research".into()),
            ..KeyProfile::default()
        }),
    };
    let hash = KeyHash::from_token(&litellm_auth_types::SecretValue::new("sk-team"));
    Arc::new(Keys(HashMap::from([(hash, status)])))
}

struct Harness {
    app: Router,
    inference: Arc<litellm_gateway_inference::Gateway>,
    s3_log: Arc<S3Log>,
}

async fn harness(s3: &MockServer) -> Harness {
    let upstream = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: gpt
    litellm_params:
      model: openai/gpt-x
      api_base: {upstream}/v1
      api_key: sk-upstream
    model_info:
      id: gpt-1
  - model_name: claude
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
litellm_settings:
  callbacks: [s3_v2]
  s3_callback_params:
    s3_bucket_name: logs
    s3_path: audit
    s3_region_name: us-east-1
    s3_endpoint_url: {endpoint}
    s3_aws_access_key_id: AKIDTEST
    s3_aws_secret_access_key: secret
    s3_use_team_prefix: true
    s3_server_side_encryption: AES256
",
        endpoint = s3.uri(),
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let s3_log = S3Log::from_config(&config, &inference)
        .await
        .unwrap()
        .unwrap();
    litellm_gateway::hooks::install(&inference, &config, Vec::new(), vec![s3_log.clone()])
        .await
        .unwrap();
    let app =
        litellm_gateway::router_with_keys(inference.clone(), &config, None, None, Some(keys()));
    Harness {
        app,
        inference,
        s3_log,
    }
}

async fn send(app: &Router, path: &str, body: Value) {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-team")
                .header("user-agent", "client/1.0")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    assert_eq!(status, 200, "{body:?}");
}

/// `YYYYMMDDTHHMMSSZ` as a time.
fn amz_date(text: &str) -> SystemTime {
    let number = |range: std::ops::Range<usize>| text[range].parse::<i64>().unwrap();
    let (year, month, day) = (number(0..4), number(4..6), number(6..8));
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    let seconds = days * 86_400 + number(9..11) * 3600 + number(11..13) * 60 + number(13..15);
    UNIX_EPOCH + Duration::from_secs(seconds as u64)
}

#[tokio::test]
async fn a_request_is_written_as_a_signed_put_and_a_503_is_retried() {
    let s3 = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path_regex(OBJECT_KEY))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&s3)
        .await;
    Mock::given(method("PUT"))
        .and(path_regex(OBJECT_KEY))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&s3)
        .await;
    let harness = harness(&s3).await;
    send(
        &harness.app,
        "/v1/chat/completions",
        json!({"model": "gpt", "temperature": 0.5, "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;

    litellm_gateway::hooks::close(&harness.inference).await;

    let requests = s3.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2, "the 503 is retried once");
    assert_eq!(requests[0].url, requests[1].url);
    let put = &requests[1];
    let header = |name: &str| {
        put.headers
            .get(name)
            .map(|value| value.to_str().unwrap().to_owned())
    };
    let payload: Value = serde_json::from_slice(&put.body).unwrap();
    assert_eq!(payload["id"], "chatcmpl-1");
    assert_eq!(payload["status"], "success");
    assert_eq!(payload["model_group"], "gpt");
    assert_eq!(payload["model_id"], "gpt-1");
    assert_eq!(
        payload["messages"],
        json!([{"role": "user", "content": "hello"}])
    );
    assert_eq!(
        payload["response"]["choices"][0]["message"]["content"],
        "hi"
    );
    assert_eq!(payload["metadata"]["user_api_key_team_alias"], "research");
    assert_eq!(payload["total_tokens"], 11);

    assert_eq!(header("content-type").as_deref(), Some("application/json"));
    assert_eq!(header("content-language").as_deref(), Some("en"));
    assert_eq!(
        header("cache-control").as_deref(),
        Some("private, immutable, max-age=31536000, s-maxage=0")
    );
    assert_eq!(
        header("x-amz-server-side-encryption").as_deref(),
        Some("AES256")
    );
    let disposition = header("content-disposition").unwrap();
    assert!(
        disposition.starts_with("inline; filename=\"time-")
            && disposition.ends_with("_chatcmpl-1.json\""),
        "{disposition}"
    );
    let sha256: String = Sha256::digest(&put.body)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(header("x-amz-content-sha256"), Some(sha256));
    let md5 = base64::engine::general_purpose::STANDARD
        .encode(<md5::Md5 as md5::Digest>::digest(&put.body));
    assert_eq!(header("content-md5"), Some(md5));

    let signed: BTreeMap<String, String> = [
        "content-type",
        "content-md5",
        "x-amz-content-sha256",
        "content-language",
        "content-disposition",
        "cache-control",
        "x-amz-server-side-encryption",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), header(name).unwrap()))
    .collect();
    let expected = sign_s3(
        "PUT",
        &format!("{}{}", s3.uri(), put.url.path()),
        &put.body,
        &signed,
        "us-east-1",
        &Credentials::new("AKIDTEST", "secret", None, None, "test"),
        amz_date(&header("x-amz-date").unwrap()),
    )
    .unwrap();
    assert_eq!(header("x-amz-date"), Some(expected["X-Amz-Date"].clone()));
    assert_eq!(
        header("authorization"),
        Some(expected["Authorization"].clone())
    );
    assert!(
        expected["Authorization"].contains("SignedHeaders=cache-control;content-disposition;content-language;content-md5;content-type;host;x-amz-content-sha256;x-amz-date;x-amz-server-side-encryption"),
        "{}",
        expected["Authorization"]
    );
}

#[tokio::test]
async fn an_object_s3_rejects_is_dropped_while_the_rest_are_written() {
    let s3 = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path_regex(r"_msg_1\.json$"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_string("<Error><Code>EntityTooLarge</Code></Error>"),
        )
        .expect(1)
        .mount(&s3)
        .await;
    Mock::given(method("PUT"))
        .and(path_regex(OBJECT_KEY))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&s3)
        .await;
    let harness = harness(&s3).await;
    send(
        &harness.app,
        "/v1/chat/completions",
        json!({"model": "gpt", "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;
    send(
        &harness.app,
        "/v1/messages",
        json!({"model": "claude", "max_tokens": 8, "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;

    litellm_gateway::hooks::close(&harness.inference).await;
    harness.s3_log.flush().await;

    assert_eq!(s3.received_requests().await.unwrap().len(), 2);
}
