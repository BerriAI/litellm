use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use litellm_core::Error;
use litellm_core::chat_completions::types::ChatCompletionsRequest;
use litellm_core::lifecycle::CallLifecycleContext;
use litellm_core::providers::auth::{AwsMechanisms, Environment, SigningClock};
use litellm_core::runtime::{
    CallServices, ChatCompletionsServices, HttpFuture, HttpRequest, HttpResponse, HttpStreamFuture,
    HttpTransport, LiteLlm, SessionFuture,
};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const RESPONSE: &str = r#"{"model":"claude-sonnet-4-5","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}"#;

#[derive(Clone)]
struct RecordingTransport {
    requests: Arc<Mutex<Vec<HttpRequest>>>,
    response: &'static str,
}

impl HttpTransport for RecordingTransport {
    fn execute(&self, request: HttpRequest) -> HttpFuture<'_> {
        self.requests.lock().unwrap().push(request);
        Box::pin(std::future::ready(Ok(HttpResponse {
            status: 200,
            body: self.response.as_bytes().to_vec(),
        })))
    }

    fn execute_stream(&self, _: HttpRequest) -> HttpStreamFuture<'_> {
        Box::pin(std::future::ready(Err(Error::Unsupported(
            "recording transport streaming",
        ))))
    }
}

#[derive(Clone)]
struct RecordingCalls {
    opened: Arc<Mutex<Vec<(String, u64)>>>,
}

struct RecordingSession;

impl CallServices for RecordingCalls {
    type Bindings = u64;
    type Session = RecordingSession;
    type OpenFuture<'a> = SessionFuture<'a, RecordingSession>;

    fn open<'a>(
        &'a self,
        context: CallLifecycleContext,
        bindings: Self::Bindings,
    ) -> Self::OpenFuture<'a> {
        self.opened
            .lock()
            .unwrap()
            .push((context.litellm_call_id, bindings));
        Box::pin(std::future::ready(Ok(RecordingSession)))
    }
}

struct Services {
    transport: RecordingTransport,
    calls: RecordingCalls,
    environment_reads: AtomicUsize,
    #[cfg(feature = "bedrock-auth")]
    aws_credentials: litellm_core::providers::auth::AwsCredentialState,
}

impl Environment for Services {
    fn environment(&self, _: &str) -> Option<String> {
        self.environment_reads.fetch_add(1, Ordering::Relaxed);
        None
    }
}

impl SigningClock for Services {
    fn signing_time(&self) -> SystemTime {
        SystemTime::now()
    }
}

impl AwsMechanisms for Services {
    #[cfg(feature = "bedrock-auth")]
    fn aws_credential_state(&self) -> &litellm_core::providers::auth::AwsCredentialState {
        &self.aws_credentials
    }
}

impl ChatCompletionsServices for Services {
    type Transport = RecordingTransport;
    type Calls = RecordingCalls;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

fn services() -> Services {
    Services {
        transport: RecordingTransport {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: RESPONSE,
        },
        calls: RecordingCalls {
            opened: Arc::new(Mutex::new(Vec::new())),
        },
        environment_reads: AtomicUsize::new(0),
        #[cfg(feature = "bedrock-auth")]
        aws_credentials: litellm_core::providers::auth::AwsCredentialState::with_clock(
            Arc::new(litellm_auth_aws::NativeCredentialRuntime),
            8,
            Arc::new(litellm_auth_aws::SystemClock),
        ),
    }
}

#[cfg(feature = "bedrock-auth")]
#[tokio::test]
async fn bedrock_client_signs_the_exact_body_given_to_the_transport() {
    const BEDROCK_RESPONSE: &str = r#"{"output":{"message":{"content":[{"text":"hello"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}"#;
    let mut services = services();
    services.transport.response = BEDROCK_RESPONSE;
    let client = LiteLlm::from_services(services);
    let call = ChatCompletionsRequest {
        model: "bedrock/us-east-1/anthropic.claude-v2",
        messages: json!([{"role": "user", "content": "signed"}]),
        optional_params: Map::from_iter([
            ("maxTokens".into(), json!(16)),
            ("aws_access_key_id".into(), json!("test-access")),
            ("aws_secret_access_key".into(), json!("test-secret")),
        ]),
        api_key: None,
        api_base: Some("https://recording.invalid"),
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    };

    let response = client
        .chat_completions_with(call, context("bedrock"), 33)
        .await
        .unwrap();
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );

    let requests = client.services().transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization")
            && value.starts_with("AWS4-HMAC-SHA256 Credential=test-access/")
    }));
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "inferenceConfig": {"maxTokens": 16},
            "messages": [{"role": "user", "content": [{"text": "signed"}]}]
        })
    );
}

fn request(message: &str, optional_params: Map<String, Value>) -> ChatCompletionsRequest<'_> {
    ChatCompletionsRequest {
        model: "anthropic/claude-sonnet-4-5",
        messages: json!([{"role": "user", "content": message}]),
        optional_params,
        api_key: Some("sk-test"),
        api_base: Some("https://recording.invalid/v1/messages"),
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    }
}

fn context(call_id: &str) -> CallLifecycleContext {
    CallLifecycleContext::new(
        "chat_completion",
        "anthropic/claude-sonnet-4-5",
        "anthropic",
        call_id,
    )
}

#[tokio::test]
async fn admission_failure_opens_no_session_and_performs_no_effects() {
    let client = LiteLlm::from_services(services());
    let error = client
        .chat_completions_with(
            request("declined", Map::from_iter([("stream".into(), json!(true))])),
            context("declined"),
            1,
        )
        .await
        .expect_err("streaming is not admitted");

    assert_eq!(error, Error::Unsupported("streaming"));
    assert!(client.services().calls.opened.lock().unwrap().is_empty());
    assert!(
        client
            .services()
            .transport
            .requests
            .lock()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        client.services().environment_reads.load(Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn clones_share_application_services_but_open_isolated_sessions() {
    let client = LiteLlm::from_services(services());
    let clone = client.clone();
    assert!(std::ptr::eq(client.services(), clone.services()));

    let (first, second) = tokio::join!(
        client.chat_completions_with(request("first", Map::new()), context("outer"), 11),
        clone.chat_completions_with(request("second", Map::new()), context("inner"), 22),
    );
    assert_eq!(
        first.unwrap().choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert_eq!(
        second.unwrap().choices[0].message.content.as_deref(),
        Some("hello")
    );

    let opened = client.services().calls.opened.lock().unwrap();
    assert_eq!(
        opened.as_slice(),
        &[("outer".to_string(), 11), ("inner".to_string(), 22)]
    );
    drop(opened);

    let requests = client.services().transport.requests.lock().unwrap();
    let bodies: Vec<Value> = requests
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    assert_eq!(bodies[0]["messages"][0]["content"][0]["text"], "first");
    assert_eq!(bodies[1]["messages"][0]["content"][0]["text"], "second");
    assert!(requests.iter().all(|request| {
        request.method == reqwest::Method::POST
            && request
                .headers
                .iter()
                .any(|(name, value)| name == "x-api-key" && value == "sk-test")
    }));
}

#[tokio::test]
async fn native_client_executes_the_same_route_against_a_recording_server() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}/v1/messages", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut received = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            received.extend_from_slice(&buffer[..read]);
            let header_end = received.windows(4).position(|window| window == b"\r\n\r\n");
            let complete = header_end.is_some_and(|header_end| {
                let headers = String::from_utf8_lossy(&received[..header_end]);
                let length = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                });
                length.is_some_and(|length| received.len() >= header_end + 4 + length)
            });
            if read == 0 || complete {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            RESPONSE.len(),
            RESPONSE
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        received
    });

    let client = LiteLlm::new();
    let call = ChatCompletionsRequest {
        api_base: Some(&api_base),
        ..request("native", Map::new())
    };
    let response = client.chat_completions(call).await.unwrap();
    let received = String::from_utf8(server.await.unwrap()).unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert!(received.contains("x-api-key: sk-test"));
    assert!(received.contains("native"));
}
