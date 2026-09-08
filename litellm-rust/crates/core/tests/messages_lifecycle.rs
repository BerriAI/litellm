use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use litellm_core::Error;
use litellm_core::integrations::custom_logger::{LogError, LogFuture};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, Clock, RequestPolicy, TerminalClassification,
    TerminalDispatcher, TerminalRecord,
};
use litellm_core::messages::lifecycle::{Options, messages};
use litellm_core::messages::types::MessagesRequest;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

type PolicyFuture<'a, T> = Pin<Box<dyn Future<Output = ActionResult<T, Error>> + Send + 'a>>;

#[derive(Default)]
struct Services {
    reject: bool,
    terminals: Mutex<Vec<TerminalRecord>>,
}

impl Clock for Services {
    fn now(&self) -> f64 {
        10.0
    }
}

impl RequestPolicy<MessagesRequest, MessagesRequest> for Services {
    type PreCallFuture<'a> = PolicyFuture<'a, MessagesRequest>;
    type DuringCallFuture<'a> = PolicyFuture<'a, MessagesRequest>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move {
            if self.reject {
                ActionResult::Reject(Error::InvalidRequest("blocked".into()))
            } else {
                ActionResult::Continue(request)
            }
        })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { ActionResult::Continue(request) })
    }
}

impl TerminalDispatcher for Services {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async move {
            self.terminals.lock().unwrap().push(terminal.clone());
            Ok::<(), LogError>(())
        })
    }
}

fn request(api_base: String) -> MessagesRequest {
    MessagesRequest {
        model: "claude-test".into(),
        body: json!({
            "model": "claude-test",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hello"}]
        }),
        api_key: Some("test-key".into()),
        api_base: Some(api_base),
        custom_llm_provider: Some("anthropic".into()),
        extra_headers: None,
        timeout: None,
    }
}

fn context() -> CallLifecycleContext {
    CallLifecycleContext::new("messages", "claude-test", "anthropic", "call-1")
}

async fn upstream(status: u16) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0_u8; 4096];
        let _ = socket.read(&mut buffer).await.unwrap();
        let body = if status == 200 {
            r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":7}}"#
        } else {
            r#"{"error":"failed"}"#
        };
        let response = format!(
            "HTTP/1.1 {status} Test\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    (format!("http://{address}"), server)
}

#[tokio::test]
async fn success_dispatches_exactly_one_terminal() {
    let (api_base, server) = upstream(200).await;
    let services = Services::default();
    let result = messages(&services, request(api_base), Options::default(), context()).await;

    result.into_result().expect("messages succeeds");
    server.await.unwrap();
    let terminals = services.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0].classification, TerminalClassification::Success);
    assert_eq!(terminals[0].usage.prompt_tokens, 11);
    assert_eq!(terminals[0].usage.completion_tokens, 7);
    assert_eq!(terminals[0].usage.total_tokens, 18);
}

#[tokio::test]
async fn provider_failure_dispatches_exactly_one_terminal() {
    let (api_base, server) = upstream(500).await;
    let services = Services::default();
    let result = messages(&services, request(api_base), Options::default(), context()).await;

    assert!(matches!(
        result.into_result(),
        Err(Error::Http { status: 500, .. })
    ));
    server.await.unwrap();
    let terminals = services.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        terminals[0].classification,
        TerminalClassification::Failure { .. }
    ));
}

#[tokio::test]
async fn pre_call_rejection_never_touches_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}", listener.local_addr().unwrap());
    let services = Services {
        reject: true,
        ..Default::default()
    };
    let result = messages(&services, request(api_base), Options::default(), context()).await;

    assert!(matches!(
        result.into_result(),
        Err(Error::InvalidRequest(_))
    ));
    assert_eq!(services.terminals.lock().unwrap().len(), 1);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}
