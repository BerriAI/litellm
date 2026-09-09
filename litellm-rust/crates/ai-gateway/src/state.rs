use std::sync::Arc;

use crate::io::realtime_pool::RealtimePool;
use litellm_core::Error;
use litellm_core::chat_completions::types::{
    ChatCompletionsRequest, ResolvedChatCompletionsRequest,
};
use litellm_core::integrations::custom_logger::{CustomLogger, CustomLoggerRunner, LogFuture};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, Clock, DeploymentFailureHooks, DeploymentPreHooks,
    DeploymentSuccessHooks, ModerationHooks, PreCallHooks, TerminalDispatcher, TerminalRecord,
};
use litellm_core::messages::types::MessagesRequest;
use litellm_core::router::Router;
use litellm_core::runtime::{
    CallServices, ChatCompletionsServices, LiteLlm, MessagesRuntimeServices, NativeHttpTransport,
};

use litellm_gateway_auth::MasterKeyState;

type EnvironmentLookup = dyn Fn(&str) -> Option<String> + Send + Sync;

#[derive(Clone, Copy)]
pub struct GatewayMessagesCallServices;

pub struct GatewayMessagesSession {
    runner: CustomLoggerRunner,
}

impl GatewayMessagesSession {
    pub fn new(loggers: Arc<Vec<Arc<dyn CustomLogger>>>) -> Self {
        Self {
            runner: CustomLoggerRunner::new(loggers.as_ref().clone()),
        }
    }
}

impl CallServices for GatewayMessagesCallServices {
    type Bindings = Arc<Vec<Arc<dyn CustomLogger>>>;
    type Session = GatewayMessagesSession;
    type OpenFuture<'a> = std::future::Ready<Result<GatewayMessagesSession, Error>>;

    fn open<'a>(
        &'a self,
        _: CallLifecycleContext,
        bindings: Self::Bindings,
    ) -> Self::OpenFuture<'a> {
        std::future::ready(Ok(GatewayMessagesSession::new(bindings)))
    }
}

impl Clock for GatewayMessagesSession {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }
}

impl PreCallHooks<MessagesRequest> for GatewayMessagesSession {
    type PreCallFuture<'a> = std::future::Ready<ActionResult<MessagesRequest, Error>>;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl ModerationHooks<MessagesRequest> for GatewayMessagesSession {
    type ModerationFuture<'a> = std::future::Ready<ActionResult<MessagesRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::ModerationFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl DeploymentPreHooks<MessagesRequest> for GatewayMessagesSession {}
impl DeploymentSuccessHooks<litellm_core::messages::types::AnthropicMessagesResponse>
    for GatewayMessagesSession
{
}
impl DeploymentFailureHooks for GatewayMessagesSession {}

impl<'request> PreCallHooks<ChatCompletionsRequest<'request>> for GatewayMessagesSession {
    type PreCallFuture<'a>
        = std::future::Ready<ActionResult<ChatCompletionsRequest<'request>, Error>>
    where
        Self: 'a;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: ChatCompletionsRequest<'request>,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl<'request> ModerationHooks<ResolvedChatCompletionsRequest<'request>>
    for GatewayMessagesSession
{
    type ModerationFuture<'a>
        = std::future::Ready<ActionResult<ResolvedChatCompletionsRequest<'request>, Error>>
    where
        Self: 'a;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: ResolvedChatCompletionsRequest<'request>,
    ) -> Self::ModerationFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl<'request> DeploymentPreHooks<ChatCompletionsRequest<'request>> for GatewayMessagesSession {}
impl DeploymentSuccessHooks<litellm_core::chat_completions::types::ChatCompletionsResponse>
    for GatewayMessagesSession
{
}

impl TerminalDispatcher for GatewayMessagesSession {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        self.runner.dispatch(terminal)
    }
}

pub struct GatewayMessagesServices {
    transport: NativeHttpTransport,
    calls: GatewayMessagesCallServices,
    environment: Arc<EnvironmentLookup>,
    aws_credentials: litellm_core::providers::auth::AwsCredentialState,
}

impl GatewayMessagesServices {
    pub fn new(environment: impl Fn(&str) -> Option<String> + Send + Sync + 'static) -> Self {
        Self {
            transport: NativeHttpTransport::new(),
            calls: GatewayMessagesCallServices,
            environment: Arc::new(environment),
            aws_credentials: litellm_core::providers::auth::native_aws_credential_state(),
        }
    }
}

impl litellm_core::providers::auth::Environment for GatewayMessagesServices {
    fn environment(&self, key: &str) -> Option<String> {
        (self.environment)(key)
    }
}

impl litellm_core::providers::auth::SigningClock for GatewayMessagesServices {
    fn signing_time(&self) -> std::time::SystemTime {
        std::time::SystemTime::now()
    }
}

impl litellm_core::providers::auth::AwsMechanisms for GatewayMessagesServices {
    fn aws_credential_state(&self) -> &litellm_core::providers::auth::AwsCredentialState {
        &self.aws_credentials
    }
}

impl ChatCompletionsServices for GatewayMessagesServices {
    type Transport = NativeHttpTransport;
    type Calls = GatewayMessagesCallServices;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

impl MessagesRuntimeServices for GatewayMessagesServices {
    type Transport = NativeHttpTransport;
    type Calls = GatewayMessagesCallServices;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

/// Shared application state handed to every route handler.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// The gateway master key. Any caller presenting it as a bearer token may
    /// invoke the gateway. `None` → auth not configured (routes fail closed).
    pub master_key: Option<Arc<str>>,
    /// Logging callbacks fanned out at the end of each realtime session.
    pub loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    /// Pre-warmed upstream realtime connection pool. Disabled
    /// (`RealtimePool::disabled()`) when `REALTIME_POOL_SIZE=0`, in which case
    /// every realtime connect fresh-dials exactly as before.
    pub realtime_pool: Arc<RealtimePool>,
    pub messages_client: LiteLlm<GatewayMessagesServices>,
}

impl MasterKeyState for AppState {
    fn master_key(&self) -> Option<&str> {
        self.master_key.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use litellm_core::chat_completions::types::ChatCompletionsRequest;
    use litellm_core::integrations::custom_logger::{
        CallbackTiming, CallbackValue, LogFuture, ModelCallDetails,
    };
    use serde_json::{Map, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    struct RecordingLogger {
        successes: AtomicUsize,
    }

    impl CustomLogger for RecordingLogger {
        fn async_log_success_event<'a>(
            &'a self,
            _: &'a ModelCallDetails,
            _: &'a CallbackValue,
            _: CallbackTiming,
        ) -> LogFuture<'a> {
            Box::pin(async move {
                self.successes.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn gateway_composition_runs_the_core_chat_route() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let api_base = format!("http://{}/v1/messages", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 2048];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..read]);
                if read == 0 || request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let body = r#"{"model":"claude-sonnet-4-5","content":[{"type":"text","text":"gateway"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let logger = Arc::new(RecordingLogger {
            successes: AtomicUsize::new(0),
        });
        let client = LiteLlm::from_services(GatewayMessagesServices::new(|_| None));
        let request = ChatCompletionsRequest {
            model: "anthropic/claude-sonnet-4-5",
            messages: json!([{"role": "user", "content": "hello"}]),
            optional_params: Map::new(),
            api_key: Some("test-key"),
            api_base: Some(&api_base),
            custom_llm_provider: None,
            extra_headers: None,
            timeout: None,
        };
        let response = client
            .chat_completions_with(
                request,
                CallLifecycleContext::new(
                    "chat_completion",
                    "claude-sonnet-4-5",
                    "anthropic",
                    "gateway-call",
                ),
                Arc::new(vec![logger.clone() as Arc<dyn CustomLogger>]),
            )
            .await
            .unwrap();

        server.await.unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("gateway")
        );
        assert_eq!(logger.successes.load(Ordering::Relaxed), 1);
    }
}
