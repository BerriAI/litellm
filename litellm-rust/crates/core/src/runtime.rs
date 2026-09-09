use crate::messages::types::ProviderMessagesRequest;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::Error;
use crate::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use crate::integrations::custom_logger::{LogError, LogFuture};
use crate::lifecycle::{
    ActionResult, BytesStream, CallLifecycleContext, Clock, DeploymentFailureHooks,
    DeploymentPreHooks, DeploymentSuccessHooks, ModerationHooks, PreCallHooks, StreamingCall,
    TerminalDispatcher, TerminalRecord,
};
use crate::messages::lifecycle::{MessagesServices, Options as MessagesOptions};
use crate::messages::types::{AnthropicMessagesResponse, MessagesRequest};

pub struct HttpRequest {
    pub method: reqwest::Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub timeout: Option<Duration>,
}

pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

pub struct HttpStreamResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub cache_control: Option<String>,
    pub stream: BytesStream,
}

pub type HttpFuture<'a> = Pin<Box<dyn Future<Output = Result<HttpResponse, Error>> + Send + 'a>>;
pub type HttpStreamFuture<'a> =
    Pin<Box<dyn Future<Output = Result<HttpStreamResponse, Error>> + Send + 'a>>;

pub trait HttpTransport: Send + Sync {
    fn execute(&self, request: HttpRequest) -> HttpFuture<'_>;
    fn execute_stream(&self, request: HttpRequest) -> HttpStreamFuture<'_>;
}

#[derive(Clone)]
pub struct NativeHttpTransport {
    standard: reqwest::Client,
}

impl NativeHttpTransport {
    pub fn new() -> Self {
        Self {
            standard: reqwest::Client::builder()
                .timeout(Duration::from_secs(crate::constants::HTTP_TIMEOUT_SECS))
                .connect_timeout(Duration::from_secs(
                    crate::constants::HTTP_CONNECT_TIMEOUT_SECS,
                ))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for NativeHttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpTransport for NativeHttpTransport {
    fn execute(&self, request: HttpRequest) -> HttpFuture<'_> {
        let client = self.standard.clone();
        Box::pin(async move {
            let builder = request.headers.into_iter().fold(
                client
                    .request(request.method, request.url)
                    .body(request.body),
                |builder, (name, value)| builder.header(name, value),
            );
            let builder = match request.timeout {
                Some(timeout) => builder.timeout(timeout),
                None => builder,
            };
            let response = builder.send().await.map_err(|error| {
                if error.is_connect() || error.is_builder() {
                    Error::Connect(error.to_string())
                } else {
                    Error::Network(error.to_string())
                }
            })?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map_err(|error| Error::Network(error.to_string()))?
                .to_vec();
            Ok(HttpResponse { status, body })
        })
    }

    fn execute_stream(&self, request: HttpRequest) -> HttpStreamFuture<'_> {
        let client = self.standard.clone();
        Box::pin(async move {
            let builder = request.headers.into_iter().fold(
                client
                    .request(request.method, request.url)
                    .body(request.body),
                |builder, (name, value)| builder.header(name, value),
            );
            let builder = match request.timeout {
                Some(timeout) => builder.timeout(timeout),
                None => builder,
            };
            let response = builder.send().await.map_err(|error| {
                if error.is_connect() || error.is_builder() {
                    Error::Connect(error.to_string())
                } else {
                    Error::Network(error.to_string())
                }
            })?;
            let status = response.status().as_u16();
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let cache_control = response
                .headers()
                .get(reqwest::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let stream = futures_util::StreamExt::map(response.bytes_stream(), |result| {
                result.map_err(|error| Error::Network(error.to_string()))
            });
            Ok(HttpStreamResponse {
                status,
                content_type,
                cache_control,
                stream: Box::pin(stream),
            })
        })
    }
}

pub type SessionFuture<'a, Session> = Pin<Box<dyn Future<Output = Result<Session, Error>> + 'a>>;

pub trait CallServices: Send + Sync {
    type Bindings;
    type Session;
    type OpenFuture<'a>: Future<Output = Result<Self::Session, Error>> + 'a
    where
        Self: 'a;

    fn open<'a>(
        &'a self,
        context: CallLifecycleContext,
        bindings: Self::Bindings,
    ) -> Self::OpenFuture<'a>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBindings;

#[derive(Debug)]
pub struct NativeSession;

impl crate::lifecycle::StreamDrain for NativeSession {}

impl Clock for NativeSession {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }
}

impl PreCallHooks<MessagesRequest> for NativeSession {
    type PreCallFuture<'a> = std::future::Ready<ActionResult<MessagesRequest, Error>>;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl ModerationHooks<ProviderMessagesRequest> for NativeSession {
    type ModerationFuture<'a> = std::future::Ready<ActionResult<ProviderMessagesRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: ProviderMessagesRequest,
    ) -> Self::ModerationFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl DeploymentPreHooks<MessagesRequest> for NativeSession {}
impl DeploymentSuccessHooks<AnthropicMessagesResponse> for NativeSession {}
impl DeploymentFailureHooks for NativeSession {}

impl<'request> PreCallHooks<ChatCompletionsRequest<'request>> for NativeSession {
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

impl<'request>
    ModerationHooks<crate::chat_completions::types::ResolvedChatCompletionsRequest<'request>>
    for NativeSession
{
    type ModerationFuture<'a>
        = std::future::Ready<
        ActionResult<
            crate::chat_completions::types::ResolvedChatCompletionsRequest<'request>,
            Error,
        >,
    >
    where
        Self: 'a;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: crate::chat_completions::types::ResolvedChatCompletionsRequest<'request>,
    ) -> Self::ModerationFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl<'request> DeploymentPreHooks<ChatCompletionsRequest<'request>> for NativeSession {}

impl DeploymentSuccessHooks<ChatCompletionsResponse> for NativeSession {}

impl TerminalDispatcher for NativeSession {
    fn dispatch<'a>(&'a self, _: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async { Ok::<(), LogError>(()) })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCallServices;

impl CallServices for NativeCallServices {
    type Bindings = NativeBindings;
    type Session = NativeSession;
    type OpenFuture<'a> = std::future::Ready<Result<NativeSession, Error>>;

    fn open<'a>(
        &'a self,
        _context: CallLifecycleContext,
        _bindings: Self::Bindings,
    ) -> Self::OpenFuture<'a> {
        std::future::ready(Ok(NativeSession))
    }
}

pub trait ChatCompletionsServices:
    crate::providers::auth::ChatAuthorizationServices + Send + Sync
{
    type Transport: HttpTransport;
    type Calls: CallServices;

    fn transport(&self) -> &Self::Transport;
    fn calls(&self) -> &Self::Calls;
}

pub trait MessagesRuntimeServices: crate::providers::auth::Environment + Send + Sync {
    type Transport: HttpTransport;
    type Calls: CallServices;

    fn transport(&self) -> &Self::Transport;
    fn calls(&self) -> &Self::Calls;
}

pub struct NativeServices {
    transport: NativeHttpTransport,
    calls: NativeCallServices,
    #[cfg(feature = "bedrock-auth")]
    aws_credentials: crate::providers::auth::AwsCredentialState,
}

impl NativeServices {
    pub fn new() -> Self {
        Self {
            transport: NativeHttpTransport::new(),
            calls: NativeCallServices,
            #[cfg(feature = "bedrock-auth")]
            aws_credentials: crate::providers::auth::native_aws_credential_state(),
        }
    }
}

impl Default for NativeServices {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::providers::auth::Environment for NativeServices {
    fn environment(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

impl crate::providers::auth::SigningClock for NativeServices {
    fn signing_time(&self) -> std::time::SystemTime {
        std::time::SystemTime::now()
    }
}

impl crate::providers::auth::AwsMechanisms for NativeServices {
    #[cfg(feature = "bedrock-auth")]
    fn aws_credential_state(&self) -> &crate::providers::auth::AwsCredentialState {
        &self.aws_credentials
    }
}

impl ChatCompletionsServices for NativeServices {
    type Transport = NativeHttpTransport;
    type Calls = NativeCallServices;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

impl MessagesRuntimeServices for NativeServices {
    type Transport = NativeHttpTransport;
    type Calls = NativeCallServices;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

pub struct LiteLlm<S = NativeServices> {
    services: Arc<S>,
}

impl<S> Clone for LiteLlm<S> {
    fn clone(&self) -> Self {
        Self {
            services: self.services.clone(),
        }
    }
}

impl<S> LiteLlm<S> {
    pub fn from_services(services: S) -> Self {
        Self {
            services: Arc::new(services),
        }
    }

    pub fn services(&self) -> &S {
        &self.services
    }
}

impl LiteLlm<NativeServices> {
    pub fn new() -> Self {
        Self::from_services(NativeServices::new())
    }

    pub async fn chat_completions(
        &self,
        request: ChatCompletionsRequest<'_>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let context = CallLifecycleContext::new(
            "chat_completion",
            request.model,
            request.custom_llm_provider.unwrap_or_default(),
            "",
        );
        self.chat_completions_with(request, context, NativeBindings)
            .await
    }

    pub async fn messages_stream(&self, request: MessagesRequest) -> Result<StreamingCall, Error> {
        let provider = request
            .custom_llm_provider
            .as_deref()
            .or_else(|| request.model.split_once('/').map(|(provider, _)| provider))
            .unwrap_or(crate::constants::ANTHROPIC_MESSAGES_PROVIDER);
        let context = CallLifecycleContext::new(
            "messages",
            &request.model,
            provider,
            format!("{:032x}", rand::random::<u128>()),
        );
        self.messages_stream_with(
            request,
            MessagesOptions {
                asynchronous: true,
                ..MessagesOptions::default()
            },
            context,
            NativeBindings,
        )
        .await
    }

    pub async fn messages(
        &self,
        request: MessagesRequest,
    ) -> Result<AnthropicMessagesResponse, Error> {
        let provider = request
            .custom_llm_provider
            .as_deref()
            .or_else(|| request.model.split_once('/').map(|(provider, _)| provider))
            .unwrap_or(crate::constants::ANTHROPIC_MESSAGES_PROVIDER);
        let context = CallLifecycleContext::new(
            "messages",
            &request.model,
            provider,
            format!("{:032x}", rand::random::<u128>()),
        );
        self.messages_with(request, context, NativeBindings).await
    }
}

impl Default for LiteLlm<NativeServices> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> LiteLlm<S>
where
    S: ChatCompletionsServices,
    <<S as ChatCompletionsServices>::Calls as CallServices>::Session:
        crate::chat_completions::lifecycle::ChatCompletionsSession,
{
    pub async fn chat_completions_with(
        &self,
        request: ChatCompletionsRequest<'_>,
        context: CallLifecycleContext,
        bindings: <<S as ChatCompletionsServices>::Calls as CallServices>::Bindings,
    ) -> Result<ChatCompletionsResponse, Error> {
        crate::chat_completions::request::resolve_request(request.clone())?;
        let session = self
            .services
            .calls()
            .open(context.clone(), bindings)
            .await?;
        crate::chat_completions::lifecycle::execute(
            &*self.services,
            self.services.transport(),
            &session,
            request,
            context,
        )
        .await
        .into_result()
    }
}

impl<S> LiteLlm<S>
where
    S: MessagesRuntimeServices,
    <<S as MessagesRuntimeServices>::Calls as CallServices>::Session: MessagesServices,
{
    pub async fn messages_with(
        &self,
        request: MessagesRequest,
        context: CallLifecycleContext,
        bindings: <<S as MessagesRuntimeServices>::Calls as CallServices>::Bindings,
    ) -> Result<AnthropicMessagesResponse, Error> {
        let invocation = self
            .services
            .calls()
            .open(context.clone(), bindings)
            .await?;
        crate::messages::lifecycle::messages_with_provider(
            &invocation,
            request,
            MessagesOptions {
                asynchronous: true,
                ..MessagesOptions::default()
            },
            context,
            |request| {
                std::future::ready(
                    crate::messages::request::build_provider_request_with_environment(
                        request,
                        &|key| self.services.environment(key),
                    ),
                )
            },
            |request| async move {
                crate::messages::execute_provider_messages_request_with_transport(
                    self.services.transport(),
                    request,
                )
                .await
            },
        )
        .await
        .into_result()
    }
}

struct StreamingRuntimeSession<S, Session> {
    _application: Arc<S>,
    invocation: Session,
}

impl<S, Session> Clock for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: Clock,
{
    fn now(&self) -> f64 {
        self.invocation.now()
    }
}

impl<S, Session> PreCallHooks<MessagesRequest> for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: PreCallHooks<MessagesRequest>,
{
    type PreCallFuture<'a>
        = Session::PreCallFuture<'a>
    where
        Self: 'a;
    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        self.invocation.async_pre_call_hook(context, request)
    }
}

impl<S, Session> ModerationHooks<ProviderMessagesRequest> for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: ModerationHooks<ProviderMessagesRequest>,
{
    type ModerationFuture<'a>
        = Session::ModerationFuture<'a>
    where
        Self: 'a;

    fn async_moderation_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: ProviderMessagesRequest,
    ) -> Self::ModerationFuture<'a> {
        self.invocation.async_moderation_hook(context, request)
    }
}

impl<S, Session> DeploymentPreHooks<MessagesRequest> for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: DeploymentPreHooks<MessagesRequest>,
{
    fn async_pre_call_deployment_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> crate::lifecycle::execution::CallbackFuture<'a, ActionResult<MessagesRequest, Error>>
    where
        MessagesRequest: 'a,
    {
        self.invocation
            .async_pre_call_deployment_hook(context, request)
    }
}

impl<S, Session> DeploymentSuccessHooks<AnthropicMessagesResponse>
    for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: DeploymentSuccessHooks<AnthropicMessagesResponse>,
{
    fn async_post_call_success_deployment_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: AnthropicMessagesResponse,
    ) -> crate::lifecycle::execution::CallbackFuture<
        'a,
        ActionResult<AnthropicMessagesResponse, Error>,
    >
    where
        AnthropicMessagesResponse: 'a,
    {
        self.invocation
            .async_post_call_success_deployment_hook(context, response)
    }
}

impl<S, Session> DeploymentFailureHooks for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: DeploymentFailureHooks,
{
    fn async_post_call_failure_deployment_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a Error,
    ) -> crate::lifecycle::execution::CallbackFuture<'a, Result<(), Error>> {
        self.invocation
            .async_post_call_failure_deployment_hook(context, error)
    }
}

impl<S, Session> TerminalDispatcher for StreamingRuntimeSession<S, Session>
where
    S: Send + Sync,
    Session: TerminalDispatcher,
{
    fn observations(&self) -> crate::lifecycle::program::Observations {
        self.invocation.observations()
    }

    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        self.invocation.dispatch(terminal)
    }
}

impl<S: Send + Sync, Session: crate::lifecycle::StreamDrain> crate::lifecycle::StreamDrain
    for StreamingRuntimeSession<S, Session>
{
    fn stream_drain_policy(&self) -> crate::lifecycle::StreamDrainPolicy {
        self.invocation.stream_drain_policy()
    }
}

impl<S> LiteLlm<S>
where
    S: MessagesRuntimeServices + 'static,
    <<S as MessagesRuntimeServices>::Calls as CallServices>::Session:
        MessagesServices + crate::lifecycle::StreamDrain + Send + Sync + 'static,
    for<'a> <<S as MessagesRuntimeServices>::Calls as CallServices>::OpenFuture<'a>: Send,
{
    pub async fn messages_stream_with(
        &self,
        request: MessagesRequest,
        options: MessagesOptions,
        context: CallLifecycleContext,
        bindings: <<S as MessagesRuntimeServices>::Calls as CallServices>::Bindings,
    ) -> Result<StreamingCall, Error> {
        let context = options.context(context);
        let invocation = self
            .services
            .calls()
            .open(context.clone(), bindings)
            .await?;
        let application = self.services.clone();
        let environment = application.clone();
        let session = Arc::new(StreamingRuntimeSession {
            _application: application.clone(),
            invocation,
        });
        crate::messages::lifecycle::messages_stream_with(
            session,
            request,
            options,
            context,
            move |request| {
                std::future::ready(
                    crate::messages::request::build_provider_request_with_environment(
                        request,
                        &|key| environment.environment(key),
                    ),
                )
            },
            move |request| async move {
                crate::messages::execute_provider_messages_stream_with_transport(
                    application.transport(),
                    request,
                )
                .await
            },
        )
        .await
    }
}
