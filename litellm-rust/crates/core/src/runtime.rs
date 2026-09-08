use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::Error;
use crate::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use crate::lifecycle::CallLifecycleContext;

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

pub type HttpFuture<'a> = Pin<Box<dyn Future<Output = Result<HttpResponse, Error>> + Send + 'a>>;

pub trait HttpTransport: Send + Sync {
    fn execute(&self, request: HttpRequest) -> HttpFuture<'_>;
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
}

pub type SessionFuture<'a, Session> = Pin<Box<dyn Future<Output = Result<Session, Error>> + 'a>>;

pub trait CallServices: Send + Sync {
    type Bindings;
    type Session;

    fn open<'a>(
        &'a self,
        context: CallLifecycleContext,
        bindings: Self::Bindings,
    ) -> SessionFuture<'a, Self::Session>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBindings;

#[derive(Debug)]
pub struct NativeSession;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCallServices;

impl CallServices for NativeCallServices {
    type Bindings = NativeBindings;
    type Session = NativeSession;

    fn open<'a>(
        &'a self,
        _context: CallLifecycleContext,
        _bindings: Self::Bindings,
    ) -> SessionFuture<'a, Self::Session> {
        Box::pin(std::future::ready(Ok(NativeSession)))
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
}

impl Default for LiteLlm<NativeServices> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> LiteLlm<S>
where
    S: ChatCompletionsServices,
{
    pub async fn chat_completions_with(
        &self,
        request: ChatCompletionsRequest<'_>,
        context: CallLifecycleContext,
        bindings: <<S as ChatCompletionsServices>::Calls as CallServices>::Bindings,
    ) -> Result<ChatCompletionsResponse, Error> {
        let request = crate::chat_completions::request::resolve_request(request)?;
        let _session = self.services.calls().open(context, bindings).await?;
        crate::chat_completions::handler::execute_chat_completions_provider_call_with_transport(
            &*self.services,
            self.services.transport(),
            request,
        )
        .await
    }
}
