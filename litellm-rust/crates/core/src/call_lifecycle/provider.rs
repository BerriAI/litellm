use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::cache::{ResponseCachePlan, lookup, store};
use super::execution::HostExchange;
use super::host::{HostFailure, LifecycleBackend, LifecycleBackendFuture, drive};
use super::workflow::{LifecycleCall, LifecycleOperation, Workflow, WorkflowFuture, WorkflowReply};
use crate::Error;
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};

#[derive(Clone, Default)]
pub struct ProviderOptions {
    pub model: String,
    pub litellm_call_id: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

impl ProviderOptions {
    pub fn lifecycle_context(&self, call_type: &str) -> CallLifecycleContext {
        CallLifecycleContext::new(
            call_type,
            self.model.clone(),
            self.custom_llm_provider.clone().unwrap_or_default(),
            self.litellm_call_id
                .clone()
                .unwrap_or_else(|| format!("native-{:032x}", rand::random::<u128>())),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub model: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub status: u16,
    pub body: String,
}

pub type ProviderHookFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait ProviderHooks: Send + Sync {
    fn before_request(&self, request: ProviderRequest) -> ProviderHookFuture<'_, ProviderRequest>;
    fn after_response(
        &self,
        response: ProviderResponse,
    ) -> ProviderHookFuture<'_, ProviderResponse>;
}

pub struct NoopProviderHooks;

impl ProviderHooks for NoopProviderHooks {
    fn before_request(&self, request: ProviderRequest) -> ProviderHookFuture<'_, ProviderRequest> {
        Box::pin(async move { Ok(request) })
    }

    fn after_response(
        &self,
        response: ProviderResponse,
    ) -> ProviderHookFuture<'_, ProviderResponse> {
        Box::pin(async move { Ok(response) })
    }
}

pub struct ProviderHookChain {
    hooks: Box<[Arc<dyn ProviderHooks>]>,
}

impl ProviderHookChain {
    pub fn new(hooks: impl IntoIterator<Item = Arc<dyn ProviderHooks>>) -> Self {
        Self {
            hooks: hooks.into_iter().collect(),
        }
    }

    fn before<'a>(
        hooks: &'a [Arc<dyn ProviderHooks>],
        request: ProviderRequest,
    ) -> ProviderHookFuture<'a, ProviderRequest> {
        Box::pin(async move {
            let Some((hook, remaining)) = hooks.split_first() else {
                return Ok(request);
            };
            let request = hook.before_request(request).await?;
            Self::before(remaining, request).await
        })
    }

    fn after<'a>(
        hooks: &'a [Arc<dyn ProviderHooks>],
        response: ProviderResponse,
    ) -> ProviderHookFuture<'a, ProviderResponse> {
        Box::pin(async move {
            let Some((hook, remaining)) = hooks.split_first() else {
                return Ok(response);
            };
            let response = hook.after_response(response).await?;
            Self::after(remaining, response).await
        })
    }
}

impl ProviderHooks for ProviderHookChain {
    fn before_request(&self, request: ProviderRequest) -> ProviderHookFuture<'_, ProviderRequest> {
        Self::before(&self.hooks, request)
    }

    fn after_response(
        &self,
        response: ProviderResponse,
    ) -> ProviderHookFuture<'_, ProviderResponse> {
        Self::after(&self.hooks, response)
    }
}

#[derive(Clone, Debug)]
pub enum CompletedOperation<T> {
    Lifecycle(LifecycleOperation<T>),
    BeforeRequest(ProviderRequest),
    AfterResponse(ProviderResponse),
}

pub enum CompletedReply<Q> {
    Prepared(Result<ResponseCachePlan, Error>),
    CacheStore(Result<Option<Value>, Error>),
    Request(Result<Q, Error>),
    Lifecycle(Result<(), HostFailure>),
    BeforeRequest(Result<ProviderRequest, Error>),
    AfterResponse(Result<ProviderResponse, Error>),
}

pub trait CompletedRoute: Send + Sync + 'static {
    type Request: Send + Sync + 'static;
    type Response: Clone + Send + Sync + serde::de::DeserializeOwned + 'static;

    fn run(request: Self::Request, hooks: Arc<dyn ProviderHooks>)
    -> WorkflowFuture<Self::Response>;
    fn context(request: &Self::Request) -> CallLifecycleContext;

    fn operation(asynchronous: bool) -> CompletedCall<Self>
    where
        Self: Sized,
    {
        CompletedCall::new(CompletedWorkflow::default(), asynchronous)
    }
}

pub struct CompletedWorkflow<R: CompletedRoute> {
    route: PhantomData<R>,
    hooks: Arc<dyn ProviderHooks>,
    cache: ResponseCachePlan,
    cached: Arc<Mutex<Option<Value>>>,
    pending_write: Option<Value>,
    terminal: Arc<Mutex<Option<(CallLifecycleContext, CallLifecycleTiming)>>>,
}

impl<R: CompletedRoute> Default for CompletedWorkflow<R> {
    fn default() -> Self {
        Self {
            route: PhantomData,
            hooks: Arc::new(NoopProviderHooks),
            cache: ResponseCachePlan::default(),
            cached: Arc::default(),
            pending_write: None,
            terminal: Arc::default(),
        }
    }
}

impl<R: CompletedRoute> CompletedWorkflow<R> {
    pub fn with_hooks(hooks: Arc<dyn ProviderHooks>) -> Self {
        Self {
            hooks,
            ..Self::default()
        }
    }
}

impl<R: CompletedRoute> Workflow for CompletedWorkflow<R> {
    type Request = R::Request;
    type Operation = CompletedOperation<R::Response>;
    type Reply = CompletedReply<R::Request>;
    type Response = R::Response;

    fn operation(operation: LifecycleOperation<Self::Response>) -> Self::Operation {
        CompletedOperation::Lifecycle(operation)
    }

    fn accepts(operation: &Self::Operation, reply: &Self::Reply) -> bool {
        match (operation, reply) {
            (_, CompletedReply::Lifecycle(Err(_))) => true,
            (
                CompletedOperation::Lifecycle(LifecycleOperation::Phase(
                    super::host::HostPhase::Prepare,
                )),
                CompletedReply::Prepared(_),
            ) => true,
            (
                CompletedOperation::Lifecycle(LifecycleOperation::Phase(
                    super::host::HostPhase::CacheStore,
                )),
                CompletedReply::CacheStore(_),
            ) => true,
            (
                CompletedOperation::Lifecycle(LifecycleOperation::ProjectRequest),
                CompletedReply::Request(_),
            ) => true,
            (CompletedOperation::Lifecycle(LifecycleOperation::ProjectRequest), _) => false,
            (CompletedOperation::Lifecycle(_), CompletedReply::Lifecycle(_)) => true,
            (CompletedOperation::BeforeRequest(_), CompletedReply::BeforeRequest(_)) => true,
            (CompletedOperation::AfterResponse(_), CompletedReply::AfterResponse(_)) => true,
            _ => false,
        }
    }

    fn reply(&mut self, reply: Self::Reply) -> WorkflowReply<Self::Request, Self::Reply> {
        match reply {
            CompletedReply::Prepared(result) => WorkflowReply::Lifecycle(
                result
                    .map(|cache| self.cache = cache)
                    .map_err(HostFailure::Error),
            ),
            CompletedReply::CacheStore(result) => WorkflowReply::Lifecycle(
                result
                    .map(|value| self.pending_write = value)
                    .map_err(HostFailure::Error),
            ),
            CompletedReply::Request(request) => WorkflowReply::Request(request),
            CompletedReply::Lifecycle(result) => WorkflowReply::Lifecycle(result),
            reply => WorkflowReply::Operation(reply),
        }
    }

    fn start(
        &mut self,
        request: Self::Request,
        host: HostExchange<Self::Operation, Self::Reply>,
    ) -> WorkflowFuture<Self::Response> {
        let context = R::context(&request);
        let started = epoch_seconds();
        let terminal = self.terminal.clone();
        let host_hooks: Arc<dyn ProviderHooks> = Arc::new(ExchangeHooks { host });
        let future = R::run(
            request,
            Arc::new(ProviderHookChain::new([self.hooks.clone(), host_hooks])),
        );
        Box::pin(async move {
            let result = future.await;
            let timing = CallLifecycleTiming::new(started, epoch_seconds());
            *terminal.lock().unwrap_or_else(|error| error.into_inner()) = Some((context, timing));
            result
        })
    }

    fn terminal(&self) -> Option<(CallLifecycleContext, CallLifecycleTiming)> {
        self.terminal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn cache_lookup(&mut self) -> WorkflowFuture<Option<Self::Response>> {
        let cached = self.cached.clone();
        let cache = self.cache.clone();
        Box::pin(async move {
            let Some(value) = lookup(&cache).await else {
                return Ok(None);
            };
            let Ok(response) = serde_json::from_value(value.clone()) else {
                return Ok(None);
            };
            *cached.lock().unwrap_or_else(|error| error.into_inner()) = Some(value);
            Ok(Some(response))
        })
    }

    fn cached_public_response(&self) -> Option<Value> {
        self.cached
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn flush_cache(&mut self) -> WorkflowFuture<()> {
        let value = self.pending_write.take();
        let cache = self.cache.clone();
        Box::pin(async move {
            store(&cache, value).await;
            Ok(())
        })
    }
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

struct ExchangeHooks<Q, T> {
    host: HostExchange<CompletedOperation<T>, CompletedReply<Q>>,
}

impl<Q: Send + 'static, T: Send + Sync + 'static> ProviderHooks for ExchangeHooks<Q, T> {
    fn before_request(&self, request: ProviderRequest) -> ProviderHookFuture<'_, ProviderRequest> {
        Box::pin(async move {
            match self
                .host
                .invoke(CompletedOperation::BeforeRequest(request))
                .await?
            {
                CompletedReply::BeforeRequest(result) => result,
                _ => Err(Error::InvalidRequest(
                    "unexpected provider request reply".into(),
                )),
            }
        })
    }

    fn after_response(
        &self,
        response: ProviderResponse,
    ) -> ProviderHookFuture<'_, ProviderResponse> {
        Box::pin(async move {
            match self
                .host
                .invoke(CompletedOperation::AfterResponse(response))
                .await?
            {
                CompletedReply::AfterResponse(result) => result,
                _ => Err(Error::InvalidRequest(
                    "unexpected provider response reply".into(),
                )),
            }
        })
    }
}

pub type CompletedCall<R> = LifecycleCall<CompletedWorkflow<R>>;

pub async fn run_completed<R: CompletedRoute>(request: R::Request) -> Result<R::Response, Error> {
    let mut call = R::operation(false);
    drive(
        &mut call,
        &CompletedBackend::<R> {
            request: Mutex::new(Some(request)),
            route: PhantomData,
        },
    )
    .await
}

pub async fn run_completed_with_hooks<R: CompletedRoute>(
    request: R::Request,
    hooks: Arc<dyn ProviderHooks>,
) -> Result<R::Response, Error> {
    let mut call = CompletedCall::<R>::new(CompletedWorkflow::with_hooks(hooks), false);
    drive(
        &mut call,
        &CompletedBackend::<R> {
            request: Mutex::new(Some(request)),
            route: PhantomData,
        },
    )
    .await
}

struct CompletedBackend<R: CompletedRoute> {
    request: Mutex<Option<R::Request>>,
    route: PhantomData<R>,
}

impl<R: CompletedRoute>
    LifecycleBackend<CompletedOperation<R::Response>, CompletedReply<R::Request>>
    for CompletedBackend<R>
{
    fn invoke(
        &self,
        operation: CompletedOperation<R::Response>,
    ) -> LifecycleBackendFuture<'_, CompletedReply<R::Request>> {
        Box::pin(async move {
            match operation {
                CompletedOperation::Lifecycle(LifecycleOperation::ProjectRequest) => {
                    CompletedReply::Request(
                        self.request
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .take()
                            .ok_or_else(|| {
                                Error::InvalidRequest("request was already projected".into())
                            }),
                    )
                }
                CompletedOperation::Lifecycle(_) => CompletedReply::Lifecycle(Ok(())),
                CompletedOperation::BeforeRequest(request) => {
                    CompletedReply::BeforeRequest(Ok(request))
                }
                CompletedOperation::AfterResponse(response) => {
                    CompletedReply::AfterResponse(Ok(response))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingHook {
        name: &'static str,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl ProviderHooks for RecordingHook {
        fn before_request(
            &self,
            request: ProviderRequest,
        ) -> ProviderHookFuture<'_, ProviderRequest> {
            Box::pin(async move {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("{}:before", self.name));
                Ok(ProviderRequest {
                    body: Value::String(format!(
                        "{}:{}",
                        request.body.as_str().unwrap(),
                        self.name
                    )),
                    ..request
                })
            })
        }

        fn after_response(
            &self,
            response: ProviderResponse,
        ) -> ProviderHookFuture<'_, ProviderResponse> {
            Box::pin(async move {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("{}:after", self.name));
                Ok(ProviderResponse {
                    body: format!("{}:{}", response.body, self.name),
                    ..response
                })
            })
        }
    }

    #[derive(Clone, Deserialize)]
    struct TestResponse(String);

    struct TestRoute;

    impl CompletedRoute for TestRoute {
        type Request = ();
        type Response = TestResponse;

        fn run((): Self::Request, hooks: Arc<dyn ProviderHooks>) -> WorkflowFuture<Self::Response> {
            Box::pin(async move {
                let request = hooks
                    .before_request(ProviderRequest {
                        model: "model".into(),
                        url: "https://example.com".into(),
                        headers: Vec::new(),
                        body: Value::String("request".into()),
                    })
                    .await?;
                let response = hooks
                    .after_response(ProviderResponse {
                        status: 200,
                        body: request.body.as_str().unwrap().to_owned(),
                    })
                    .await?;
                Ok(TestResponse(response.body))
            })
        }

        fn context(_: &Self::Request) -> CallLifecycleContext {
            CallLifecycleContext::new("test", "model", "provider", "call")
        }
    }

    #[tokio::test]
    async fn native_hook_chain_runs_in_order_through_the_lifecycle() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let hooks = ProviderHookChain::new([
            Arc::new(RecordingHook {
                name: "first",
                events: events.clone(),
            }) as Arc<dyn ProviderHooks>,
            Arc::new(RecordingHook {
                name: "second",
                events: events.clone(),
            }) as Arc<dyn ProviderHooks>,
        ]);
        let response = run_completed_with_hooks::<TestRoute>((), Arc::new(hooks))
            .await
            .unwrap();
        assert_eq!(response.0, "request:first:second:first:second");
        assert_eq!(
            *events.lock().unwrap(),
            [
                "first:before",
                "second:before",
                "first:after",
                "second:after"
            ]
        );
    }
}
