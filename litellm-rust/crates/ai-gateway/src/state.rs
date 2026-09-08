use std::sync::Arc;

use crate::io::realtime_pool::RealtimePool;
use litellm_core::Error;
use litellm_core::integrations::custom_logger::{CustomLogger, CustomLoggerRunner, LogFuture};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, Clock, RequestPolicy, TerminalDispatcher, TerminalRecord,
};
use litellm_core::messages::types::MessagesRequest;
use litellm_core::router::Router;
use litellm_core::runtime::{CallServices, LiteLlm, MessagesRuntimeServices, NativeHttpTransport};

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

impl RequestPolicy<MessagesRequest, MessagesRequest> for GatewayMessagesSession {
    type PreCallFuture<'a> = std::future::Ready<ActionResult<MessagesRequest, Error>>;
    type DuringCallFuture<'a> = std::future::Ready<ActionResult<MessagesRequest, Error>>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::DuringCallFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
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
}

impl GatewayMessagesServices {
    pub fn new(environment: impl Fn(&str) -> Option<String> + Send + Sync + 'static) -> Self {
        Self {
            transport: NativeHttpTransport::new(),
            calls: GatewayMessagesCallServices,
            environment: Arc::new(environment),
        }
    }
}

impl litellm_core::providers::auth::Environment for GatewayMessagesServices {
    fn environment(&self, key: &str) -> Option<String> {
        (self.environment)(key)
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
