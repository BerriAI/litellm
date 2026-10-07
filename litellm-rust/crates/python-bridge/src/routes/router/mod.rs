//! The Rust router backend's native half: a `Router` holding the routing engine, and one
//! routed call per `route`, driven by `host-python`'s coroutine driver. Every attempt is an
//! `Invoke` host op answered by the per-call Python driver in `litellm/router_backends`.

mod host;
mod python;

use std::sync::Arc;

use litellm_host::{
    call::{CallOutput, hosted_call},
    machine::MachineFault,
    protocol::{Protocol, Reply},
};
use litellm_host_python::{HookChain, enter_native, from_py_argument};
use litellm_router::{
    engine::{Engine, Failed, Resume, RouteError, Routed, RouterCall},
    host::{Attempt, Invoked},
    random::PythonRandom,
    settings::Settings,
    snapshot::{RoutedDeployment, Snapshot},
    store::{RedisTarget, Store, SystemClock},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use host::{BridgeHost, RouterBinding};

/// A Python object the router carries without looking inside.
pub(crate) type PyObj = Arc<Py<PyAny>>;

#[derive(Clone)]
pub(crate) enum BridgeError {
    Failed(Arc<Failed<PyObj>>),
    Machine(MachineFault),
    Request(String),
}

impl From<MachineFault> for BridgeError {
    fn from(fault: MachineFault) -> Self {
        Self::Machine(fault)
    }
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(_) => formatter.write_str("routed call failed"),
            Self::Machine(fault) => write!(formatter, "router machine fault: {fault:?}"),
            Self::Request(message) => formatter.write_str(message),
        }
    }
}

pub(crate) enum RouterHostCall {
    Invoke {
        attempt: Attempt<PyObj>,
        reply: Reply<Invoked<PyObj, PyObj>>,
    },
    Sleep {
        seconds: f64,
        reply: Reply<()>,
    },
}

/// A call to route, or to resume from a stream that failed after its content reached the caller.
pub(crate) struct RouterRequest {
    call: RouterCall,
    resume: Option<Resume<PyObj>>,
}

pub(crate) struct RouterProtocol;

impl Protocol for RouterProtocol {
    type Request = RouterRequest;
    type Response = Routed<PyObj, PyObj>;
    type Error = BridgeError;
    type HostCall = RouterHostCall;
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

#[pyclass(frozen, name = "Router")]
pub(crate) struct NativeRouter {
    engine: Arc<Engine>,
}

#[pymethods]
impl NativeRouter {
    /// `deployments` are normalized deployments projected to the fields routing reads,
    /// `settings` the router attributes `PythonRouter.__init__` resolved. `seed` makes the
    /// shuffle and jitter draws those of `random.seed(seed)`.
    #[new]
    #[pyo3(signature = (deployments, settings, providers, redis_url = None, seed = None))]
    fn new(
        #[pyo3(from_py_with = from_py_argument)] deployments: Vec<RoutedDeployment>,
        #[pyo3(from_py_with = from_py_argument)] settings: Settings,
        providers: Vec<String>,
        redis_url: Option<String>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        enter_native()?;
        let redis = redis_url
            .as_deref()
            .map(RedisTarget::open)
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let read_interval = settings.tunables.cooldown_redis_read_interval;
        let random = seed.map_or_else(PythonRandom::from_entropy, PythonRandom::seeded);
        Ok(Self {
            engine: Arc::new(Engine::new(
                Snapshot::new(deployments, settings, providers),
                Store::new(Arc::new(SystemClock), redis, read_interval),
                random,
            )),
        })
    }

    /// Swaps in the deployments and settings after a runtime change. Calls already running
    /// keep the snapshot they started with.
    fn replace(
        &self,
        #[pyo3(from_py_with = from_py_argument)] deployments: Vec<RoutedDeployment>,
        #[pyo3(from_py_with = from_py_argument)] settings: Settings,
    ) {
        let registry = self.engine.registry();
        let providers = registry.load().providers.clone();
        registry.replace(Snapshot::new(deployments, settings, providers));
    }

    /// One routed call. `call` holds the routing-relevant request fields; `driver` answers
    /// the call's host ops and builds its result. Returns a coroutine when `asynchronous`.
    fn route(
        &self,
        py: Python<'_>,
        call: Bound<'_, PyDict>,
        driver: Bound<'_, PyAny>,
        asynchronous: bool,
    ) -> PyResult<Py<PyAny>> {
        let engine = Arc::clone(&self.engine);
        litellm_host_python::run_call(
            py,
            move |_, _, request: RouterRequest| {
                Ok(hosted_call::<RouterProtocol, _, _>(
                    request,
                    None,
                    move |request, services, _, _| async move {
                        let host = BridgeHost(services);
                        let routed = match request.resume {
                            None => engine.route(&host, request.call).await,
                            Some(resume) => engine.resume(&host, request.call, resume).await,
                        };
                        match routed {
                            Ok(routed) => Ok(CallOutput::Complete(routed)),
                            Err(RouteError::Failed(failed)) => {
                                Err(BridgeError::Failed(Arc::from(failed)))
                            }
                            Err(RouteError::Host(error)) => Err(error),
                        }
                    },
                ))
            },
            RouterBinding::new(driver.unbind(), asynchronous),
            HookChain::new(),
            call.unbind(),
            crate::lifecycle::call_options(asynchronous),
        )
    }
}
