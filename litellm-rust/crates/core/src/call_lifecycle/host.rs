use std::future::Future;
use std::pin::Pin;

pub enum HostCallStep<O, C> {
    Host(O),
    Complete(C),
}

pub type HostCallFuture<'a, O, C> =
    Pin<Box<dyn Future<Output = Result<HostCallStep<O, C>, crate::Error>> + Send + 'a>>;

pub trait HostCall: Send + Sync {
    type Operation: Send + 'static;
    type Result: Send + 'static;
    type Complete: Send + 'static;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete>;

    fn interrupt(
        &mut self,
        failure: HostFailure,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete>;
}

pub type LifecycleBackendFuture<'a, R> = Pin<Box<dyn Future<Output = R> + Send + 'a>>;

pub trait LifecycleBackend<O, R>: Send + Sync {
    fn invoke(&self, operation: O) -> LifecycleBackendFuture<'_, R>;
}

pub async fn drive<C, B>(call: &mut C, backend: &B) -> Result<C::Complete, crate::Error>
where
    C: HostCall,
    B: LifecycleBackend<C::Operation, C::Result> + ?Sized,
{
    let mut result = None;
    loop {
        match call.resume(result.take()).await? {
            HostCallStep::Host(operation) => result = Some(backend.invoke(operation).await),
            HostCallStep::Complete(response) => return Ok(response),
        }
    }
}

pub enum HostStep<V, S> {
    Ready(V),
    Suspend(S),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPhase {
    Setup,
    DeploymentPreCall,
    Prepare,
    CacheLookup,
    Execute,
    ConstructResponse,
    PostProcess,
    DeploymentPostCall,
    CacheStore,
    Finalize,
    Success,
    MapFailure,
    DeploymentFailure,
    Failure,
    AsyncFailure,
    Complete,
}

#[derive(Clone, Debug)]
pub enum HostFailure {
    Error(crate::Error),
    Cancelled(crate::Error),
}

pub struct HostLifecycle {
    phase: HostPhase,
    asynchronous: bool,
    cached: bool,
}

impl HostLifecycle {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            phase: HostPhase::Setup,
            asynchronous,
            cached: false,
        }
    }

    pub fn cache_hit(&mut self) {
        self.cached = true;
        self.phase = HostPhase::ConstructResponse;
    }

    pub fn phase(&self) -> HostPhase {
        self.phase
    }

    pub fn accept(&mut self, result: Result<(), HostFailure>) -> Option<crate::Error> {
        if let Err(failure) = result {
            if self.phase == HostPhase::DeploymentFailure {
                self.phase = HostPhase::Failure;
                return None;
            }
            let error = match failure {
                HostFailure::Cancelled(error) => {
                    self.phase = HostPhase::Complete;
                    return Some(error);
                }
                HostFailure::Error(error) => error,
            };
            match self.phase {
                HostPhase::Failure | HostPhase::AsyncFailure => {
                    self.advance();
                    return None;
                }
                HostPhase::Success => self.phase = HostPhase::Complete,
                HostPhase::Execute | HostPhase::ConstructResponse => {
                    self.phase = HostPhase::MapFailure;
                }
                _ => self.phase = HostPhase::Failure,
            }
            return Some(error);
        }
        self.advance();
        None
    }

    fn advance(&mut self) {
        self.phase = match self.phase {
            HostPhase::Setup if self.asynchronous => HostPhase::DeploymentPreCall,
            HostPhase::Setup | HostPhase::DeploymentPreCall => HostPhase::Prepare,
            HostPhase::Prepare => HostPhase::CacheLookup,
            HostPhase::CacheLookup => HostPhase::Execute,
            HostPhase::Execute => HostPhase::ConstructResponse,
            HostPhase::ConstructResponse => HostPhase::PostProcess,
            HostPhase::PostProcess if self.cached => HostPhase::Finalize,
            HostPhase::PostProcess if self.asynchronous => HostPhase::DeploymentPostCall,
            HostPhase::PostProcess => HostPhase::CacheStore,
            HostPhase::DeploymentPostCall if self.cached => HostPhase::Finalize,
            HostPhase::DeploymentPostCall => HostPhase::CacheStore,
            HostPhase::CacheStore => HostPhase::Finalize,
            HostPhase::Finalize => HostPhase::Success,
            HostPhase::MapFailure if self.asynchronous => HostPhase::DeploymentFailure,
            HostPhase::MapFailure | HostPhase::DeploymentFailure => HostPhase::Failure,
            HostPhase::Failure if self.asynchronous => HostPhase::AsyncFailure,
            HostPhase::Failure
            | HostPhase::AsyncFailure
            | HostPhase::Success
            | HostPhase::Complete => HostPhase::Complete,
        };
    }
}
