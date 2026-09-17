use std::future::Future;
use std::pin::Pin;

pub enum NativeCallStep<O, C> {
    Host(O),
    Complete(C),
}

pub type NativeCallFuture<'a, O, C, E> =
    Pin<Box<dyn Future<Output = Result<NativeCallStep<O, C>, E>> + Send + 'a>>;

pub trait NativeCall: Send + Sync {
    type Error: Send + Sync + 'static;
    type Operation: Send + 'static;
    type Result: Send + 'static;
    type Complete: Send + 'static;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> NativeCallFuture<'_, Self::Operation, Self::Complete, Self::Error>;

    fn interrupt(
        &mut self,
        failure: HostFailure<Self::Error>,
    ) -> NativeCallFuture<'_, Self::Operation, Self::Complete, Self::Error>;
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
    Execute,
    ConstructResponse,
    DeploymentPostCall,
    Finalize,
    Success,
    MapFailure,
    DeploymentFailure,
    Failure,
    AsyncFailure,
    Complete,
}

#[derive(Clone, Debug)]
pub enum HostFailure<E> {
    Error(E),
    Cancelled(E),
}

pub struct HostLifecycle {
    phase: HostPhase,
    asynchronous: bool,
}

impl HostLifecycle {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            phase: HostPhase::Setup,
            asynchronous,
        }
    }

    pub fn phase(&self) -> HostPhase {
        self.phase
    }

    pub fn accept<E>(&mut self, result: Result<(), HostFailure<E>>) -> Option<E> {
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
            HostPhase::Prepare => HostPhase::Execute,
            HostPhase::Execute => HostPhase::ConstructResponse,
            HostPhase::ConstructResponse if self.asynchronous => HostPhase::DeploymentPostCall,
            HostPhase::ConstructResponse | HostPhase::DeploymentPostCall => HostPhase::Finalize,
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
