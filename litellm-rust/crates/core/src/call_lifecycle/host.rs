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
pub enum HostFailure {
    Error(crate::Error),
    Cancelled(crate::Error),
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
