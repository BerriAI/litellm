pub enum HostStep<V, S> {
    Ready(V),
    Suspend(S),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPhase {
    Setup,
    DeploymentPreCall,
    Prepare,
    PrepareTransport,
    PreCall,
    Execute,
    PostCall,
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

pub trait CallHost {
    type Value;
    type Error;
    type Suspension;

    fn invoke(
        &mut self,
        phase: HostPhase,
    ) -> Result<HostStep<Self::Value, Self::Suspension>, Self::Error>;
    fn accept(&mut self, phase: HostPhase, value: Self::Value) -> Result<(), Self::Error>;
    fn retain_failure(&mut self, phase: HostPhase, error: Self::Error);
    fn is_cancellation(error: &Self::Error) -> bool;
    fn finish(&mut self) -> Result<Self::Value, Self::Error>;
    fn cleanup(&mut self);
}

pub struct HostLifecycle<H: CallHost> {
    host: H,
    phase: HostPhase,
    asynchronous: bool,
    cleaned: bool,
}

impl<H: CallHost> HostLifecycle<H> {
    pub fn new(host: H, asynchronous: bool) -> Self {
        Self {
            host,
            phase: HostPhase::Setup,
            asynchronous,
            cleaned: false,
        }
    }

    pub fn resume(
        &mut self,
        result: Option<Result<H::Value, H::Error>>,
    ) -> Result<HostStep<H::Value, H::Suspension>, H::Error> {
        if let Some(result) = result {
            self.accept(result);
        }
        loop {
            if self.phase == HostPhase::Complete {
                let result = self.host.finish();
                self.cleanup();
                return result.map(HostStep::Ready);
            }
            match self.host.invoke(self.phase) {
                Ok(HostStep::Suspend(pending)) => return Ok(HostStep::Suspend(pending)),
                Ok(HostStep::Ready(value)) => self.accept(Ok(value)),
                Err(error) => self.accept(Err(error)),
            }
        }
    }

    fn accept(&mut self, result: Result<H::Value, H::Error>) {
        let result = result.and_then(|value| self.host.accept(self.phase, value));
        if let Err(error) = result {
            if self.phase == HostPhase::DeploymentFailure {
                self.phase = HostPhase::Failure;
                return;
            }
            if H::is_cancellation(&error) {
                self.host.retain_failure(self.phase, error);
                self.phase = HostPhase::Complete;
                return;
            }
            match self.phase {
                HostPhase::MapFailure => {
                    self.host.retain_failure(self.phase, error);
                    self.phase = HostPhase::Failure;
                }
                HostPhase::Failure if self.asynchronous => self.phase = HostPhase::AsyncFailure,
                HostPhase::Failure | HostPhase::AsyncFailure => {
                    self.phase = HostPhase::Complete;
                }
                HostPhase::Success => {
                    self.host.retain_failure(self.phase, error);
                    self.phase = HostPhase::Complete;
                }
                _ => {
                    self.host.retain_failure(self.phase, error);
                    self.phase = HostPhase::MapFailure;
                }
            }
            return;
        }
        self.phase = match self.phase {
            HostPhase::Setup if self.asynchronous => HostPhase::DeploymentPreCall,
            HostPhase::Setup | HostPhase::DeploymentPreCall => HostPhase::Prepare,
            HostPhase::Prepare => HostPhase::PrepareTransport,
            HostPhase::PrepareTransport => HostPhase::PreCall,
            HostPhase::PreCall => HostPhase::Execute,
            HostPhase::Execute => HostPhase::PostCall,
            HostPhase::PostCall => HostPhase::ConstructResponse,
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

    fn cleanup(&mut self) {
        if !self.cleaned {
            self.cleaned = true;
            self.host.cleanup();
        }
    }
}

impl<H: CallHost> Drop for HostLifecycle<H> {
    fn drop(&mut self) {
        self.cleanup();
    }
}
