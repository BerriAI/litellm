use std::future::Future;
use std::pin::Pin;

pub enum HostCallStep<O, C> {
    Host(O),
    Complete(C),
}

pub type HostCallFuture<'a, O, C, E> =
    Pin<Box<dyn Future<Output = Result<HostCallStep<O, C>, E>> + Send + 'a>>;

pub trait HostCall: Send + Sync {
    type Error: Send + Sync + 'static;
    type Operation: Send + 'static;
    type Result: Send + 'static;
    type Complete: Send + 'static;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete, Self::Error>;

    fn interrupt(
        &mut self,
        failure: HostFailure<Self::Error>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete, Self::Error>;
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

#[cfg(test)]
mod tests {
    use super::{HostFailure, HostLifecycle, HostPhase};

    fn run(
        fail_at: Option<HostPhase>,
        asynchronous: bool,
    ) -> (Vec<HostPhase>, Vec<crate::ocr::Error>) {
        let mut lifecycle = HostLifecycle::new(asynchronous);
        let mut events = Vec::new();
        let mut failures = Vec::new();

        while lifecycle.phase() != HostPhase::Complete {
            let phase = lifecycle.phase();
            events.push(phase);
            let result = if Some(phase) == fail_at {
                Err(HostFailure::Error(crate::ocr::Error::InvalidRequest(
                    "selected failure".into(),
                )))
            } else {
                Ok(())
            };
            if let Some(error) = lifecycle.accept(result) {
                failures.push(error);
            }
        }
        (events, failures)
    }

    #[test]
    fn public_outcome_is_finalized_before_a_single_terminal_dispatch() {
        for asynchronous in [false, true] {
            let (events, failures) = run(None, asynchronous);
            assert!(failures.is_empty());
            assert_eq!(
                &events[events.len() - 2..],
                &[HostPhase::Finalize, HostPhase::Success]
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|phase| **phase == HostPhase::Execute)
                    .count(),
                1
            );
            assert_eq!(
                events.contains(&HostPhase::DeploymentPostCall),
                asynchronous
            );
        }
    }

    #[test]
    fn only_provider_and_response_construction_failures_use_provider_mapping() {
        for phase in [
            HostPhase::Setup,
            HostPhase::DeploymentPreCall,
            HostPhase::Prepare,
            HostPhase::Execute,
            HostPhase::ConstructResponse,
            HostPhase::DeploymentPostCall,
            HostPhase::Finalize,
        ] {
            let (events, failures) = run(Some(phase), true);
            assert_eq!(failures.len(), 1);
            assert!(!events.contains(&HostPhase::Success));
            let mapped = matches!(phase, HostPhase::Execute | HostPhase::ConstructResponse);
            assert_eq!(events.contains(&HostPhase::MapFailure), mapped);
            assert_eq!(events.contains(&HostPhase::DeploymentFailure), mapped);
            assert_eq!(
                &events[events.len() - 2..],
                &[HostPhase::Failure, HostPhase::AsyncFailure]
            );
            assert!(
                events
                    .iter()
                    .filter(|phase| **phase == HostPhase::Execute)
                    .count()
                    <= 1
            );
        }
    }

    #[test]
    fn failure_handler_errors_do_not_replace_selected_failure_or_suppress_async_dispatch() {
        let mut lifecycle = HostLifecycle::new(true);
        while lifecycle.phase() != HostPhase::Execute {
            lifecycle.accept::<crate::ocr::Error>(Ok(()));
        }
        let selected = crate::ocr::Error::InvalidRequest("provider".into());
        assert!(matches!(
            lifecycle.accept(Err(HostFailure::Error(selected.clone()))),
            Some(crate::ocr::Error::InvalidRequest(message)) if message == "provider"
        ));
        lifecycle.accept::<crate::ocr::Error>(Ok(()));
        for phase in [
            HostPhase::DeploymentFailure,
            HostPhase::Failure,
            HostPhase::AsyncFailure,
        ] {
            assert_eq!(lifecycle.phase(), phase);
            assert!(
                lifecycle
                    .accept(Err(HostFailure::Error(crate::ocr::Error::InvalidRequest(
                        "callback".into()
                    ))))
                    .is_none()
            );
        }
        assert_eq!(lifecycle.phase(), HostPhase::Complete);
    }

    #[test]
    fn cancellation_skips_terminal_dispatch() {
        let mut lifecycle = HostLifecycle::new(true);
        let error = crate::ocr::Error::InvalidRequest("cancelled".into());
        assert!(matches!(
            lifecycle.accept(Err(HostFailure::Cancelled(error.clone()))),
            Some(crate::ocr::Error::InvalidRequest(message)) if message == "cancelled"
        ));
        assert_eq!(lifecycle.phase(), HostPhase::Complete);
    }
}
