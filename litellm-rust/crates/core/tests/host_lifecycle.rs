use std::sync::{Arc, Mutex};

use crate::call_lifecycle::host::{CallHost, HostLifecycle, HostPhase, HostStep};

struct RecordingHost {
    events: Arc<Mutex<Vec<HostPhase>>>,
    fail_at: Option<HostPhase>,
    suspend_at: Option<HostPhase>,
    failure: Option<Arc<String>>,
    outcome: Arc<String>,
    cleanups: Arc<Mutex<usize>>,
}

impl CallHost for RecordingHost {
    type Value = Arc<String>;
    type Error = Arc<String>;
    type Suspension = HostPhase;

    fn invoke(
        &mut self,
        phase: HostPhase,
    ) -> Result<HostStep<Self::Value, Self::Suspension>, Self::Error> {
        self.events.lock().unwrap().push(phase);
        if self.fail_at == Some(phase) {
            return Err(self.outcome.clone());
        }
        if self.suspend_at == Some(phase) {
            return Ok(HostStep::Suspend(phase));
        }
        if matches!(phase, HostPhase::Failure | HostPhase::AsyncFailure) {
            assert!(Arc::ptr_eq(self.failure.as_ref().unwrap(), &self.outcome));
        }
        Ok(HostStep::Ready(self.outcome.clone()))
    }

    fn accept(&mut self, _: HostPhase, _: Self::Value) -> Result<(), Self::Error> {
        Ok(())
    }
    fn retain_failure(&mut self, _: HostPhase, error: Self::Error) {
        self.failure = Some(error);
    }
    fn is_cancellation(error: &Self::Error) -> bool {
        error.as_str() == "cancelled"
    }
    fn finish(&mut self) -> Result<Self::Value, Self::Error> {
        self.failure
            .take()
            .map_or_else(|| Ok(self.outcome.clone()), Err)
    }
    fn cleanup(&mut self) {
        *self.cleanups.lock().unwrap() += 1;
    }
}

fn host(fail_at: Option<HostPhase>, suspend_at: Option<HostPhase>) -> RecordingHost {
    RecordingHost {
        events: Arc::new(Mutex::new(Vec::new())),
        fail_at,
        suspend_at,
        failure: None,
        outcome: Arc::new("public outcome".into()),
        cleanups: Arc::new(Mutex::new(0)),
    }
}

#[test]
fn public_outcome_is_finalized_before_a_single_terminal_dispatch() {
    for asynchronous in [false, true] {
        let host = host(None, Some(HostPhase::Execute));
        let events = host.events.clone();
        let cleanups = host.cleanups.clone();
        let outcome = host.outcome.clone();
        let mut lifecycle = HostLifecycle::new(host, asynchronous);
        assert!(matches!(
            lifecycle.resume(None).unwrap(),
            HostStep::Suspend(HostPhase::Execute)
        ));
        assert!(!events.lock().unwrap().contains(&HostPhase::Success));
        let HostStep::Ready(returned) = lifecycle.resume(Some(Ok(outcome.clone()))).unwrap() else {
            panic!("call did not complete")
        };
        assert!(Arc::ptr_eq(&returned, &outcome));
        drop(lifecycle);
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|phase| **phase == HostPhase::Execute)
                .count(),
            1
        );
        assert_eq!(
            &events[events.len() - 2..],
            &[HostPhase::Finalize, HostPhase::Success]
        );
        assert_eq!(*cleanups.lock().unwrap(), 1);
    }
}

#[test]
fn every_fallible_phase_maps_one_failure_without_reexecution() {
    for phase in [
        HostPhase::Setup,
        HostPhase::DeploymentPreCall,
        HostPhase::Prepare,
        HostPhase::PrepareTransport,
        HostPhase::PreCall,
        HostPhase::Execute,
        HostPhase::PostCall,
        HostPhase::ConstructResponse,
        HostPhase::DeploymentPostCall,
        HostPhase::Finalize,
    ] {
        let host = host(Some(phase), None);
        let events = host.events.clone();
        let cleanups = host.cleanups.clone();
        let outcome = host.outcome.clone();
        let mut lifecycle = HostLifecycle::new(host, true);
        let Err(error) = lifecycle.resume(None) else {
            panic!("call should fail")
        };
        assert!(Arc::ptr_eq(&error, &outcome));
        drop(lifecycle);
        let events = events.lock().unwrap();
        assert_eq!(
            &events[events.len() - 4..],
            &[
                HostPhase::MapFailure,
                HostPhase::DeploymentFailure,
                HostPhase::Failure,
                HostPhase::AsyncFailure
            ]
        );
        assert!(!events.contains(&HostPhase::Success));
        assert!(
            events
                .iter()
                .filter(|phase| **phase == HostPhase::Execute)
                .count()
                <= 1
        );
        assert_eq!(*cleanups.lock().unwrap(), 1);
    }
}

#[test]
fn cancellation_or_dropped_pending_work_cleans_up_without_dispatch() {
    for cancel in [false, true] {
        let host = host(None, Some(HostPhase::DeploymentPostCall));
        let events = host.events.clone();
        let cleanups = host.cleanups.clone();
        let mut lifecycle = HostLifecycle::new(host, true);
        assert!(matches!(
            lifecycle.resume(None).unwrap(),
            HostStep::Suspend(_)
        ));
        if cancel {
            let cancellation = Arc::new("cancelled".to_string());
            let Err(error) = lifecycle.resume(Some(Err(cancellation.clone()))) else {
                panic!("cancellation should propagate")
            };
            assert!(Arc::ptr_eq(&error, &cancellation));
        }
        drop(lifecycle);
        let events = events.lock().unwrap();
        assert!(!events.contains(&HostPhase::Success));
        assert!(!events.contains(&HostPhase::Failure));
        assert_eq!(*cleanups.lock().unwrap(), 1);
    }
}
