use crate::Error;
use crate::call_lifecycle::host::{HostFailure, HostLifecycle, HostPhase};

fn run(fail_at: Option<HostPhase>, asynchronous: bool) -> (Vec<HostPhase>, Vec<Error>) {
    let mut lifecycle = HostLifecycle::new(asynchronous);
    let mut events = Vec::new();
    let mut failures = Vec::new();
    while lifecycle.phase() != HostPhase::Complete {
        let phase = lifecycle.phase();
        events.push(phase);
        let result = if Some(phase) == fail_at {
            Err(HostFailure::Error(Error::InvalidRequest(
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
        lifecycle.accept(Ok(()));
    }
    let selected = Error::InvalidRequest("provider".into());
    assert_eq!(
        lifecycle.accept(Err(HostFailure::Error(selected.clone()))),
        Some(selected)
    );
    lifecycle.accept(Ok(()));
    for phase in [
        HostPhase::DeploymentFailure,
        HostPhase::Failure,
        HostPhase::AsyncFailure,
    ] {
        assert_eq!(lifecycle.phase(), phase);
        assert_eq!(
            lifecycle.accept(Err(HostFailure::Error(Error::InvalidRequest(
                "callback".into()
            )))),
            None
        );
    }
    assert_eq!(lifecycle.phase(), HostPhase::Complete);
}

#[test]
fn cancellation_skips_terminal_dispatch() {
    let mut lifecycle = HostLifecycle::new(true);
    let error = Error::InvalidRequest("cancelled".into());
    assert_eq!(
        lifecycle.accept(Err(HostFailure::Cancelled(error.clone()))),
        Some(error)
    );
    assert_eq!(lifecycle.phase(), HostPhase::Complete);
}
