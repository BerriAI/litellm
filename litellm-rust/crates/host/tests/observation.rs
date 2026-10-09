use std::{cell::Cell, convert::Infallible, num::NonZeroUsize, rc::Rc};

use litellm_host::{
    hooks::NativeHooks,
    interceptors::RawResponse,
    lifecycle::{CallEvent, ExecutionEvent, FailureOrigin, Timing, observe_unary},
    machine::{CallMachine, Machine, MachineFault, MachineStep},
    observation::observation_channel,
    protocol::Protocol,
};
use rstest::{fixture, rstest};

struct TestProtocol;

impl Protocol for TestProtocol {
    type Request = ();
    type Response = usize;
    type Error = MachineFault;
    type HostCall = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

#[fixture]
fn capacity() -> NonZeroUsize {
    NonZeroUsize::new(2).unwrap()
}

#[rstest]
#[case::success(false)]
#[case::failure(true)]
fn snapshots_do_not_retain_runtime_objects(capacity: NonZeroUsize, #[case] failed: bool) {
    let payload = Rc::new(Cell::new(7));
    let retained = Rc::downgrade(&payload);
    let timing = Timing {
        start_time: 11.0,
        end_time: 19.0,
    };
    let event: CallEvent<Rc<Cell<u8>>, Rc<Cell<u8>>> = if failed {
        CallEvent::Failed {
            timing,
            origin: FailureOrigin::Host,
            error: payload,
        }
    } else {
        CallEvent::Succeeded {
            timing,
            response: payload,
        }
    };
    let (sender, mut receiver) = observation_channel(capacity);
    sender.emit(event.snapshot());
    match &event {
        CallEvent::Succeeded { response, .. } => response.set(9),
        CallEvent::Failed { error, .. } => error.set(9),
        _ => unreachable!(),
    }
    assert_eq!(retained.upgrade().unwrap().get(), 9);
    drop(event);
    assert!(retained.upgrade().is_none());
    let expected = if failed {
        CallEvent::Failed {
            timing,
            origin: FailureOrigin::Host,
            error: (),
        }
    } else {
        CallEvent::Succeeded {
            timing,
            response: (),
        }
    };
    assert_eq!(receiver.try_recv().unwrap(), expected);
}

#[rstest]
fn provider_snapshots_own_the_response_body(capacity: NonZeroUsize) {
    let mut raw = RawResponse {
        body: "provider response".into(),
    };
    let event: CallEvent<(), (), &RawResponse> =
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw: &raw });
    let expected = raw.clone();
    let (sender, mut receiver) = observation_channel(capacity);
    sender.emit(event.snapshot());
    raw.body.clear();
    assert_eq!(
        receiver.try_recv().unwrap(),
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw: expected })
    );
}

#[rstest]
#[tokio::test]
async fn observations_do_not_suspend_the_machine(capacity: NonZeroUsize) {
    let (sender, mut receiver) = observation_channel(capacity);
    let mut machine = CallMachine::<TestProtocol>::new(move |_| {
        Box::pin(async move {
            sender.on_event(&CallEvent::Started { start_time: 1.0 });
            Ok(42)
        })
    });
    assert!(matches!(
        machine.resume().await,
        Ok(MachineStep::Complete(42))
    ));
    assert!(matches!(
        receiver.recv().await,
        Some(CallEvent::Started { start_time: 1.0 })
    ));
    assert_eq!(receiver.recv().await, None);
}

#[rstest]
#[case::full(false)]
#[case::closed(true)]
#[tokio::test]
async fn unavailable_observers_do_not_change_the_call_outcome(
    capacity: NonZeroUsize,
    #[case] closed: bool,
) {
    let (sender, mut receiver) = observation_channel(capacity);
    sender.emit(CallEvent::Started { start_time: 1.0 });
    sender.emit(CallEvent::Started { start_time: 2.0 });
    if closed {
        receiver.close();
    }
    let outcome = observe_unary(&sender, async { Err::<(), _>("provider failed") }).await;
    assert_eq!(outcome, Err("provider failed"));
    assert_eq!(sender.dropped_events(), 2);
    assert_eq!(
        receiver.try_recv().unwrap(),
        CallEvent::Started { start_time: 1.0 }
    );
    assert_eq!(
        receiver.try_recv().unwrap(),
        CallEvent::Started { start_time: 2.0 }
    );
    assert!(receiver.try_recv().is_err());
    if !closed {
        sender.emit(CallEvent::Started { start_time: 3.0 });
        assert_eq!(
            receiver.try_recv().unwrap(),
            CallEvent::Started { start_time: 3.0 }
        );
        assert_eq!(sender.dropped_events(), 2);
    }
}

#[rstest]
#[tokio::test]
async fn the_receiver_drains_after_all_publishers_are_dropped(capacity: NonZeroUsize) {
    let (sender, mut receiver) = observation_channel(capacity);
    let other = sender.clone();
    sender.emit(CallEvent::Started { start_time: 1.0 });
    other.emit(CallEvent::Started { start_time: 2.0 });
    other.emit(CallEvent::Started { start_time: 3.0 });
    assert_eq!(sender.dropped_events(), 1);
    assert_eq!(other.dropped_events(), 1);
    drop(sender);
    drop(other);
    assert_eq!(
        receiver.recv().await,
        Some(CallEvent::Started { start_time: 1.0 })
    );
    assert_eq!(
        receiver.recv().await,
        Some(CallEvent::Started { start_time: 2.0 })
    );
    assert_eq!(receiver.recv().await, None);
}
