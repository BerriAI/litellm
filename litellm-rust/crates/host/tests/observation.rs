use std::{convert::Infallible, num::NonZeroUsize};

use litellm_host::{
    lifecycle::{CallEvent, observe_unary},
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
#[tokio::test]
async fn observations_do_not_suspend_the_machine(capacity: NonZeroUsize) {
    let (sender, mut receiver) = observation_channel(capacity);
    let mut machine = CallMachine::<TestProtocol>::new(Some(sender), |host| {
        Box::pin(async move {
            host.observers
                .unwrap()
                .emit(CallEvent::Started { start_time: 1.0 });
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
    let outcome = observe_unary(Some(sender.clone()), async {
        Err::<(), _>("provider failed")
    })
    .await;
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
