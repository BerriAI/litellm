use litellm_host::protocol::StreamDelivery;
use std::{
    convert::Infallible,
    ops::ControlFlow,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, hosted_call},
    lifecycle::{CallEvent, CallObserver, observe_call, observe_unary},
    machine::{Machine, MachineFault, MachineStep},
    protocol::{DetachedDrain, HostRequest, Protocol},
};
use rstest::{fixture, rstest};

#[derive(Debug, Clone)]
struct TestError;

struct TestProtocol;

impl Protocol for TestProtocol {
    type Response = &'static str;
    type Error = TestError;
    type Request = usize;
    type HostCall = Infallible;
    type Chunk = usize;
    type StreamHead = &'static str;
}

impl From<MachineFault> for TestError {
    fn from(_: MachineFault) -> Self {
        Self
    }
}

#[rstest]
#[case::end(None, 3, HostedCompletion::StreamEnded)]
#[case::detach_at_open(Some(0), 0, HostedCompletion::Detached)]
#[case::detach_after_chunk(Some(1), 1, HostedCompletion::Detached)]
#[tokio::test]
async fn delivery_obeys_demand_and_distinguishes_detachment(
    #[case] detach_after: Option<usize>,
    #[case] expected_polls: usize,
    #[case] expected: HostedCompletion<&'static str>,
) {
    let polls = Arc::new(AtomicUsize::new(0));
    let stream_polls = polls.clone();
    let mut machine =
        hosted_call::<TestProtocol, _, _>(3, None, move |count, _, _, _observations| async move {
            let chunks = stream::iter((0..count).map(Ok))
                .inspect(move |_| {
                    stream_polls.fetch_add(1, Ordering::SeqCst);
                })
                .boxed();
            Ok(CallOutput::Stream {
                head: "headers",
                chunks,
            })
        });
    let MachineStep::Suspended(HostRequest::Stream(StreamDelivery::Open(head, reply))) =
        machine.resume().await.unwrap()
    else {
        panic!()
    };
    assert_eq!(head, "headers");
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    reply.send(if detach_after == Some(0) {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    });
    let mut delivered = Vec::new();
    let completed = loop {
        match machine.resume().await.unwrap() {
            MachineStep::Suspended(HostRequest::Stream(StreamDelivery::Chunk(chunk, reply))) => {
                delivered.push(chunk);
                assert_eq!(polls.load(Ordering::SeqCst), delivered.len());
                reply.send(if detach_after == Some(delivered.len()) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                });
            }
            MachineStep::Complete(result) => break result,
            _ => panic!("unexpected operation"),
        }
    };
    assert_eq!(completed, expected);
    assert_eq!(polls.load(Ordering::SeqCst), expected_polls);
    assert_eq!(delivered, (0..expected_polls).collect::<Vec<_>>());
}

struct DrainingProtocol;

impl Protocol for DrainingProtocol {
    type Response = &'static str;
    type Error = TestError;
    type Request = usize;
    type HostCall = Infallible;
    type Chunk = usize;
    type StreamHead = Option<DetachedDrain>;

    fn detached_drain(head: &Option<DetachedDrain>) -> Option<DetachedDrain> {
        *head
    }
}

#[rstest]
#[case::draining(
    Some(DetachedDrain { timeout: Duration::from_secs(5), limit: 4 }),
    vec![0, 1, 2]
)]
#[case::no_drain(None, vec![0])]
#[case::no_free_slot(Some(DetachedDrain { timeout: Duration::from_secs(5), limit: 0 }), vec![0])]
#[tokio::test]
async fn a_draining_stream_delivers_the_rest_after_the_caller_detaches(
    #[case] drain: Option<DetachedDrain>,
    #[case] expected: Vec<usize>,
) {
    let mut machine =
        hosted_call::<DrainingProtocol, _, _>(3, None, move |count, _, _, _| async move {
            Ok(CallOutput::Stream {
                head: drain,
                chunks: stream::iter((0..count).map(Ok)).boxed(),
            })
        });
    let mut delivered = Vec::new();
    let completed = loop {
        match machine.resume().await.unwrap() {
            MachineStep::Suspended(HostRequest::Stream(StreamDelivery::Open(_, reply))) => {
                reply.send(ControlFlow::Continue(()));
            }
            MachineStep::Suspended(HostRequest::Stream(StreamDelivery::Chunk(chunk, reply))) => {
                delivered.push(chunk);
                reply.send(ControlFlow::Break(()));
            }
            MachineStep::Complete(result) => break result,
            _ => panic!("unexpected operation"),
        }
    };
    assert_eq!(completed, HostedCompletion::Detached);
    assert_eq!(delivered, expected);
}

#[tokio::test]
async fn a_detached_drain_stops_at_its_timeout() {
    let mut machine =
        hosted_call::<DrainingProtocol, _, _>(0, None, move |_, _, _, _| async move {
            let pending = stream::iter([Ok(0)]).chain(stream::pending());
            Ok(CallOutput::Stream {
                head: Some(DetachedDrain {
                    timeout: Duration::from_millis(50),
                    limit: 4,
                }),
                chunks: pending.boxed(),
            })
        });
    let completed = loop {
        match machine.resume().await.unwrap() {
            MachineStep::Suspended(HostRequest::Stream(StreamDelivery::Open(_, reply))) => {
                reply.send(ControlFlow::Continue(()));
            }
            MachineStep::Suspended(HostRequest::Stream(StreamDelivery::Chunk(_, reply))) => {
                reply.send(ControlFlow::Break(()));
            }
            MachineStep::Complete(result) => break result,
            _ => panic!("unexpected operation"),
        }
    };
    assert_eq!(completed, HostedCompletion::Detached);
}

#[derive(Default)]
struct Observer(Observations);
struct Observations {
    sender: litellm_host::observation::ObservationSender,
    receiver: Mutex<tokio::sync::mpsc::Receiver<CallEvent>>,
    recorded: Mutex<Vec<CallEvent>>,
}

impl Default for Observations {
    fn default() -> Self {
        let (sender, receiver) = litellm_host::observation::observation_channel(
            std::num::NonZeroUsize::new(128).unwrap(),
        );
        Self {
            sender,
            receiver: Mutex::new(receiver),
            recorded: Mutex::new(Vec::new()),
        }
    }
}

impl Observations {
    fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, Vec<CallEvent>>> {
        let mut events = self.recorded.lock()?;
        let mut receiver = self.receiver.lock().unwrap();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        Ok(events)
    }
}

impl CallObserver for Observer {
    fn observe(&self, event: CallEvent) {
        self.0.sender.emit(event);
    }
}

#[fixture]
fn observer() -> Arc<Observer> {
    Arc::new(Observer::default())
}

type Output = CallOutput<(), (), usize, &'static str>;

#[rstest]
#[case::success(false)]
#[case::failure(true)]
#[tokio::test]
async fn unary_calls_emit_one_terminal_event(observer: Arc<Observer>, #[case] fail: bool) {
    let expected = if fail { Err("provider") } else { Ok(7) };
    assert_eq!(
        observe_unary(Some(observer.0.sender.clone()), async { expected }).await,
        expected
    );
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], CallEvent::Started { .. }));
    assert_eq!(matches!(events[1], CallEvent::Failed { .. }), fail);
    assert_eq!(matches!(events[1], CallEvent::Succeeded { .. }), !fail);
}

#[rstest]
#[case::end(false)]
#[case::error(true)]
#[tokio::test]
async fn streams_finish_only_when_consumed(observer: Arc<Observer>, #[case] fail: bool) {
    let chunks = stream::iter([Ok(1), if fail { Err("provider") } else { Ok(2) }]).boxed();
    let output = observe_call(Some(observer.0.sender.clone()), async {
        Ok::<Output, _>(CallOutput::Stream { head: (), chunks })
    })
    .await
    .unwrap();
    assert_eq!(observer.0.lock().unwrap().len(), 1);
    let CallOutput::Stream { mut chunks, .. } = output else {
        panic!()
    };
    assert_eq!(chunks.next().await, Some(Ok(1)));
    assert_eq!(observer.0.lock().unwrap().len(), 1);
    let last = chunks.next().await;
    if fail {
        assert_eq!(last, Some(Err("provider")));
    } else {
        assert_eq!(last, Some(Ok(2)));
        assert_eq!(chunks.next().await, None);
    }
    drop(chunks);
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(matches!(events[1], CallEvent::Failed { .. }), fail);
    assert_eq!(matches!(events[1], CallEvent::Succeeded { .. }), !fail);
}

#[rstest]
#[tokio::test]
async fn dropping_a_stream_cancels_without_success(observer: Arc<Observer>) {
    let chunks = stream::pending().boxed();
    let output = observe_call(Some(observer.0.sender.clone()), async {
        Ok::<Output, _>(CallOutput::Stream { head: (), chunks })
    })
    .await
    .unwrap();
    drop(output);
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[1], CallEvent::Cancelled { .. }));
}

#[rstest]
#[tokio::test]
async fn cancelling_provider_execution_releases_the_lifecycle(observer: Arc<Observer>) {
    let mut call = Box::pin(observe_unary(
        Some(observer.0.sender.clone()),
        std::future::pending::<Result<(), ()>>(),
    ));
    assert!(futures_util::poll!(&mut call).is_pending());
    drop(call);
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[1], CallEvent::Cancelled { .. }));
}
