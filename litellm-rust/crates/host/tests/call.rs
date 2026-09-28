use litellm_host::protocol::StreamDelivery;
use std::{
    convert::Infallible,
    ops::ControlFlow,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, hosted_call},
    event::CallEvent,
    lifecycle::{CallObserver, observe_call, observe_unary},
    machine::{Machine, MachineFault, MachineStep},
    protocol::{HostRequest, Protocol},
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
    let mut machine = hosted_call::<TestProtocol, _, _>(3, move |count, _, _| async move {
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

#[derive(Default)]
struct Observer(Mutex<Vec<CallEvent>>);

impl CallObserver for Observer {
    fn observe(&self, event: CallEvent) {
        self.0.lock().unwrap().push(event);
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
        observe_unary(Some(observer.clone()), async { expected }).await,
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
    let output = observe_call(Some(observer.clone()), async {
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
    let output = observe_call(Some(observer.clone()), async {
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
        Some(observer.clone()),
        std::future::pending::<Result<(), ()>>(),
    ));
    assert!(futures_util::poll!(&mut call).is_pending());
    drop(call);
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[1], CallEvent::Cancelled { .. }));
}
