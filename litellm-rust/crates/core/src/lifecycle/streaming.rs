use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::Stream;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::Error;
use crate::integrations::custom_logger::CallbackTiming;
use crate::integrations::types::Usage;

use super::{
    CallLifecycleContext, Clock, TerminalClassification, TerminalDispatcher, TerminalRecord,
};

pub type BytesStream = Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingMetadata {
    pub status: u16,
    pub content_type: Option<String>,
    pub cache_control: Option<String>,
}

pub trait StreamingObserver: Send {
    fn observe(&mut self, bytes: &Bytes) -> Result<(), Error>;
    fn usage(&self) -> Usage;
    fn projection(&self) -> Value;
    fn finished(&self) -> bool {
        false
    }
    fn finish(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

pub struct StreamingSource {
    pub metadata: StreamingMetadata,
    pub stream: BytesStream,
}

pub struct StreamingCall {
    pub metadata: StreamingMetadata,
    pub stream: BytesStream,
    pub completion: StreamingCompletion,
}

impl StreamingCall {
    pub(crate) fn new<S>(
        source: StreamingSource,
        observer: Box<dyn StreamingObserver>,
        context: CallLifecycleContext,
        start_time: f64,
        services: Arc<S>,
    ) -> Self
    where
        S: Clock + TerminalDispatcher + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        Self {
            metadata: source.metadata,
            stream: Box::pin(ObservedStream {
                inner: Some(source.stream),
                observer,
                sender: Some(sender),
            }),
            completion: StreamingCompletion {
                runtime: tokio::runtime::Handle::current(),
                receiver: Some(receiver),
                context: Some(context),
                start_time,
                services,
            },
        }
    }
}

pub struct StreamingCompletion {
    runtime: tokio::runtime::Handle,
    receiver: Option<oneshot::Receiver<StreamTerminal>>,
    context: Option<CallLifecycleContext>,
    start_time: f64,
    services: Arc<dyn CompletionServices>,
}

impl StreamingCompletion {
    pub fn register(mut self) -> JoinHandle<TerminalRecord> {
        self.spawn()
    }

    fn spawn(&mut self) -> JoinHandle<TerminalRecord> {
        let receiver = self
            .receiver
            .take()
            .expect("stream completion registered once");
        let context = self
            .context
            .take()
            .expect("stream completion registered once");
        let services = self.services.clone();
        let start_time = self.start_time;
        self.runtime.spawn(async move {
            let terminal_result = receiver.await.unwrap_or_else(|_| StreamTerminal {
                usage: Usage::default(),
                projection: json!({"stream": true}),
                classification: TerminalClassification::Failure {
                    kind: "Cancelled".to_string(),
                    message: "stream completion was cancelled".to_string(),
                },
            });
            let mut context = context;
            context.usage = terminal_result.usage;
            let terminal = context.terminal(
                CallbackTiming::new(start_time, services.now()),
                terminal_result.classification,
                terminal_result.projection,
            );
            let _ = services.dispatch(&terminal).await;
            terminal
        })
    }
}

impl Drop for StreamingCompletion {
    fn drop(&mut self) {
        if self.receiver.is_some() {
            drop(self.spawn());
        }
    }
}

trait CompletionServices: Clock + TerminalDispatcher {}

impl<T> CompletionServices for T where T: Clock + TerminalDispatcher {}

struct StreamTerminal {
    usage: Usage,
    projection: Value,
    classification: TerminalClassification,
}

struct ObservedStream {
    inner: Option<BytesStream>,
    observer: Box<dyn StreamingObserver>,
    sender: Option<oneshot::Sender<StreamTerminal>>,
}

impl ObservedStream {
    fn complete(&mut self, classification: TerminalClassification) {
        self.inner.take();
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(StreamTerminal {
                usage: self.observer.usage(),
                projection: self.observer.projection(),
                classification,
            });
        }
    }
}

impl Stream for ObservedStream {
    type Item = Result<Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.sender.is_none() {
            return Poll::Ready(None);
        }
        match self
            .inner
            .as_mut()
            .expect("active stream")
            .as_mut()
            .poll_next(cx)
        {
            Poll::Ready(Some(Ok(bytes))) => {
                if let Err(error) = self.observer.observe(&bytes) {
                    self.complete(TerminalClassification::Failure {
                        kind: "InvalidResponse".into(),
                        message: error.to_string(),
                    });
                    return Poll::Ready(Some(Err(error)));
                }
                if self.observer.finished() {
                    self.complete(TerminalClassification::Success);
                }
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.complete(TerminalClassification::Failure {
                    kind: "NetworkError".to_string(),
                    message: error.to_string(),
                });
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                if let Err(error) = self.observer.finish() {
                    self.complete(TerminalClassification::Failure {
                        kind: "InvalidResponse".into(),
                        message: error.to_string(),
                    });
                    return Poll::Ready(Some(Err(error)));
                }
                self.complete(TerminalClassification::Success);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ObservedStream {
    fn drop(&mut self) {
        self.complete(TerminalClassification::Failure {
            kind: "Cancelled".to_string(),
            message: "stream consumer dropped before completion".to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    use futures_util::{StreamExt, stream};

    use super::*;

    #[derive(Default)]
    struct Probe {
        polls: AtomicUsize,
        drops: AtomicUsize,
        observations: AtomicUsize,
        wakes: AtomicUsize,
    }

    impl Wake for Probe {
        fn wake(self: Arc<Self>) {
            self.wakes.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct Source {
        stream: BytesStream,
        probe: Arc<Probe>,
    }

    impl Stream for Source {
        type Item = Result<Bytes, Error>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.probe.polls.fetch_add(1, Ordering::Relaxed);
            self.stream.as_mut().poll_next(cx)
        }
    }

    impl Drop for Source {
        fn drop(&mut self) {
            self.probe.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct Observer {
        probe: Arc<Probe>,
        last: Bytes,
    }

    impl StreamingObserver for Observer {
        fn observe(&mut self, bytes: &Bytes) -> Result<(), Error> {
            self.probe.observations.fetch_add(1, Ordering::Relaxed);
            self.last = bytes.clone();
            if self.last == "invalid" {
                Err(Error::InvalidResponse("bad event".into()))
            } else {
                Ok(())
            }
        }

        fn usage(&self) -> Usage {
            Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            }
        }

        fn projection(&self) -> Value {
            json!({"last": String::from_utf8_lossy(&self.last)})
        }

        fn finished(&self) -> bool {
            self.last == "stop"
        }

        fn finish(&mut self) -> Result<(), Error> {
            if self.last == "truncated" {
                Err(Error::InvalidResponse("missing stop".into()))
            } else {
                Ok(())
            }
        }
    }

    fn observed(
        inner: BytesStream,
        probe: &Arc<Probe>,
    ) -> (ObservedStream, oneshot::Receiver<StreamTerminal>) {
        let (sender, receiver) = oneshot::channel();
        (
            ObservedStream {
                inner: Some(Box::pin(Source {
                    stream: inner,
                    probe: probe.clone(),
                })),
                observer: Box::new(Observer {
                    probe: probe.clone(),
                    last: Bytes::new(),
                }),
                sender: Some(sender),
            },
            receiver,
        )
    }

    #[test]
    fn pending_forwards_the_current_waker_without_consuming_or_completing() {
        let probe = Arc::new(Probe::default());
        let (sender, receiver) = futures_channel::mpsc::unbounded();
        let (mut stream, mut terminal) = observed(Box::pin(receiver), &probe);
        let old = Waker::from(probe.clone());
        assert!(
            Pin::new(&mut stream)
                .poll_next(&mut Context::from_waker(&old))
                .is_pending()
        );
        let latest = Arc::new(Probe::default());
        let waker = Waker::from(latest.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut stream).poll_next(&mut cx).is_pending());
        assert_eq!(probe.observations.load(Ordering::Relaxed), 0);
        assert_eq!(probe.polls.load(Ordering::Relaxed), 2);
        assert!(matches!(
            terminal.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let bytes = Bytes::from_static(b"payload");
        sender.unbounded_send(Ok(bytes.clone())).unwrap();
        assert_eq!(latest.wakes.load(Ordering::Relaxed), 1);
        assert_eq!(probe.wakes.load(Ordering::Relaxed), 0);
        let Poll::Ready(Some(Ok(received))) = Pin::new(&mut stream).poll_next(&mut cx) else {
            panic!("expected data")
        };
        assert_eq!(received.as_ptr(), bytes.as_ptr());
        assert_eq!(probe.observations.load(Ordering::Relaxed), 1);
        assert!(matches!(
            terminal.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        drop(stream);
        let terminal = terminal.try_recv().unwrap();
        assert!(
            matches!(terminal.classification, TerminalClassification::Failure { kind, .. } if kind == "Cancelled")
        );
        assert_eq!(terminal.usage.total_tokens, 5);
        assert_eq!(terminal.projection, json!({"last":"payload"}));
        assert_eq!(probe.drops.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn terminal_paths_release_upstream_and_never_poll_it_again() {
        for (item, expected, eof) in [
            (Ok(Bytes::from_static(b"stop")), "Success", false),
            (Ok(Bytes::from_static(b"invalid")), "InvalidResponse", false),
            (Err(Error::Network("broken".into())), "NetworkError", false),
            (
                Ok(Bytes::from_static(b"truncated")),
                "InvalidResponse",
                true,
            ),
            (Ok(Bytes::from_static(b"ordinary")), "Success", true),
        ] {
            let probe = Arc::new(Probe::default());
            let (mut stream, terminal) = observed(Box::pin(stream::iter([item])), &probe);
            let first = stream.next().await.unwrap();
            if eof {
                assert!(first.is_ok());
                let end = stream.next().await;
                assert_eq!(end.is_some(), expected == "InvalidResponse");
                if let Some(error) = end {
                    assert!(error.is_err());
                }
            } else {
                assert_eq!(first.is_ok(), expected == "Success");
            }
            let terminal = tokio::time::timeout(std::time::Duration::from_secs(1), terminal)
                .await
                .unwrap()
                .unwrap();
            match terminal.classification {
                TerminalClassification::Success => assert_eq!(expected, "Success"),
                TerminalClassification::Failure { kind, .. } => assert_eq!(kind, expected),
            }
            assert_eq!(terminal.usage.total_tokens, 5);
            assert_eq!(probe.drops.load(Ordering::Relaxed), 1, "{expected}");
            let polls = probe.polls.load(Ordering::Relaxed);
            for _ in 0..3 {
                assert!(stream.next().await.is_none());
            }
            assert_eq!(probe.polls.load(Ordering::Relaxed), polls);
            drop(stream);
            assert_eq!(probe.drops.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn dropping_before_first_poll_cancels_without_polling_upstream() {
        let probe = Arc::new(Probe::default());
        let (stream, mut terminal) = observed(Box::pin(stream::pending()), &probe);
        drop(stream);
        assert_eq!(probe.polls.load(Ordering::Relaxed), 0);
        assert_eq!(probe.observations.load(Ordering::Relaxed), 0);
        assert_eq!(probe.drops.load(Ordering::Relaxed), 1);
        assert!(matches!(terminal.try_recv().unwrap().classification,
            TerminalClassification::Failure { kind, .. } if kind == "Cancelled"));
    }
    struct Dispatcher(tokio::sync::mpsc::UnboundedSender<TerminalRecord>);

    impl Clock for Dispatcher {
        fn now(&self) -> f64 {
            10.0
        }
    }

    impl TerminalDispatcher for Dispatcher {
        fn dispatch<'a>(
            &'a self,
            terminal: &'a TerminalRecord,
        ) -> crate::integrations::custom_logger::LogFuture<'a> {
            Box::pin(async move {
                self.0.send(terminal.clone()).unwrap();
                Ok(())
            })
        }
    }

    #[test]
    fn unregistered_completion_can_be_dropped_outside_the_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for consume in [false, true] {
            let probe = Arc::new(Probe::default());
            let (sender, mut terminals) = tokio::sync::mpsc::unbounded_channel();
            let call = runtime.block_on(async {
                StreamingCall::new(
                    StreamingSource {
                        metadata: StreamingMetadata {
                            status: 200,
                            content_type: None,
                            cache_control: None,
                        },
                        stream: Box::pin(stream::iter([Ok(Bytes::from_static(b"stop"))])),
                    },
                    Box::new(Observer {
                        probe,
                        last: Bytes::new(),
                    }),
                    CallLifecycleContext::new("messages", "model", "anthropic", "call"),
                    1.0,
                    Arc::new(Dispatcher(sender)),
                )
            });
            drop(call.completion);
            let mut stream = call.stream;
            if consume {
                assert!(runtime.block_on(stream.next()).unwrap().is_ok());
            }
            drop(stream);
            let terminal = runtime.block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(1), terminals.recv())
                    .await
                    .unwrap()
                    .unwrap()
            });
            assert_eq!(
                matches!(terminal.classification, TerminalClassification::Success),
                consume
            );
            assert_eq!(terminal.usage.total_tokens, 5);
            assert!(runtime.block_on(terminals.recv()).is_none());
        }
    }
}
