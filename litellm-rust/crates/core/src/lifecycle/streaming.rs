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
    fn observe(&mut self, bytes: &[u8]);
    fn usage(&self) -> Usage;
    fn projection(&self) -> Value;
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
                inner: source.stream,
                observer,
                sender: Some(sender),
            }),
            completion: StreamingCompletion {
                receiver: Some(receiver),
                context: Some(context),
                start_time,
                services,
            },
        }
    }
}

pub struct StreamingCompletion {
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
        tokio::spawn(async move {
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
    inner: BytesStream,
    observer: Box<dyn StreamingObserver>,
    sender: Option<oneshot::Sender<StreamTerminal>>,
}

impl ObservedStream {
    fn complete(&mut self, classification: TerminalClassification) {
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
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                self.observer.observe(&bytes);
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
