use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::{Stream, StreamExt};
use litellm_core::CoreError;
use litellm_python_extension_protocol::{
    AuthContext, InvocationContext, PublicError, StreamFrame, StreamFrameKind, StreamOpen,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::adapters::next_invocation_id;
use super::client::PythonExtensionClient;

#[derive(Clone)]
pub struct RemoteStreamTransformer {
    plugin_id: String,
    client: Arc<PythonExtensionClient>,
    iterator_hook: bool,
}

impl RemoteStreamTransformer {
    pub fn new(plugin_id: String, client: Arc<PythonExtensionClient>, iterator_hook: bool) -> Self {
        Self {
            plugin_id,
            client,
            iterator_hook,
        }
    }

    pub fn transform<S>(
        &self,
        request: Value,
        auth: AuthContext,
        input: S,
    ) -> ReceiverStream<Result<Value, CoreError>>
    where
        S: Stream<Item = Result<Value, CoreError>> + Send + Unpin + 'static,
    {
        let stream_id = next_invocation_id();
        let (frame_tx, frame_rx) = mpsc::channel(8);
        let (output_tx, output_rx) = mpsc::channel(8);
        let pending = Arc::new(Mutex::new(VecDeque::new()));
        let failed = Arc::new(AtomicBool::new(false));
        let terminal_error = Arc::new(Mutex::new(None));
        let producer = tokio::spawn(produce_frames(
            frame_tx,
            output_tx.clone(),
            input,
            pending.clone(),
            failed.clone(),
            terminal_error.clone(),
            StreamOpen {
                context: Some(InvocationContext {
                    request_id: stream_id.clone(),
                    invocation_id: stream_id.clone(),
                    active_revision: self.client.manifest().revision_id.clone(),
                    api_surface: "stream".to_string(),
                    call_type: "stream".to_string(),
                    trace_context: Default::default(),
                }),
                plugin_id: self.plugin_id.clone(),
                request_json: serde_json::to_vec(&request).unwrap_or_else(|_| b"{}".to_vec()),
                auth: Some(auth),
                cache: None,
                iterator_hook: self.iterator_hook,
            },
            stream_id.clone(),
        ));
        let client = self.client.clone();
        let consumer_output = output_tx.clone();
        tokio::spawn(async move {
            let result = client.transform_stream(ReceiverStream::new(frame_rx)).await;
            let outcome = match result {
                Ok(mut output) => {
                    consume_output(&mut output, &consumer_output, &terminal_error).await
                }
                Err(_) => ConsumeOutcome::Failed,
            };
            match outcome {
                ConsumeOutcome::Complete | ConsumeOutcome::Cancelled => producer.abort(),
                ConsumeOutcome::Failed => {
                    failed.store(true, Ordering::Release);
                    client.record_stream_bypass("remote_stream_failed");
                    let originals = pending
                        .lock()
                        .map(|mut values| values.drain(..).collect::<Vec<_>>())
                        .unwrap_or_default();
                    for original in originals {
                        if consumer_output.send(Ok(original)).await.is_err() {
                            break;
                        }
                    }
                    let upstream_error = terminal_error
                        .lock()
                        .ok()
                        .and_then(|mut error| error.take());
                    if let Some(error) = upstream_error {
                        let _ = consumer_output.send(Err(error)).await;
                    }
                }
            }
        });
        drop(output_tx);
        ReceiverStream::new(output_rx)
    }
}

async fn produce_frames<S>(
    sender: mpsc::Sender<StreamFrame>,
    output: mpsc::Sender<Result<Value, CoreError>>,
    mut input: S,
    pending: Arc<Mutex<VecDeque<Value>>>,
    failed: Arc<AtomicBool>,
    terminal_error: Arc<Mutex<Option<CoreError>>>,
    open: StreamOpen,
    stream_id: String,
) where
    S: Stream<Item = Result<Value, CoreError>> + Send + Unpin + 'static,
{
    if sender
        .send(StreamFrame {
            kind: StreamFrameKind::Open.into(),
            stream_id: stream_id.clone(),
            open: Some(open),
            ..Default::default()
        })
        .await
        .is_err()
    {
        failed.store(true, Ordering::Release);
        forward_remaining(&mut input, &output).await;
        return;
    }
    while let Some(chunk) = input.next().await {
        match chunk {
            Ok(chunk) => {
                if failed.load(Ordering::Acquire) {
                    forward_pending(&pending, &output).await;
                    if output.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    continue;
                }
                if let Ok(mut values) = pending.lock() {
                    values.push_back(chunk.clone());
                }
                if sender
                    .send(StreamFrame {
                        kind: StreamFrameKind::InputChunk.into(),
                        stream_id: stream_id.clone(),
                        chunk_json: serde_json::to_vec(&chunk).ok(),
                        ..Default::default()
                    })
                    .await
                    .is_err()
                {
                    failed.store(true, Ordering::Release);
                    let originals = pending
                        .lock()
                        .map(|mut values| values.drain(..).collect::<Vec<_>>())
                        .unwrap_or_default();
                    for original in originals {
                        if output.send(Ok(original)).await.is_err() {
                            return;
                        }
                    }
                } else if failed.load(Ordering::Acquire) {
                    forward_pending(&pending, &output).await;
                }
            }
            Err(error) => {
                let message = error.to_string();
                if let Ok(mut terminal) = terminal_error.lock() {
                    *terminal = Some(error);
                }
                if failed.load(Ordering::Acquire) {
                    let upstream_error = terminal_error
                        .lock()
                        .ok()
                        .and_then(|mut value| value.take());
                    if let Some(error) = upstream_error {
                        let _ = output.send(Err(error)).await;
                    }
                    return;
                }
                let _ = sender
                    .send(StreamFrame {
                        kind: StreamFrameKind::Error.into(),
                        stream_id,
                        error: Some(PublicError {
                            r#type: "upstream_error".to_string(),
                            message,
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .await;
                return;
            }
        }
    }
    let _ = sender
        .send(StreamFrame {
            kind: StreamFrameKind::End.into(),
            stream_id,
            ..Default::default()
        })
        .await;
}

async fn forward_remaining<S>(input: &mut S, output: &mpsc::Sender<Result<Value, CoreError>>)
where
    S: Stream<Item = Result<Value, CoreError>> + Send + Unpin + 'static,
{
    while let Some(chunk) = input.next().await {
        if output.send(chunk).await.is_err() {
            return;
        }
    }
}

async fn forward_pending(
    pending: &Arc<Mutex<VecDeque<Value>>>,
    output: &mpsc::Sender<Result<Value, CoreError>>,
) {
    let originals = pending
        .lock()
        .map(|mut values| values.drain(..).collect::<Vec<_>>())
        .unwrap_or_default();
    for original in originals {
        if output.send(Ok(original)).await.is_err() {
            return;
        }
    }
}

enum ConsumeOutcome {
    Complete,
    Failed,
    Cancelled,
}

async fn consume_output(
    output: &mut tonic::Streaming<StreamFrame>,
    sender: &mpsc::Sender<Result<Value, CoreError>>,
    terminal_error: &Arc<Mutex<Option<CoreError>>>,
) -> ConsumeOutcome {
    while let Some(frame) = output.next().await {
        let Ok(frame) = frame else {
            return ConsumeOutcome::Failed;
        };
        match StreamFrameKind::try_from(frame.kind).unwrap_or(StreamFrameKind::Error) {
            StreamFrameKind::OutputChunk => {
                let Some(chunk_json) = frame.chunk_json else {
                    return ConsumeOutcome::Failed;
                };
                let Ok(chunk) = serde_json::from_slice(&chunk_json) else {
                    return ConsumeOutcome::Failed;
                };
                if sender.send(Ok(chunk)).await.is_err() {
                    return ConsumeOutcome::Cancelled;
                }
            }
            StreamFrameKind::End => return ConsumeOutcome::Complete,
            StreamFrameKind::Error => {
                let upstream_error = terminal_error
                    .lock()
                    .ok()
                    .and_then(|mut error| error.take());
                if let Some(error) = upstream_error {
                    return if sender.send(Err(error)).await.is_ok() {
                        ConsumeOutcome::Complete
                    } else {
                        ConsumeOutcome::Cancelled
                    };
                }
                return ConsumeOutcome::Failed;
            }
            _ => return ConsumeOutcome::Failed,
        }
    }
    ConsumeOutcome::Failed
}
