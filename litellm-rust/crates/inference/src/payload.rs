use bytes::{Bytes, BytesMut};
use futures_util::{Stream, stream::BoxStream};
use litellm_framer::sse::SseCodec;
use litellm_tracing::{
    Logger, PayloadShape, ShapeLimits, ShapeSource, ShapeVisitor,
    payload::{self, PayloadEvent, PayloadOutcome, PayloadStage},
};
use serde_json::{Map, Value};
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use tokio_util::codec::Decoder;

pub struct JsonRequest<'a> {
    pub input_name: &'static str,
    pub input: &'a Value,
    pub parameters: &'a Map<String, Value>,
}

impl ShapeSource for JsonRequest<'_> {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool {
        visitor.field("model", &Value::Null)
            && visitor.field(self.input_name, self.input)
            && self
                .parameters
                .iter()
                .all(|(key, value)| visitor.field(key, value))
    }
}

pub fn observe_sse<E: Send + 'static>(
    chunks: BoxStream<'static, Result<Bytes, E>>,
    stage: PayloadStage,
) -> BoxStream<'static, Result<Bytes, E>> {
    if !payload::enabled() {
        return chunks;
    }
    let Some(capture_id) = payload::capture_id() else {
        return chunks;
    };
    Box::pin(ShapeStream {
        chunks,
        stage,
        capture_id,
        shape: PayloadShape::default(),
        codec: SseCodec::default(),
        pending: BytesMut::new(),
        bytes: 0,
        events: 0,
        logger: Logger::current(),
        finished: false,
    })
}

struct ShapeStream<E> {
    chunks: BoxStream<'static, Result<Bytes, E>>,
    stage: PayloadStage,
    capture_id: String,
    shape: PayloadShape,
    codec: SseCodec,
    pending: BytesMut,
    bytes: usize,
    events: usize,
    logger: Logger,
    finished: bool,
}

impl<E> ShapeStream<E> {
    fn truncate(&mut self) {
        self.pending.clear();
        self.shape = PayloadShape {
            field_paths: vec![],
            truncated: true,
        };
    }
    fn observe(&mut self, chunk: &[u8]) {
        if self.finished || self.shape.truncated {
            return;
        }
        self.bytes = self.bytes.saturating_add(chunk.len());
        if self.bytes > 1_048_576 {
            self.truncate();
            return;
        }
        self.pending.extend_from_slice(chunk);
        loop {
            match self.codec.decode(&mut self.pending) {
                Ok(None) => return,
                Err(_) => {
                    self.truncate();
                    return;
                }
                Ok(Some(event)) => {
                    self.events += 1;
                    if self.events > 4096 {
                        self.truncate();
                        return;
                    }
                    if event.data == "[DONE]" {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
                        self.truncate();
                        return;
                    };
                    self.shape.merge(
                        PayloadShape::extract(&value, ShapeLimits::default()),
                        ShapeLimits::default(),
                    );
                    if self.shape.truncated {
                        self.pending.clear();
                        return;
                    }
                }
            }
        }
    }
    fn finish(&mut self, outcome: PayloadOutcome) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.pending.is_empty() {
            self.truncate();
        }
        self.logger.scope(|| {
            PayloadEvent {
                stage: self.stage,
                shape: self.shape.clone(),
                capture_id: self.capture_id.clone(),
                outcome: Some(outcome),
            }
            .emit()
        });
    }
}

impl<E> Stream for ShapeStream<E> {
    type Item = Result<Bytes, E>;
    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = self.chunks.as_mut().poll_next(context);
        match &next {
            Poll::Ready(Some(Ok(chunk))) => self.observe(chunk),
            Poll::Ready(Some(Err(_))) => self.finish(PayloadOutcome::Failure),
            Poll::Ready(None) => self.finish(PayloadOutcome::Success),
            Poll::Pending => {}
        }
        next
    }
}
impl<E> Drop for ShapeStream<E> {
    fn drop(&mut self) {
        self.finish(PayloadOutcome::Cancelled);
    }
}
