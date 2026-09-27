use std::{
    pin::Pin,
    task::{Context, Poll},
};

use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt, stream::BoxStream};
use litellm_cost::{PromptConvention, Usage};
use litellm_framing::sse::SseCodec;
use serde_json::Value;
use tokio::sync::watch;
use tokio_util::codec::Decoder;

#[derive(Clone, Debug, PartialEq)]
pub struct Metered {
    pub model_group: String,
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub usage: Usage,
    pub completed: bool,
}

#[derive(Clone, Debug)]
pub struct Metering(watch::Receiver<Option<Metered>>);

impl Metering {
    pub async fn settled(self) -> Option<Metered> {
        let Self(mut receiver) = self;
        receiver
            .wait_for(Option::is_some)
            .await
            .ok()
            .and_then(|settled| settled.clone())
    }
}

pub(crate) struct Meter {
    metered: Metered,
    sender: Option<watch::Sender<Option<Metered>>>,
}

impl Meter {
    pub(crate) fn start(
        model_group: &str,
        model: &str,
        custom_llm_provider: Option<&str>,
    ) -> (Self, Metering) {
        let (sender, receiver) = watch::channel(None);
        let meter = Self {
            metered: Metered {
                model_group: model_group.to_owned(),
                model: model.to_owned(),
                custom_llm_provider: custom_llm_provider.map(str::to_owned),
                usage: NO_USAGE,
                completed: false,
            },
            sender: Some(sender),
        };
        (meter, Metering(receiver))
    }

    pub(crate) fn complete(mut self, usage: &Value) {
        self.absorb(usage);
        self.metered.completed = true;
        self.settle();
    }

    fn absorb(&mut self, usage: &Value) {
        self.metered.usage = merged_usage(self.metered.usage, usage);
    }

    fn settle(&mut self) {
        if let Some(sender) = self.sender.take() {
            sender.send_replace(Some(self.metered.clone()));
        }
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        self.settle();
    }
}

const NO_USAGE: Usage = Usage {
    prompt_tokens: 0,
    completion_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cache_write_5m_tokens: None,
    cache_write_1h_tokens: None,
    prompt_convention: PromptConvention::ExcludesCache,
};

fn merged_usage(so_far: Usage, update: &Value) -> Usage {
    let field =
        |name: &str, current: u64| update.get(name).and_then(Value::as_u64).unwrap_or(current);
    Usage {
        prompt_tokens: field("input_tokens", so_far.prompt_tokens),
        completion_tokens: field("output_tokens", so_far.completion_tokens),
        cache_read_tokens: field("cache_read_input_tokens", so_far.cache_read_tokens),
        cache_write_tokens: field("cache_creation_input_tokens", so_far.cache_write_tokens),
        ..so_far
    }
}

pub(crate) struct MeteredStream<E> {
    chunks: BoxStream<'static, Result<Bytes, E>>,
    pending: BytesMut,
    codec: SseCodec,
    meter: Meter,
}

impl<E> MeteredStream<E> {
    pub(crate) fn new(chunks: BoxStream<'static, Result<Bytes, E>>, meter: Meter) -> Self {
        Self {
            chunks,
            pending: BytesMut::new(),
            codec: SseCodec::default(),
            meter,
        }
    }

    fn observe(&mut self, chunk: &[u8]) {
        self.pending.extend_from_slice(chunk);
        while let Ok(Some(event)) = self.codec.decode(&mut self.pending) {
            let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            match data.get("type").and_then(Value::as_str) {
                Some("message_start") => self.meter.absorb(&data["message"]["usage"]),
                Some("message_delta") => self.meter.absorb(&data["usage"]),
                Some("message_stop") => self.meter.metered.completed = true,
                _ => {}
            }
        }
    }
}

impl<E> Stream for MeteredStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.chunks.poll_next_unpin(cx);
        match &polled {
            Poll::Ready(Some(Ok(chunk))) => self.observe(chunk),
            Poll::Ready(None) => self.meter.settle(),
            Poll::Ready(Some(Err(_))) | Poll::Pending => {}
        }
        polled
    }
}
