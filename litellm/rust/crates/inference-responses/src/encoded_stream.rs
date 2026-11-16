//! A Responses event stream carrying LiteLLM's managed ids: each event is rewritten by
//! [`encode_stream_event`] and re-framed only when that changed it, so every other event keeps the
//! provider's bytes.

use bytes::{Bytes, BytesMut};
use futures_util::{
    StreamExt, future,
    stream::{self, BoxStream},
};
use litellm_framer::sse::{SseBlock, SseBlockCodec};
use serde_json::Value;
use tokio_util::codec::Decoder;

use crate::ids::{IdScope, encode_stream_event};

struct Framing {
    codec: SseBlockCodec,
    pending: BytesMut,
    scope: IdScope,
}

impl Framing {
    /// The complete events pending holds, encoded; at the end, whatever is left as it arrived.
    fn drain(&mut self, end: bool) -> Bytes {
        let mut out = BytesMut::new();
        loop {
            let decoded = if end {
                self.codec.decode_eof(&mut self.pending)
            } else {
                self.codec.decode(&mut self.pending)
            };
            match decoded {
                Ok(Some(block)) => out.extend_from_slice(&encoded(block, &self.scope)),
                Ok(None) => break,
                Err(_) => {
                    out.extend_from_slice(&self.pending.split());
                    break;
                }
            }
        }
        if end {
            out.extend_from_slice(&self.pending.split());
        }
        out.freeze()
    }
}

fn encoded(block: SseBlock, scope: &IdScope) -> Bytes {
    let Ok(Value::Object(original)) = serde_json::from_str::<Value>(&block.event.data) else {
        return block.bytes;
    };
    let mut event = original.clone();
    encode_stream_event(&mut event, scope);
    if event == original {
        return block.bytes;
    }
    let named = block
        .event
        .event
        .map(|name| format!("event: {name}\n"))
        .unwrap_or_default();
    Bytes::from(format!("{named}data: {}\n\n", Value::Object(event)))
}

pub fn encode_stream<E: Send + 'static>(
    chunks: BoxStream<'static, Result<Bytes, E>>,
    scope: IdScope,
) -> BoxStream<'static, Result<Bytes, E>> {
    let framing = Framing {
        codec: SseBlockCodec::default(),
        pending: BytesMut::new(),
        scope,
    };
    chunks
        .map(Some)
        .chain(stream::once(future::ready(None)))
        .scan(framing, |framing, chunk| {
            future::ready(Some(match chunk {
                Some(Ok(bytes)) => {
                    framing.pending.extend_from_slice(&bytes);
                    Ok(framing.drain(false))
                }
                Some(Err(error)) => Err(error),
                None => Ok(framing.drain(true)),
            }))
        })
        .filter(|chunk| future::ready(!matches!(chunk, Ok(bytes) if bytes.is_empty())))
        .boxed()
}
