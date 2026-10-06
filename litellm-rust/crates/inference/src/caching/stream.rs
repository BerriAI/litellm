use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_cache_response::ResponseEnvelope;
use litellm_host::call::OutputOf;
use serde_json::Value;
use tokio_util::codec::Decoder;

use super::{Cachable, CachedOutput, session::CacheSession};
use crate::RouteError;

pub trait StreamCachable: Cachable {
    const TERMINAL_EVENT: &'static str;

    fn replay(data: Bytes) -> Option<OutputOf<Self>>;
    fn bytes(chunk: &Self::Chunk) -> &[u8];
}

pub(super) fn capture_stream<P: StreamCachable>(
    chunks: futures_util::stream::BoxStream<'static, Result<P::Chunk, RouteError>>,
    session: CacheSession<P>,
) -> futures_util::stream::BoxStream<'static, Result<P::Chunk, RouteError>> {
    stream::try_unfold(
        (chunks, Some(Vec::<u8>::new()), session),
        |(mut chunks, captured, session)| async move {
            match chunks.try_next().await? {
                Some(chunk) => {
                    let captured = captured.and_then(|mut data| {
                        let bytes = P::bytes(&chunk);
                        if data.len().saturating_add(bytes.len()) > session.max_entry_bytes() {
                            return None;
                        }
                        data.extend_from_slice(bytes);
                        Some(data)
                    });
                    Ok(Some((chunk, (chunks, captured, session))))
                }
                None => {
                    if let Some(data) = captured
                        && let Ok(text) = String::from_utf8(data)
                        && successful_stream(&text, P::TERMINAL_EVENT)
                        && let Ok(entry) = serde_json::to_value(ResponseEnvelope::new(
                            P::SURFACE,
                            CachedOutput::<Value>::Stream(text),
                        ))
                    {
                        session.store(entry).await;
                    }
                    Ok::<_, RouteError>(None)
                }
            }
        },
    )
    .boxed()
}

fn successful_stream(text: &str, terminal: &str) -> bool {
    let mut pending = BytesMut::from(text.as_bytes());
    let mut codec = litellm_framer::sse::SseCodec::default();
    let mut complete = false;
    loop {
        let event = match codec.decode(&mut pending) {
            Ok(Some(event)) => event,
            Ok(None) => return complete && pending.is_empty(),
            Err(_) => return false,
        };
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            return false;
        };
        let Some(kind) = value.get("type").and_then(Value::as_str) else {
            return false;
        };
        if matches!(kind, "error" | "response.failed" | "response.incomplete") {
            return false;
        }
        complete |= kind == terminal;
    }
}
