use std::{sync::OnceLock, time::Duration};

use axum::body::Bytes;
use bytes::BytesMut;
use futures_util::{FutureExt, Stream, StreamExt, stream::BoxStream};

const MAX_BATCH_BYTES: usize = 64 * 1024;

pub(crate) fn window() -> Option<Duration> {
    static WINDOW: OnceLock<Option<Duration>> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("LITELLM_SSE_COALESCE_MS")
            .ok()
            .and_then(|ms| ms.parse::<u64>().ok())
            .map(Duration::from_millis)
    })
}

struct State<E> {
    inner: BoxStream<'static, Result<Bytes, E>>,
    window: Duration,
    stashed: Option<Result<Bytes, E>>,
    done: bool,
    started: bool,
}

pub(crate) fn coalesce<S, E>(inner: S, window: Duration) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let state = State {
        inner: inner.boxed(),
        window,
        stashed: None,
        done: false,
        started: false,
    };
    futures_util::stream::unfold(state, |mut state| async move {
        if let Some(item) = state.stashed.take() {
            return Some((item, state));
        }
        if state.done {
            return None;
        }
        let first = match state.inner.next().await? {
            Ok(first) => first,
            Err(error) => return Some((Err(error), state)),
        };
        if !state.started {
            state.started = true;
            return Some((Ok(first), state));
        }
        let mut batch: Option<BytesMut> = None;
        let deadline = tokio::time::Instant::now() + state.window;
        loop {
            let pending = batch.as_ref().map_or(first.len(), BytesMut::len);
            if pending >= MAX_BATCH_BYTES {
                break;
            }
            let next = if state.window.is_zero() {
                match state.inner.next().now_or_never() {
                    Some(next) => next,
                    None => break,
                }
            } else {
                tokio::select! {
                    biased;
                    next = state.inner.next() => next,
                    () = tokio::time::sleep_until(deadline) => break,
                }
            };
            match next {
                Some(Ok(chunk)) => batch
                    .get_or_insert_with(|| BytesMut::from(&first[..]))
                    .extend_from_slice(&chunk),
                Some(Err(error)) => {
                    state.stashed = Some(Err(error));
                    break;
                }
                None => {
                    state.done = true;
                    break;
                }
            }
        }
        let out = batch.map_or(first, BytesMut::freeze);
        Some((Ok(out), state))
    })
}
