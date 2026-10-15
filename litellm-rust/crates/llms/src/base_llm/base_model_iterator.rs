use std::{collections::VecDeque, convert::Infallible, io, pin::Pin};

use bytes::Bytes;
use futures_util::{Stream, StreamExt, stream, stream::BoxStream};

pub type ByteStream = BoxStream<'static, Result<Bytes, io::Error>>;

pub trait StreamTransformer {
    type Input;
    type Output;
    type Error;

    fn transform(&mut self, input: Self::Input) -> Result<Vec<Self::Output>, Self::Error>;

    fn finish(&mut self) -> Result<Vec<Self::Output>, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StreamError<D, T> {
    #[error(transparent)]
    Decode(D),
    #[error(transparent)]
    Transform(T),
}

impl<D> StreamError<D, Infallible> {
    pub fn into_decode(self) -> D {
        match self {
            Self::Decode(error) => error,
            Self::Transform(never) => match never {},
        }
    }
}

struct Driver<S, T: StreamTransformer> {
    events: Pin<Box<S>>,
    transformer: T,
    ready: VecDeque<T::Output>,
    finished: bool,
}

/// Drives `transformer` over `events`, then flushes it with `finish`. The first error ends the
/// stream.
pub fn transform_stream<S, T, D>(
    events: S,
    transformer: T,
) -> impl Stream<Item = Result<T::Output, StreamError<D, T::Error>>> + Send
where
    S: Stream<Item = Result<T::Input, D>> + Send,
    T: StreamTransformer + Send,
    T::Output: Send,
    T::Error: Send,
    D: Send,
{
    let driver = Driver {
        events: Box::pin(events),
        transformer,
        ready: VecDeque::new(),
        finished: false,
    };
    stream::unfold(driver, |mut driver| async move {
        loop {
            if let Some(output) = driver.ready.pop_front() {
                return Some((Ok(output), driver));
            }
            if driver.finished {
                return None;
            }
            match driver.events.next().await {
                Some(Ok(event)) => match driver.transformer.transform(event) {
                    Ok(outputs) => driver.ready.extend(outputs),
                    Err(error) => {
                        driver.finished = true;
                        return Some((Err(StreamError::Transform(error)), driver));
                    }
                },
                Some(Err(error)) => {
                    driver.finished = true;
                    return Some((Err(StreamError::Decode(error)), driver));
                }
                None => {
                    driver.finished = true;
                    match driver.transformer.finish() {
                        Ok(outputs) => driver.ready.extend(outputs),
                        Err(error) => return Some((Err(StreamError::Transform(error)), driver)),
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use futures_util::TryStreamExt;

    use super::*;

    struct Doubler;

    impl StreamTransformer for Doubler {
        type Input = u32;
        type Output = u32;
        type Error = String;

        fn transform(&mut self, input: u32) -> Result<Vec<u32>, String> {
            match input {
                0 => Err("zero".into()),
                n => Ok(vec![n, n * 2]),
            }
        }

        fn finish(&mut self) -> Result<Vec<u32>, String> {
            Ok(vec![u32::MAX])
        }
    }

    #[tokio::test]
    async fn flat_maps_each_event_and_flushes_at_the_end() {
        let output = transform_stream(stream::iter([Ok::<_, String>(1), Ok(2)]), Doubler)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(output, vec![1, 2, 2, 4, u32::MAX]);
    }

    #[tokio::test]
    async fn a_transform_error_ends_the_stream_without_flushing() {
        let output = transform_stream(stream::iter([Ok::<_, String>(1), Ok(0), Ok(3)]), Doubler)
            .collect::<Vec<_>>()
            .await;

        assert_eq!(
            output,
            vec![
                Ok(1),
                Ok(2),
                Err(StreamError::Transform("zero".to_string()))
            ]
        );
    }

    #[tokio::test]
    async fn a_decode_error_ends_the_stream_without_flushing() {
        let output = transform_stream(
            stream::iter([Ok(1), Err("bad frame".to_string()), Ok(3)]),
            Doubler,
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(
            output,
            vec![
                Ok(1),
                Ok(2),
                Err(StreamError::Decode("bad frame".to_string()))
            ]
        );
    }
}
