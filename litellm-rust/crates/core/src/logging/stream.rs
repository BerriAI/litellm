use std::fmt::Display;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::Stream;
use futures_util::StreamExt;

use super::CallLogger;

pub fn count_forwarded_stream<S, E>(
    stream: S,
    logger: Arc<CallLogger>,
) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Display + Send + 'static,
{
    futures_util::stream::unfold(
        (Box::pin(stream), logger, false, false),
        |(mut stream, logger, trailing_newline, failed)| async move {
            match stream.next().await {
                None => {
                    if !failed {
                        logger.stream_finished();
                    }
                    None
                }
                Some(Ok(bytes)) => {
                    let events = bytes
                        .iter()
                        .enumerate()
                        .filter(|(index, byte)| {
                            **byte == b'\n'
                                && ((*index > 0 && bytes[*index - 1] == b'\n')
                                    || (*index == 0 && trailing_newline))
                        })
                        .count();
                    let trailing_newline = bytes.last().copied() == Some(b'\n');
                    logger.stream_chunk_observed(bytes.len(), events);
                    Some((Ok(bytes), (stream, logger, trailing_newline, failed)))
                }
                Some(Err(error)) => {
                    logger.failure(None, "stream_error", error.to_string(), None);
                    Some((Err(error), (stream, logger, trailing_newline, true)))
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use futures_util::StreamExt;

    use crate::call_lifecycle::CallLifecycleContext;

    use super::super::{CallLogger, LogEvent, LogSink};
    use super::count_forwarded_stream;

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<LogEvent>>>);

    impl LogSink for Sink {
        fn emit(&self, event: &LogEvent) {
            self.0.lock().expect("lock").push(event.clone());
        }
    }

    fn logger(sink: Sink) -> Arc<CallLogger> {
        Arc::new(CallLogger::new(
            &CallLifecycleContext::new("messages", "model", "anthropic", "call"),
            Some(Arc::new(sink)),
        ))
    }

    #[tokio::test]
    async fn counts_only_delimiters_and_handles_split_boundaries() {
        let sink = Sink::default();
        let logger = logger(sink.clone());
        let stream = futures_util::stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: a\n")),
            Ok(Bytes::from_static(b"\ndata: b\nx\n")),
        ]);
        let chunks = count_forwarded_stream(stream, logger)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(chunks.len(), 2);
        let events = sink.0.lock().expect("lock");
        let LogEvent::StreamCompleted(completed) = events.last().expect("completion") else {
            panic!("expected completion");
        };
        assert_eq!(completed.events_decoded, 1);
    }

    #[tokio::test]
    async fn newline_after_non_delimiter_does_not_count_from_carry() {
        let sink = Sink::default();
        let logger = logger(sink.clone());
        let stream = futures_util::stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: a\n")),
            Ok(Bytes::from_static(b"x\n")),
        ]);
        let _ = count_forwarded_stream(stream, logger)
            .collect::<Vec<_>>()
            .await;
        let events = sink.0.lock().expect("lock");
        let LogEvent::StreamCompleted(completed) = events.last().expect("completion") else {
            panic!("expected completion");
        };
        assert_eq!(completed.events_decoded, 0);
    }
}
