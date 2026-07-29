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
    S: Stream<Item = Result<Bytes, E>>,
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
                                && ((*index > 0 && bytes[*index - 1] == b'\n') || trailing_newline)
                        })
                        .count();
                    let trailing_newline = bytes.last().copied() == Some(b'\n');
                    logger.stream_chunk_observed(bytes.len(), events);
                    Some((Ok(bytes), (stream, logger, trailing_newline, failed)))
                }
                Some(Err(error)) => {
                    logger.failure(
                        None,
                        "stream_error",
                        "provider stream failed".to_string(),
                        None,
                    );
                    Some((Err(error), (stream, logger, trailing_newline, true)))
                }
            }
        },
    )
}
