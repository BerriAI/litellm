use std::io;

use bytes::Buf;
use futures_util::{Stream, StreamExt, TryStreamExt};
use tokio_util::{
    codec::{Decoder, FramedRead},
    io::StreamReader,
};

/// `input` decoded by `codec` into its frames. Frame is tokio-util's word for one whole unit of a
/// codec's format, whatever the format is. With `SseCodec` a frame is an `SseEvent`, and with
/// `AwsEventStreamCodec` it is a `Message`. SSE itself has no frames, so nothing in `sse` is
/// named one.
pub fn frames<S, B, E, D>(
    input: S,
    codec: D,
) -> impl Stream<Item = Result<D::Item, D::Error>> + Send
where
    S: Stream<Item = Result<B, E>> + Send,
    B: Buf + Send,
    E: std::error::Error + Send + Sync + 'static,
    D: Decoder + Send,
{
    FramedRead::new(StreamReader::new(input.map_err(io::Error::other)), codec).fuse()
}
