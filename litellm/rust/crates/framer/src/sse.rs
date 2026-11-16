use std::str;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures_util::{Stream, StreamExt, future, stream};
use tokio_util::codec::{Decoder, Encoder};

use crate::SseError;

const BOM: &[u8] = b"\xEF\xBB\xBF";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SseCodec {
    past_bom: bool,
}

impl Decoder for SseCodec {
    type Item = SseEvent;
    type Error = SseError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<SseEvent>, SseError> {
        if !self.skip_bom(src) {
            return Ok(None);
        }
        while let Some(end) = block_end(src) {
            if let Some(event) = event_in(&src.split_to(end))? {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }

    /// An event the stream ended inside of is dropped, as the SSE spec requires.
    fn decode_eof(&mut self, pending: &mut BytesMut) -> Result<Option<SseEvent>, SseError> {
        self.decode(pending)
    }
}

pub fn decode_all(wire: &[u8]) -> Result<Vec<SseEvent>, SseError> {
    let mut codec = SseCodec::default();
    let mut buffer = BytesMut::from(wire);
    let mut events = Vec::new();
    while let Some(event) = codec.decode_eof(&mut buffer)? {
        events.push(event);
    }
    Ok(events)
}

impl SseCodec {
    fn skip_bom(&mut self, src: &mut BytesMut) -> bool {
        skip_bom(&mut self.past_bom, src)
    }
}

fn event_in(block: &[u8]) -> Result<Option<SseEvent>, SseError> {
    Ok(lines(block)
        .map(|(line, _)| line)
        .take_while(|line| !line.is_empty())
        .try_fold(Pending::default(), Pending::apply)?
        .dispatch())
}

/// One stretch of a relayed stream and the event it carries. `event` is `None` for bytes that
/// carry none: a comment, a block that is not UTF-8, or the tail a stream ended inside of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseBlock {
    pub bytes: Bytes,
    pub event: Option<SseEvent>,
}

#[derive(Default)]
struct Unread {
    bytes: BytesMut,
    past_bom: bool,
}

impl Unread {
    /// The next block the bytes read so far close. Once the stream is `ended`, what is left is
    /// its last block.
    fn next_block(&mut self, ended: bool) -> Option<SseBlock> {
        let unread = self.bytes.len();
        let closed = block_end(&self.bytes)
            .filter(|end| ended || *end < unread || !self.bytes.ends_with(b"\r"));
        let end = closed.or_else(|| (ended && unread > 0).then_some(unread))?;
        let bytes = self.bytes.split_to(end).freeze();
        let block = if self.past_bom {
            &bytes[..]
        } else {
            bytes.strip_prefix(BOM).unwrap_or(&bytes)
        };
        self.past_bom = true;
        let event = closed.and_then(|_| event_in(block).ok().flatten());
        Some(SseBlock { bytes, event })
    }
}

/// `chunks` regrouped at the blank lines that close each block, with no byte added or dropped,
/// for relays that forward a provider's stream untouched but still need to see each event go by.
/// A failed chunk passes through where it arrived.
pub fn blocks<S, E>(chunks: S) -> impl Stream<Item = Result<SseBlock, E>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send,
    E: Send,
{
    chunks
        .map(Some)
        .chain(stream::once(future::ready(None)))
        .scan(Unread::default(), |unread, chunk| {
            let ended = chunk.is_none();
            let failed = match chunk {
                Some(Ok(bytes)) => {
                    unread.bytes.extend_from_slice(&bytes);
                    None
                }
                Some(Err(error)) => Some(Err(error)),
                None => None,
            };
            let closed = std::iter::from_fn(|| unread.next_block(ended)).map(Ok);
            let read: Vec<Result<SseBlock, E>> = failed.into_iter().chain(closed).collect();
            future::ready(Some(stream::iter(read)))
        })
        .flatten()
}

fn skip_bom(past_bom: &mut bool, src: &mut BytesMut) -> bool {
    if *past_bom {
        return true;
    }
    if src.starts_with(BOM) {
        src.advance(BOM.len());
    } else if BOM.starts_with(src) {
        return false;
    }
    *past_bom = true;
    true
}

fn block_end(bytes: &[u8]) -> Option<usize> {
    lines(bytes)
        .find(|(line, _)| line.is_empty())
        .map(|(_, end)| end)
}

fn lines(bytes: &[u8]) -> impl Iterator<Item = (&[u8], usize)> {
    let mut cursor: usize = 0;
    std::iter::from_fn(move || {
        let rest = &bytes[cursor..];
        let end = rest.iter().position(|byte| matches!(byte, b'\n' | b'\r'))?;
        cursor += end + terminator_len(&rest[end..]);
        Some((&rest[..end], cursor))
    })
}

fn terminator_len(terminated: &[u8]) -> usize {
    match terminated {
        [b'\r', b'\n', ..] => 2,
        _ => 1,
    }
}

#[derive(Default)]
struct Pending {
    event: Option<String>,
    data: Option<String>,
    id: Option<String>,
    retry: Option<u64>,
}

impl Pending {
    fn apply(self, line: &[u8]) -> Result<Self, SseError> {
        let (name, value) = split_field(line);
        Ok(match name {
            b"event" => Self {
                event: Some(str::from_utf8(value)?.to_owned()),
                ..self
            },
            b"data" => Self {
                data: Some(append_data(self.data, str::from_utf8(value)?)),
                ..self
            },
            b"id" if !value.contains(&0) => Self {
                id: Some(str::from_utf8(value)?.to_owned()),
                ..self
            },
            b"retry" => Self {
                retry: parse_retry(value).or(self.retry),
                ..self
            },
            _ => self,
        })
    }

    fn dispatch(self) -> Option<SseEvent> {
        Some(SseEvent {
            event: self.event,
            data: self.data?,
            id: self.id,
            retry: self.retry,
        })
    }
}

fn split_field(line: &[u8]) -> (&[u8], &[u8]) {
    let Some(colon) = line.iter().position(|byte| *byte == b':') else {
        return (line, &[]);
    };
    let value = &line[colon + 1..];
    (&line[..colon], value.strip_prefix(b" ").unwrap_or(value))
}

fn append_data(buffer: Option<String>, line: &str) -> String {
    match buffer {
        Some(existing) => format!("{existing}\n{line}"),
        None => line.to_owned(),
    }
}

fn parse_retry(value: &[u8]) -> Option<u64> {
    if !value.iter().all(u8::is_ascii_digit) {
        return None;
    }
    str::from_utf8(value).ok()?.parse().ok()
}

impl Encoder<SseEvent> for SseCodec {
    type Error = SseError;

    fn encode(&mut self, event: SseEvent, dst: &mut BytesMut) -> Result<(), SseError> {
        if let Some(name) = event.event {
            dst.put_slice(format!("event: {name}\n").as_bytes());
        }
        for line in event.data.split('\n') {
            dst.put_slice(format!("data: {line}\n").as_bytes());
        }
        if let Some(id) = event.id {
            dst.put_slice(format!("id: {id}\n").as_bytes());
        }
        if let Some(retry) = event.retry {
            dst.put_slice(format!("retry: {retry}\n").as_bytes());
        }
        dst.put_u8(b'\n');
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};
    use rstest::rstest;

    use super::*;

    fn pieces(wire: &[u8], size: usize) -> Vec<Bytes> {
        wire.chunks(size).map(Bytes::copy_from_slice).collect()
    }

    async fn read(chunks: Vec<Result<Bytes, &'static str>>) -> Vec<Result<SseBlock, &'static str>> {
        blocks(stream::iter(chunks)).collect().await
    }

    /// Each block of `wire` as its bytes and the data of the event it carries.
    async fn split(wire: &[u8], size: usize) -> Vec<(Bytes, Option<String>)> {
        read(pieces(wire, size).into_iter().map(Ok).collect())
            .await
            .into_iter()
            .map(|block| {
                let block = block.unwrap();
                (block.bytes, block.event.map(|event| event.data))
            })
            .collect()
    }

    #[rstest]
    #[case::lf(
        b": keepalive\n\nevent: delta\ndata: a\ndata: b\n\n",
        vec![(&b": keepalive\n\n"[..], None), (&b"event: delta\ndata: a\ndata: b\n\n"[..], Some("a\nb"))]
    )]
    #[case::crlf(
        b"data: a\r\n\r\ndata: b\r\n\r\n",
        vec![(&b"data: a\r\n\r\n"[..], Some("a")), (&b"data: b\r\n\r\n"[..], Some("b"))]
    )]
    #[case::cr(
        b"data: a\r\rdata: b\r\r",
        vec![(&b"data: a\r\r"[..], Some("a")), (&b"data: b\r\r"[..], Some("b"))]
    )]
    #[case::bom(
        b"\xEF\xBB\xBFdata: \xe2\x82\xac\n\n",
        vec![(&b"\xEF\xBB\xBFdata: \xe2\x82\xac\n\n"[..], Some("\u{20ac}"))]
    )]
    #[case::a_bom_past_the_start(
        b"data: a\n\n\xEF\xBB\xBFdata: b\n\n",
        vec![(&b"data: a\n\n"[..], Some("a")), (&b"\xEF\xBB\xBFdata: b\n\n"[..], None)]
    )]
    #[case::not_utf8(
        b"data: \xff\n\ndata: b\n\n",
        vec![(&b"data: \xff\n\n"[..], None), (&b"data: b\n\n"[..], Some("b"))]
    )]
    #[case::a_cut_tail(
        b"data: a\n\ndata: partial\n",
        vec![(&b"data: a\n\n"[..], Some("a")), (&b"data: partial\n"[..], None)]
    )]
    #[case::nothing(b"", vec![])]
    #[tokio::test]
    async fn blocks_keep_every_byte_and_carry_the_event_each_dispatches(
        #[case] wire: &[u8],
        #[case] expected: Vec<(&[u8], Option<&str>)>,
        #[values(1, 3, 4096)] size: usize,
    ) {
        let expected: Vec<(Bytes, Option<String>)> = expected
            .into_iter()
            .map(|(bytes, data)| (Bytes::copy_from_slice(bytes), data.map(str::to_owned)))
            .collect();

        assert_eq!(split(wire, size).await, expected);
    }

    #[tokio::test]
    async fn a_failed_chunk_passes_through_where_it_arrived() {
        let read = read(vec![
            Ok(Bytes::from_static(b"data: a\n\ndata: par")),
            Err("reset"),
            Ok(Bytes::from_static(b"tial\n\n")),
        ])
        .await;

        let read: Vec<Result<Bytes, &str>> = read
            .into_iter()
            .map(|block| block.map(|block| block.bytes))
            .collect();
        assert_eq!(
            read,
            [
                Ok(Bytes::from_static(b"data: a\n\n")),
                Err("reset"),
                Ok(Bytes::from_static(b"data: partial\n\n")),
            ]
        );
    }

    #[rstest]
    fn bulk_decode_preserves_every_field() {
        let events = decode_all(b"id: 1\nretry: 20\nevent: delta\ndata: a\ndata: b\n\n").unwrap();
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("delta".into()),
                data: "a\nb".into(),
                id: Some("1".into()),
                retry: Some(20),
            }]
        );
    }

    #[rstest]
    #[case::only_a_tail(b"data: partial\n", vec![])]
    #[case::tail_after_an_event(b"data: complete\n\ndata: partial\n", vec!["complete"])]
    #[case::no_terminator_at_all(b"data: partial", vec![])]
    fn bulk_decode_drops_the_event_a_stream_ends_inside(
        #[case] wire: &[u8],
        #[case] expected: Vec<&str>,
    ) {
        let data: Vec<String> = decode_all(wire)
            .unwrap()
            .into_iter()
            .map(|e| e.data)
            .collect();
        assert_eq!(data, expected);
    }

    #[rstest]
    #[case::trailing_blank_lines(b"data: x\n\n\n\n")]
    #[case::trailing_comment(b"data: x\n\n: bye\n\n")]
    #[case::lone_cr_at_eof(b"data: x\r\r")]
    fn bulk_decode_accepts_a_stream_that_ends_between_events(#[case] wire: &[u8]) {
        assert_eq!(decode_all(wire).unwrap().len(), 1);
    }

    #[rstest]
    fn bulk_decode_rejects_invalid_utf8() {
        assert!(matches!(
            decode_all(b"data: \xff\n\n"),
            Err(SseError::InvalidUtf8(_))
        ));
    }
}
