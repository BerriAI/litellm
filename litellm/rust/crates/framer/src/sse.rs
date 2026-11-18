//! Server-sent events, named as the WHATWG spec names them
//! (<https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream>).
//!
//! | Name | What it is |
//! |---|---|
//! | chunk | The bytes one read of the body returned. It can end anywhere, inside a line included. The spec has no word for it |
//! | line | Bytes up to a `\n`, a `\r\n` or a `\r` |
//! | field | A line split at its first colon into a name and a value. A line that starts with a colon is a comment |
//! | block | The lines up to and including the blank line that closes them, kept as bytes in [`SseBlock`] |
//! | event | What a block dispatches, read into [`SseEvent`]. A block of comments dispatches none |
//!
//! The spec's grammar has one word for the last two, `event = *( comment / field ) end-of-line`,
//! and its examples say block for the lines ("The following stream contains four blocks"). This
//! module keeps event for what is dispatched and block for the bytes, so that `block.event`
//! reads as what it is.
//!
//! A stream is read one of two ways. [`SseCodec`] yields each event and drops every other byte,
//! for a reader that consumes the stream. [`blocks`] keeps every byte and puts each event beside
//! the block that dispatched it, for a relay that forwards the stream and still has to see its
//! events go by.

use std::str;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures_util::{Stream, StreamExt, future, stream};
use tokio_util::codec::{Decoder, Encoder};

use crate::SseError;

const BOM: &[u8] = b"\xEF\xBB\xBF";

/// What one block dispatches: its `data` lines joined by `\n`, and the last `event`, `id` and
/// `retry` field it holds. Nothing carries over from an earlier block, so `id` is the block's own
/// and not the spec's last event ID.
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
/// carry none: a comment, a block that is not UTF-8, or the tail a stream ended or failed inside
/// of.
///
/// The tail is the one stretch the spec would not call a block, since no blank line closes it. It
/// is handed out all the same so that a relay drops no byte, and it carries no event because the
/// spec dispatches none for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseBlock {
    pub bytes: Bytes,
    pub event: Option<SseEvent>,
}

impl SseBlock {
    /// This block carrying `data` in place of its event's data, every other byte as it was read:
    /// the other fields, the comments, and the terminator each line ends with. `data` is written
    /// where the event's first `data` line was, one line per `\n` of it. A block that carries
    /// no event stays as it is.
    #[must_use]
    pub fn with_data(self, data: &str) -> Self {
        let Some(event) = self.event else {
            return self;
        };
        let (bom, fields) = self.bytes.split_at(fields_start(&self.bytes, &event));
        let mut bytes = BytesMut::with_capacity(self.bytes.len() + data.len());
        bytes.put_slice(bom);
        let mut written = false;
        for (line, terminator) in terminated(fields) {
            if split_field(line).0 != b"data" {
                bytes.put_slice(line);
                bytes.put_slice(terminator);
            } else if !std::mem::replace(&mut written, true) {
                for line in data.split('\n') {
                    bytes.put_slice(b"data: ");
                    bytes.put_slice(line.as_bytes());
                    bytes.put_slice(terminator);
                }
            }
        }
        Self {
            bytes: bytes.freeze(),
            event: Some(SseEvent {
                data: data.to_owned(),
                ..event
            }),
        }
    }
}

/// Where the fields of a block that dispatched `event` start: past the BOM, when the block
/// opened its stream with one. A BOM anywhere else is part of the field name it precedes.
fn fields_start(block: &[u8], event: &SseEvent) -> usize {
    match block.strip_prefix(BOM) {
        Some(fields) if event_in(fields).ok().flatten().as_ref() == Some(event) => BOM.len(),
        _ => 0,
    }
}

#[derive(Default)]
struct Unread {
    bytes: BytesMut,
    past_bom: bool,
}

impl Unread {
    /// The next block the bytes read so far close. Once the stream is `cut`, by its end or by a
    /// failed chunk, what is left is the tail.
    fn next_block(&mut self, cut: bool) -> Option<SseBlock> {
        let unread = self.bytes.len();
        let bom = if !self.past_bom && self.bytes.starts_with(BOM) {
            BOM.len()
        } else {
            0
        };
        let closed = block_end(&self.bytes[bom..])
            .map(|end| bom + end)
            .filter(|end| cut || *end < unread || !self.bytes.ends_with(b"\r"));
        let end = closed.or_else(|| (cut && unread > 0).then_some(unread))?;
        let bytes = self.bytes.split_to(end).freeze();
        self.past_bom = true;
        let event = closed.and_then(|_| event_in(&bytes[bom..]).ok().flatten());
        Some(SseBlock { bytes, event })
    }
}

/// `chunks` regrouped at the blank lines that close each block, with no byte added or dropped,
/// for relays that forward a provider's stream untouched but still need to see each event go by.
/// A failed chunk passes through after every byte read before it, so the block it cut comes
/// first as a tail.
pub fn blocks<S, E>(chunks: S) -> impl Stream<Item = Result<SseBlock, E>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send,
    E: Send,
{
    chunks
        .map(Some)
        .chain(stream::once(future::ready(None)))
        .scan(Unread::default(), |unread, chunk| {
            let (cut, failed) = match chunk {
                Some(Ok(bytes)) => {
                    unread.bytes.extend_from_slice(&bytes);
                    (false, None)
                }
                Some(Err(error)) => (true, Some(Err(error))),
                None => (true, None),
            };
            let closed = std::iter::from_fn(|| unread.next_block(cut)).map(Ok);
            let read: Vec<Result<SseBlock, E>> = closed.chain(failed).collect();
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

/// Each line of `bytes` and the terminator that ends it.
fn terminated(bytes: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    lines(bytes).scan(0, move |start, (line, end)| {
        let terminator = &bytes[*start + line.len()..end];
        *start = end;
        Some((line, terminator))
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
    #[case::a_bom_before_a_blank_line(
        b"\xEF\xBB\xBF\ndata: a\n\n",
        vec![(&b"\xEF\xBB\xBF\n"[..], None), (&b"data: a\n\n"[..], Some("a"))]
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

    #[rstest]
    #[case::lines_that_are_not_data_stay(
        b": note\nvendor: 1\nevent: delta\nid: 7\ndata: old\n\n",
        vec![&b": note\nvendor: 1\nevent: delta\nid: 7\ndata: new\ndata: data\n\n"[..]]
    )]
    #[case::every_data_line_gives_way_to_the_new_data_at_the_first(
        b"data: a\nretry: 5\ndata: b\n\n",
        vec![&b"data: new\ndata: data\nretry: 5\n\n"[..]]
    )]
    #[case::each_line_keeps_its_own_terminator(
        b"event: delta\r\ndata: old\r\r\n",
        vec![&b"event: delta\r\ndata: new\rdata: data\r\r\n"[..]]
    )]
    #[case::a_bom_that_opens_the_stream_stays_ahead_of_the_fields(
        b"\xEF\xBB\xBFdata: old\n\n",
        vec![&b"\xEF\xBB\xBFdata: new\ndata: data\n\n"[..]]
    )]
    #[case::a_bom_past_the_start_is_part_of_its_field_name(
        b"data: a\n\n\xEF\xBB\xBFdata: b\ndata: c\n\n",
        vec![&b"data: new\ndata: data\n\n"[..], &b"\xEF\xBB\xBFdata: b\ndata: new\ndata: data\n\n"[..]]
    )]
    #[case::a_block_that_carries_no_event_stays(
        b": keepalive\n\ndata: partial\n",
        vec![&b": keepalive\n\n"[..], &b"data: partial\n"[..]]
    )]
    #[tokio::test]
    async fn a_block_given_new_data_keeps_every_other_byte_and_carries_the_event_it_now_dispatches(
        #[case] wire: &[u8],
        #[case] expected: Vec<&[u8]>,
        #[values(1, 4096)] size: usize,
    ) {
        let renewed: Vec<Result<SseBlock, &str>> =
            read(pieces(wire, size).into_iter().map(Ok).collect())
                .await
                .into_iter()
                .map(|block| block.map(|block| block.with_data("new\ndata")))
                .collect();

        let written: Vec<Bytes> = renewed
            .iter()
            .map(|block| block.clone().unwrap().bytes)
            .collect();
        let reread = read(vec![Ok(Bytes::from(written.concat()))]).await;
        assert_eq!(
            (written, reread),
            (
                expected.into_iter().map(Bytes::copy_from_slice).collect(),
                renewed
            )
        );
    }

    #[tokio::test]
    async fn a_failed_chunk_comes_after_the_bytes_read_before_it() {
        let read = read(vec![
            Ok(Bytes::from_static(b"data: a\n\ndata: par")),
            Err("reset"),
            Ok(Bytes::from_static(b"tial\n\n")),
        ])
        .await;

        let read: Vec<Result<(Bytes, Option<String>), &str>> = read
            .into_iter()
            .map(|block| block.map(|block| (block.bytes, block.event.map(|event| event.data))))
            .collect();
        assert_eq!(
            read,
            [
                Ok((Bytes::from_static(b"data: a\n\n"), Some("a".to_owned()))),
                Ok((Bytes::from_static(b"data: par"), None)),
                Err("reset"),
                Ok((Bytes::from_static(b"tial\n\n"), None)),
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
