use std::str;

use bytes::{Buf, BufMut, Bytes, BytesMut};
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

/// One event together with the exact wire bytes that carried it, for relays that forward a
/// provider's bytes untouched but still need to see each event go by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseBlock {
    pub bytes: Bytes,
    pub event: SseEvent,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SseBlockCodec {
    past_bom: bool,
}

impl Decoder for SseBlockCodec {
    type Item = SseBlock;
    type Error = SseError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<SseBlock>, SseError> {
        self.decode_block(src, false)
    }

    fn decode_eof(&mut self, pending: &mut BytesMut) -> Result<Option<SseBlock>, SseError> {
        self.decode_block(pending, true)
    }
}

impl SseBlockCodec {
    fn decode_block(
        &mut self,
        src: &mut BytesMut,
        eof: bool,
    ) -> Result<Option<SseBlock>, SseError> {
        if !skip_bom(&mut self.past_bom, src) {
            return Ok(None);
        }
        while let Some(end) = block_end(src) {
            if !eof && end == src.len() && src.ends_with(b"\r") {
                return Ok(None);
            }
            let block = src.split_to(end);
            if let Some(event) = event_in(&block)? {
                return Ok(Some(SseBlock {
                    bytes: block.freeze(),
                    event,
                }));
            }
        }
        Ok(None)
    }
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
    use futures_util::{StreamExt, TryStreamExt, stream};
    use rstest::rstest;

    use super::*;
    use crate::frames;

    fn pieces(wire: &[u8], size: usize) -> Vec<Bytes> {
        wire.chunks(size).map(Bytes::copy_from_slice).collect()
    }

    async fn blocks(wire: &[u8], size: usize) -> Result<Vec<SseBlock>, SseError> {
        frames(
            stream::iter(pieces(wire, size).into_iter().map(Ok::<_, std::io::Error>)),
            SseBlockCodec::default(),
        )
        .try_collect()
        .await
    }

    #[rstest]
    #[case::lf(
        b": keepalive\n\nevent: delta\ndata: a\ndata: b\n\n",
        b"event: delta\ndata: a\ndata: b\n\n"
    )]
    #[case::crlf(
        b": keepalive\r\n\r\nevent: delta\r\ndata: a\r\ndata: b\r\n\r\n",
        b"event: delta\r\ndata: a\r\ndata: b\r\n\r\n"
    )]
    #[case::cr(
        b": keepalive\r\revent: delta\rdata: a\rdata: b\r\r",
        b"event: delta\rdata: a\rdata: b\r\r"
    )]
    #[case::bom(b"\xEF\xBB\xBFdata: \xe2\x82\xac\n\n", b"data: \xe2\x82\xac\n\n")]
    #[case::empty_data(b"event: ignored\n\ndata:\n\n", b"data:\n\n")]
    #[case::extensions(
        b"id: 1\nretry: 10\nvendor: x\ndata: a\n\n",
        b"id: 1\nretry: 10\nvendor: x\ndata: a\n\n"
    )]
    #[tokio::test]
    async fn blocks_carry_the_wire_bytes_and_the_event_they_dispatch(
        #[case] wire: &[u8],
        #[case] expected: &[u8],
        #[values(1, 3, 4096)] size: usize,
    ) {
        let blocks = blocks(wire, size).await.unwrap();
        let expected_events = decode_all(expected).unwrap();
        assert_eq!(expected_events.len(), 1);
        assert_eq!(
            blocks,
            vec![SseBlock {
                bytes: Bytes::copy_from_slice(expected),
                event: expected_events[0].clone(),
            }]
        );
        assert_eq!(expected_events, decode_all(wire).unwrap());
    }

    #[rstest]
    #[case::empty(b"", vec![])]
    #[case::unterminated(b"data: partial\n", vec![])]
    #[case::partial_tail(b"data: complete\n\ndata: partial\n", vec![&b"data: complete\n\n"[..]])]
    #[case::multiple(b"data: first\n\ndata: second\n\n", vec![&b"data: first\n\n"[..], &b"data: second\n\n"[..]])]
    #[case::trailing_comment(b"data: first\n\n: bye\n\n", vec![&b"data: first\n\n"[..]])]
    #[tokio::test]
    async fn blocks_dispatch_complete_events_and_drop_a_cut_tail(
        #[case] wire: &[u8],
        #[case] expected: Vec<&[u8]>,
    ) {
        let mut blocks = Box::pin(frames(
            stream::iter(pieces(wire, 1).into_iter().map(Ok::<_, std::io::Error>)),
            SseBlockCodec::default(),
        ));
        for expected in expected {
            assert_eq!(blocks.next().await.unwrap().unwrap().bytes, expected);
        }
        assert!(blocks.next().await.is_none());
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

    #[rstest]
    #[tokio::test]
    async fn block_framing_preserves_transport_errors() {
        let input = stream::iter([
            Ok(Bytes::from_static(b"data: first\n\n")),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "reset",
            )),
        ]);
        let mut blocks = Box::pin(frames(input, SseBlockCodec::default()));
        assert_eq!(
            blocks.next().await.unwrap().unwrap().bytes,
            Bytes::from_static(b"data: first\n\n")
        );
        let Some(Err(SseError::Body(error))) = blocks.next().await else {
            panic!("expected body error");
        };
        let cause = error
            .get_ref()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::ConnectionReset);
        assert!(blocks.next().await.is_none());
    }

    #[rstest]
    #[tokio::test]
    async fn invalid_utf8_stops_block_framing_after_prior_events() {
        let mut blocks = Box::pin(frames(
            stream::iter(
                pieces(b"data: first\n\ndata: \xff\n\n", 1)
                    .into_iter()
                    .map(Ok::<_, std::io::Error>),
            ),
            SseBlockCodec::default(),
        ));
        assert_eq!(
            blocks.next().await.unwrap().unwrap().bytes,
            Bytes::from_static(b"data: first\n\n")
        );
        assert!(matches!(
            blocks.next().await,
            Some(Err(SseError::InvalidUtf8(_)))
        ));
        assert!(blocks.next().await.is_none());
    }
}
