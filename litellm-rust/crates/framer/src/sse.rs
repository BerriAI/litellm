use std::str;

use bytes::{Buf, BufMut, BytesMut};
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
            let block = src.split_to(end);
            let pending = lines(&block)
                .map(|(line, _)| line)
                .take_while(|line| !line.is_empty())
                .try_fold(Pending::default(), Pending::apply)?;
            if let Some(event) = pending.dispatch() {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }

    fn decode_eof(&mut self, _pending: &mut BytesMut) -> Result<Option<SseEvent>, SseError> {
        Ok(None)
    }
}

impl SseCodec {
    fn skip_bom(&mut self, src: &mut BytesMut) -> bool {
        if self.past_bom {
            return true;
        }
        if src.starts_with(BOM) {
            src.advance(BOM.len());
        } else if BOM.starts_with(src) {
            return false;
        }
        self.past_bom = true;
        true
    }
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
