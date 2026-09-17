use bytes::Buf;
use futures_util::{Stream, StreamExt};

use crate::{Error, MAX_FRAME_BYTES};

const BOM: &[u8] = b"\xEF\xBB\xBF";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseFrame {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

pub fn frames<S, B, E>(input: S) -> impl Stream<Item = Result<SseFrame, Error>> + Send
where
    S: Stream<Item = Result<B, E>> + Send,
    B: Buf + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    futures_util::stream::try_unfold(
        (Box::pin(input), None::<B>, Parser::default()),
        |(mut input, mut chunk, mut parser)| async move {
            loop {
                if let Some(current) = chunk.as_mut()
                    && current.has_remaining()
                {
                    let (consumed, frame) = parser.scan(current.chunk())?;
                    current.advance(consumed);
                    if let Some(frame) = frame {
                        return Ok(Some((frame, (input, chunk, parser))));
                    }
                    continue;
                }
                if parser.pending_bytes > MAX_FRAME_BYTES {
                    return Err(Error::FrameTooLarge);
                }
                match input.next().await {
                    Some(Ok(next)) => chunk = Some(next),
                    Some(Err(error)) => return Err(Error::Body(Box::new(error))),
                    None => return Ok(None),
                }
            }
        },
    )
}

#[derive(Default)]
struct Parser {
    tail: Vec<u8>,
    fields: Fields,
    pending_bytes: usize,
    past_first_line: bool,
    skip_lf: bool,
}

impl Parser {
    fn scan(&mut self, slice: &[u8]) -> Result<(usize, Option<SseFrame>), Error> {
        let skip = usize::from(std::mem::take(&mut self.skip_lf) && slice.first() == Some(&b'\n'));
        let mut rest = &slice[skip..];
        while let Some(end) = rest.iter().position(|byte| matches!(byte, b'\n' | b'\r')) {
            let terminator = match (rest[end], rest.get(end + 1)) {
                (b'\r', Some(b'\n')) => 2,
                (b'\r', None) => {
                    self.skip_lf = true;
                    1
                }
                _ => 1,
            };
            let frame = self.line(&rest[..end])?;
            rest = &rest[end + terminator..];
            if frame.is_some() {
                return Ok((slice.len() - rest.len(), frame));
            }
        }
        self.tail.extend_from_slice(rest);
        self.pending_bytes += rest.len();
        Ok((slice.len(), None))
    }

    fn line(&mut self, line: &[u8]) -> Result<Option<SseFrame>, Error> {
        self.pending_bytes += line.len();
        if self.tail.is_empty() {
            return self.complete_line(line);
        }
        let mut joined = std::mem::take(&mut self.tail);
        joined.extend_from_slice(line);
        let frame = self.complete_line(&joined);
        joined.clear();
        self.tail = joined;
        frame
    }

    fn complete_line(&mut self, line: &[u8]) -> Result<Option<SseFrame>, Error> {
        let line = if std::mem::replace(&mut self.past_first_line, true) {
            line
        } else {
            line.strip_prefix(BOM).unwrap_or(line)
        };
        if line.is_empty() {
            self.pending_bytes = 0;
            return Ok(std::mem::take(&mut self.fields).dispatch());
        }
        let (name, value) = line
            .iter()
            .position(|byte| *byte == b':')
            .map_or((line, &b""[..]), |colon| {
                (&line[..colon], &line[colon + 1..])
            });
        let value = value.strip_prefix(b" ").unwrap_or(value);
        self.fields.with(name, value)?;
        Ok(None)
    }
}

#[derive(Default)]
struct Fields {
    event: Option<String>,
    data: Option<String>,
    id: Option<String>,
    retry: Option<u64>,
}

impl Fields {
    fn with(&mut self, name: &[u8], value: &[u8]) -> Result<(), Error> {
        let text = || std::str::from_utf8(value).map_err(Error::InvalidUtf8);
        match name {
            b"data" => match &mut self.data {
                Some(data) => {
                    data.push('\n');
                    data.push_str(text()?);
                }
                None => self.data = Some(text()?.to_owned()),
            },
            b"event" => self.event = Some(text()?.to_owned()),
            b"id" if !value.contains(&0) => self.id = Some(text()?.to_owned()),
            b"retry" if !value.is_empty() && value.iter().all(u8::is_ascii_digit) => {
                self.retry = text()?.parse().ok().or(self.retry);
            }
            _ => {}
        }
        Ok(())
    }

    fn dispatch(self) -> Option<SseFrame> {
        self.data.map(|data| SseFrame {
            event: self.event,
            data,
            id: self.id,
            retry: self.retry,
        })
    }
}
