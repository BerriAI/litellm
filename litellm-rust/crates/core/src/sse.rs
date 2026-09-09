use std::convert::Infallible;

use bytes::Bytes;
use futures_util::stream::{BoxStream, poll_fn};
use futures_util::{FutureExt, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::Error;
use crate::integrations::types::Usage;
use crate::lifecycle::StreamingObserver;

pub(crate) use sse_stream::Sse as SseEvent;

pub(crate) trait SseEventObserver: Send {
    fn on_event(&mut self, event: &SseEvent) -> Result<(), Error>;
    fn usage(&self) -> Usage;
    fn projection(&self) -> Value;
    fn finished(&self) -> bool;
    fn finish(&self) -> Result<(), Error>;
}

struct Decoder {
    sender: Option<mpsc::Sender<Bytes>>,
    events: BoxStream<'static, Result<SseEvent, sse_stream::Error>>,
}

pub(crate) struct SseObserver<T> {
    observer: T,
    decoder: Option<Decoder>,
    failed: bool,
}

impl<T: SseEventObserver> SseObserver<T> {
    pub(crate) fn new(observer: T) -> Self {
        let (sender, mut receiver) = mpsc::channel::<Bytes>(1);
        let input = poll_fn(move |cx| {
            receiver
                .poll_recv(cx)
                .map(|item| item.map(Ok::<_, Infallible>))
        });
        Self {
            observer,
            decoder: Some(Decoder {
                sender: Some(sender),
                events: Box::pin(sse_stream::SseStream::from_bytes_stream(input)),
            }),
            failed: false,
        }
    }

    fn drain(&mut self) -> Result<(), Error> {
        let Some(decoder) = &mut self.decoder else {
            return Ok(());
        };
        while !self.observer.finished() {
            // The private feed must drain even when the caller's Tokio poll budget is exhausted.
            match tokio::task::unconstrained(decoder.events.next()).now_or_never() {
                Some(Some(Ok(event))) => self.observer.on_event(&event)?,
                Some(Some(Err(error))) => {
                    return Err(Error::InvalidResponse(format!(
                        "invalid SSE stream: {error}"
                    )));
                }
                Some(None) => break,
                None => {
                    assert!(decoder.sender.is_some(), "closed SSE feed must reach EOF");
                    break;
                }
            }
        }
        Ok(())
    }
}

impl<T: SseEventObserver> StreamingObserver for SseObserver<T> {
    fn observe(&mut self, bytes: &Bytes) -> Result<(), Error> {
        if self.failed {
            return Err(Error::InvalidResponse("SSE observation has failed".into()));
        }
        if self.observer.finished() || bytes.is_empty() {
            return Ok(());
        }
        let sender = self
            .decoder
            .as_ref()
            .and_then(|decoder| decoder.sender.as_ref())
            .ok_or_else(|| Error::InvalidResponse("SSE observation is closed".into()))?;
        sender
            .try_send(bytes.clone())
            .expect("SSE input drained before the next observation");
        let result = self.drain();
        self.failed = result.is_err();
        if self.failed || self.observer.finished() {
            self.decoder.take();
        }
        result
    }

    fn usage(&self) -> Usage {
        self.observer.usage()
    }

    fn projection(&self) -> Value {
        self.observer.projection()
    }

    fn finished(&self) -> bool {
        !self.failed && self.observer.finished()
    }

    fn finish(&mut self) -> Result<(), Error> {
        if self.failed {
            return Err(Error::InvalidResponse("SSE observation has failed".into()));
        }
        if let Some(decoder) = &mut self.decoder {
            decoder.sender.take();
        }
        let result = self.drain().and_then(|()| self.observer.finish());
        self.decoder.take();
        self.failed = result.is_err();
        result
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct Recorder {
        events: Vec<SseEvent>,
        require_stop: bool,
    }

    impl SseEventObserver for Recorder {
        fn on_event(&mut self, event: &SseEvent) -> Result<(), Error> {
            if event.data.as_deref() == Some("error") {
                return Err(Error::InvalidResponse("consumer rejected event".into()));
            }
            self.events.push(event.clone());
            Ok(())
        }

        fn usage(&self) -> Usage {
            Usage {
                completion_tokens: self.events.len() as u64,
                total_tokens: self.events.len() as u64,
                ..Default::default()
            }
        }

        fn projection(&self) -> Value {
            json!(
                self.events
                    .iter()
                    .map(|event| &event.data)
                    .collect::<Vec<_>>()
            )
        }

        fn finished(&self) -> bool {
            self.events
                .last()
                .is_some_and(|event| event.data.as_deref() == Some("[DONE]"))
        }

        fn finish(&self) -> Result<(), Error> {
            if self.require_stop && !self.finished() {
                return Err(Error::InvalidResponse("missing stop".into()));
            }
            Ok(())
        }
    }

    #[test]
    fn all_chunk_boundaries_preserve_events() {
        let expected = vec![
            SseEvent::default()
                .event("delta")
                .data(" hé🦀  \nsecond")
                .id("123")
                .retry(1000),
            SseEvent::default().data(""),
            SseEvent::default().data("plain"),
        ];
        for newline in ["\n", "\r\n", "\r"] {
            let wire = Bytes::from(format!(
                "\u{feff}: heartbeat{newline}event: delta{newline}id: 123{newline}retry: 1000{newline}data:  hé🦀  {newline}data: second{newline}{newline}data:{newline}{newline}data: plain{newline}{newline}"
            ));
            for split in 0..=wire.len() {
                let mut observer = SseObserver::new(Recorder::default());
                observer.observe(&wire.slice(..split)).unwrap();
                observer.observe(&Bytes::new()).unwrap();
                observer.observe(&wire.slice(split..)).unwrap();
                observer.finish().unwrap();
                assert_eq!(
                    observer.observer.events, expected,
                    "{newline:?}, split {split}"
                );
            }
            let mut observer = SseObserver::new(Recorder::default());
            for index in 0..wire.len() {
                observer.observe(&wire.slice(index..index + 1)).unwrap();
            }
            observer.finish().unwrap();
            assert_eq!(observer.observer.events, expected);
            assert_eq!(observer.usage().total_tokens, 3);
        }
    }

    #[test]
    fn partial_events_are_buffered_until_complete_and_discarded_at_eof() {
        let mut observer = SseObserver::new(Recorder::default());
        observer.observe(&Bytes::from_static(b"data: par")).unwrap();
        assert_eq!(observer.projection(), json!([]));
        observer
            .observe(&Bytes::from_static(b"tial\n\ndata: unfinished\n"))
            .unwrap();
        observer.finish().unwrap();
        assert_eq!(observer.projection(), json!(["partial"]));
        assert!(observer.observe(&Bytes::from_static(b"\n")).is_err());

        let mut observer = SseObserver::new(Recorder {
            require_stop: true,
            ..Default::default()
        });
        observer
            .observe(&Bytes::from_static(b"data: [DONE]\n"))
            .unwrap();
        assert_eq!(
            observer.finish(),
            Err(Error::InvalidResponse("missing stop".into()))
        );
        assert!(!observer.finished());
    }

    #[test]
    fn consumer_errors_and_completion_stop_later_delivery() {
        for (terminal, failure) in [("error", true), ("[DONE]", false)] {
            let mut observer = SseObserver::new(Recorder::default());
            let result = observer.observe(&Bytes::from(format!(
                "data: first\n\ndata: {terminal}\n\ndata: later\n\n"
            )));
            if failure {
                assert_eq!(
                    result,
                    Err(Error::InvalidResponse("consumer rejected event".into()))
                );
                assert_eq!(observer.projection(), json!(["first"]));
            } else {
                result.unwrap();
                assert_eq!(observer.projection(), json!(["first", "[DONE]"]));
            }
            let projection = observer.projection();
            assert_eq!(
                observer
                    .observe(&Bytes::from_static(b"data: extra\n\n"))
                    .is_err(),
                failure
            );
            assert_eq!(observer.projection(), projection);
            assert_eq!(observer.finished(), !failure);
            assert_eq!(observer.finish().is_err(), failure);
        }
    }

    #[test]
    fn parser_errors_fail_the_chunk_and_prevent_recovery() {
        for wire in [
            b"data: first\n\nunknown: value\n\n".as_slice(),
            b"data: \xff\n\n",
            b"event: first\nevent: second\ndata: value\n\n",
        ] {
            let mut observer = SseObserver::new(Recorder::default());
            assert!(matches!(
                observer.observe(&Bytes::copy_from_slice(wire)),
                Err(Error::InvalidResponse(_))
            ));
            assert_eq!(observer.projection(), json!([]));
            assert!(
                observer
                    .observe(&Bytes::from_static(b"data: [DONE]\n\n"))
                    .is_err()
            );
            assert!(!observer.finished());
            assert!(observer.finish().is_err());
        }
    }

    #[tokio::test]
    async fn repeated_observations_drain_despite_tokio_cooperative_budget() {
        let mut observer = SseObserver::new(Recorder::default());
        for _ in 0..4096 {
            observer
                .observe(&Bytes::from_static(b"data: x\n\n"))
                .unwrap();
        }
        observer.finish().unwrap();
        assert_eq!(observer.usage().total_tokens, 4096);
    }
}
