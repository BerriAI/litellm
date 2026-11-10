use std::{collections::VecDeque, sync::Arc};

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, future::BoxFuture, stream::BoxStream};
use serde_json::Value;

use super::{Error, MessagesBody, MessagesCallResponse};

const SAFEGUARD_RESULTS: &str = "safeguard_results";

pub trait Safeguards: Send + Sync {
    fn results<'a>(
        &'a self,
        response: &'a MessagesBody,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>>;

    fn open_stream(&self) -> BoxFuture<'_, Result<bool, Error>>;

    fn frames(&self, chunk: Bytes) -> BoxFuture<'_, Result<Vec<Bytes>, Error>>;

    fn flush(&self) -> BoxFuture<'_, Result<Vec<Bytes>, Error>>;
}

pub(super) async fn stamped(
    safeguards: Option<&Arc<dyn Safeguards>>,
    output: MessagesCallResponse,
) -> Result<MessagesCallResponse, Error> {
    let Some(safeguards) = safeguards else {
        return Ok(output);
    };
    match output {
        MessagesCallResponse::Complete(response) => {
            let results = safeguards.results(&response).await?;
            Ok(MessagesCallResponse::Complete(match results {
                Some(results) => with_results(response, results),
                None => response,
            }))
        }
        MessagesCallResponse::Stream { head, chunks } => {
            if !safeguards.open_stream().await? {
                return Ok(MessagesCallResponse::Stream { head, chunks });
            }
            Ok(MessagesCallResponse::Stream {
                head,
                chunks: stamped_frames(Arc::clone(safeguards), chunks),
            })
        }
    }
}

fn with_results(response: MessagesBody, results: Value) -> MessagesBody {
    match response {
        MessagesBody::Message(mut message) => {
            message.extra.insert(SAFEGUARD_RESULTS.to_owned(), results);
            MessagesBody::Message(message)
        }
        MessagesBody::Unchecked(mut body) => {
            body.insert(SAFEGUARD_RESULTS.to_owned(), results);
            MessagesBody::Unchecked(body)
        }
    }
}

struct Framing {
    chunks: Option<BoxStream<'static, Result<Bytes, Error>>>,
    safeguards: Arc<dyn Safeguards>,
    ready: VecDeque<Bytes>,
}

fn stamped_frames(
    safeguards: Arc<dyn Safeguards>,
    chunks: BoxStream<'static, Result<Bytes, Error>>,
) -> BoxStream<'static, Result<Bytes, Error>> {
    let framing = Framing {
        chunks: Some(chunks),
        safeguards,
        ready: VecDeque::new(),
    };
    futures_util::stream::try_unfold(framing, |mut framing| async move {
        loop {
            if let Some(frame) = framing.ready.pop_front() {
                return Ok(Some((frame, framing)));
            }
            let Some(mut chunks) = framing.chunks.take() else {
                return Ok(None);
            };
            let frames = match chunks.try_next().await? {
                Some(chunk) => {
                    framing.chunks = Some(chunks);
                    framing.safeguards.frames(chunk).await?
                }
                None => framing.safeguards.flush().await?,
            };
            framing.ready.extend(frames);
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::{Map, json};

    use super::*;
    use crate::route::MessagesStreamHead;

    struct Scripted {
        results: Option<Value>,
        streams: bool,
        fed: Mutex<Vec<Bytes>>,
    }

    impl Scripted {
        fn new(results: Option<Value>, streams: bool) -> Arc<Self> {
            Arc::new(Self {
                results,
                streams,
                fed: Mutex::new(Vec::new()),
            })
        }
    }

    impl Safeguards for Scripted {
        fn results<'a>(
            &'a self,
            _: &'a MessagesBody,
        ) -> BoxFuture<'a, Result<Option<Value>, Error>> {
            Box::pin(async move { Ok(self.results.clone()) })
        }

        fn open_stream(&self) -> BoxFuture<'_, Result<bool, Error>> {
            Box::pin(async move { Ok(self.streams) })
        }

        fn frames(&self, chunk: Bytes) -> BoxFuture<'_, Result<Vec<Bytes>, Error>> {
            Box::pin(async move {
                self.fed.lock().unwrap().push(chunk.clone());
                Ok(chunk
                    .split_inclusive(|byte| *byte == b';')
                    .map(Bytes::copy_from_slice)
                    .collect())
            })
        }

        fn flush(&self) -> BoxFuture<'_, Result<Vec<Bytes>, Error>> {
            Box::pin(async move { Ok(vec![Bytes::from_static(b"tail")]) })
        }
    }

    fn message() -> MessagesBody {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [], "stop_reason": "end_turn", "stop_sequence": null,
        }))
        .unwrap();
        MessagesBody::Message(Box::new(response))
    }

    fn stream(chunks: &[&'static [u8]]) -> MessagesCallResponse {
        let chunks: Vec<Result<Bytes, Error>> = chunks
            .iter()
            .map(|chunk| Ok(Bytes::from_static(chunk)))
            .collect();
        MessagesCallResponse::Stream {
            head: MessagesStreamHead {
                headers: Vec::new(),
            },
            chunks: futures_util::stream::iter(chunks).boxed(),
        }
    }

    async fn collected(output: MessagesCallResponse) -> Vec<Bytes> {
        let MessagesCallResponse::Stream { chunks, .. } = output else {
            panic!("expected a stream");
        };
        chunks.try_collect().await.unwrap()
    }

    #[rstest]
    #[case::message_gains_the_results(message(), Some(json!([{"type": "dangerous_tool_use"}])))]
    #[case::unchecked_gains_the_results(
        MessagesBody::Unchecked(Map::new()),
        Some(json!([{"type": "dangerous_tool_use"}]))
    )]
    #[case::declined_leaves_it(message(), None)]
    #[tokio::test]
    async fn a_complete_response_carries_what_the_classifier_answered(
        #[case] response: MessagesBody,
        #[case] results: Option<Value>,
    ) {
        let safeguards: Arc<dyn Safeguards> = Scripted::new(results.clone(), false);
        let MessagesCallResponse::Complete(stamped) = stamped(
            Some(&safeguards),
            MessagesCallResponse::Complete(response.clone()),
        )
        .await
        .unwrap() else {
            panic!("expected a complete response");
        };
        let body = serde_json::to_value(&stamped).unwrap();
        assert_eq!(body.get(SAFEGUARD_RESULTS), results.as_ref());
        let mut unstamped = body;
        unstamped.as_object_mut().unwrap().remove(SAFEGUARD_RESULTS);
        assert_eq!(unstamped, serde_json::to_value(&response).unwrap());
    }

    #[tokio::test]
    async fn a_stream_is_reframed_through_the_classifier_and_flushed_at_its_end() {
        let scripted = Scripted::new(None, true);
        let safeguards: Arc<dyn Safeguards> = scripted.clone();
        let frames = collected(
            stamped(Some(&safeguards), stream(&[b"a;b", b";c;"]))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            frames,
            ["a;", "b", ";", "c;", "tail"]
                .map(|frame| Bytes::from_static(frame.as_bytes()))
                .to_vec()
        );
        assert_eq!(scripted.fed.lock().unwrap().len(), 2);
    }

    #[rstest]
    #[case::declined(Some(Scripted::new(None, false) as Arc<dyn Safeguards>))]
    #[case::not_classified(None)]
    #[tokio::test]
    async fn a_stream_the_classifier_does_not_answer_passes_through(
        #[case] safeguards: Option<Arc<dyn Safeguards>>,
    ) {
        let frames = collected(
            stamped(safeguards.as_ref(), stream(&[b"a;b", b";c;"]))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            frames,
            [b"a;b".as_slice(), b";c;"].map(Bytes::from_static).to_vec()
        );
    }
}
