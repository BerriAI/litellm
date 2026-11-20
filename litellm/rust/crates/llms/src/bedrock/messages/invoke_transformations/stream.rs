use std::collections::VecDeque;

use bytes::Bytes;
use futures_util::{Stream, StreamExt, stream};
use litellm_llms_types::{
    formats::messages::StopReason, providers::bedrock::INVOCATION_METRICS_KEY,
};
use litellm_python_compat::{Value, json, repr, truthy::truthy};

use crate::{
    Error, ErrorDetail,
    base_llm::messages::streaming::{ByteStream, FrameStream},
    bedrock::common_utils::guardrail::InvokeStreamGuardrail,
};

use crate::bedrock::chat::invoke_handler::{BedrockStreamRaise, aiter_bytes, raised};

const METRICS_USAGE_KEYS: [(&str, &str); 4] = [
    ("input_tokens", "inputTokenCount"),
    ("output_tokens", "outputTokenCount"),
    ("cache_read_input_tokens", "cacheReadInputTokenCount"),
    ("cache_creation_input_tokens", "cacheWriteInputTokenCount"),
];

const CACHE_FIELDS: [&str; 2] = ["cache_creation_input_tokens", "cache_read_input_tokens"];

type Fields = Vec<(Value, Value)>;

fn key(name: &str) -> Value {
    Value::Str(name.to_string())
}

fn position(fields: &Fields, name: &str) -> Option<usize> {
    fields
        .iter()
        .position(|(key, _)| matches!(key, Value::Str(key) if key == name))
}

fn get<'a>(fields: &'a Fields, name: &str) -> Option<&'a Value> {
    position(fields, name).map(|at| &fields[at].1)
}

fn set(fields: &mut Fields, name: &str, value: Value) {
    match position(fields, name) {
        Some(at) => fields[at].1 = value,
        None => fields.push((key(name), value)),
    }
}

fn set_pair(fields: &mut Fields, name: Value, value: Value) {
    match fields.iter().position(|(key, _)| *key == name) {
        Some(at) => fields[at].1 = value,
        None => fields.push((name, value)),
    }
}

fn pop(fields: &mut Fields, name: &str) -> Option<Value> {
    position(fields, name).map(|at| fields.remove(at).1)
}

fn is_type(fields: &Fields, event_type: &str) -> bool {
    matches!(get(fields, "type"), Some(Value::Str(value)) if value == event_type)
}

fn dict_or_empty(value: Option<&Value>) -> Fields {
    match value {
        Some(Value::Dict(fields)) => fields.clone(),
        _ => Fields::new(),
    }
}

/// Port of Python's `_refused`: a `message_delta` whose reply a guardrail blocked stops on
/// `refusal`.
fn refused(mut chunk: Fields) -> Fields {
    if !is_type(&chunk, "message_delta") {
        return chunk;
    }
    let mut delta = dict_or_empty(get(&chunk, "delta"));
    set(
        &mut delta,
        "stop_reason",
        Value::Str(StopReason::Refusal.to_string()),
    );
    set(&mut chunk, "delta", Value::Dict(delta));
    chunk
}

/// `AmazonAnthropicClaudeMessagesStreamDecoder._chunk_parser`: Bedrock's invocation metrics
/// become Anthropic usage, under the chunk's own usage.
fn chunk_parser(chunk: Value) -> Result<Fields, Error> {
    let mut fields = match chunk {
        Value::Dict(fields) => fields,
        other => return Err(raised(BedrockStreamRaise::NotAnObject(dumped(&other)?))),
    };
    let Some(metrics) = pop(&mut fields, INVOCATION_METRICS_KEY).filter(truthy) else {
        return Ok(fields);
    };
    let metrics = dict_or_empty(Some(&metrics));
    let preserved = dict_or_empty(get(&fields, "usage"));
    let usage = preserved.into_iter().fold(
        METRICS_USAGE_KEYS
            .iter()
            .filter_map(|(anthropic, metric)| {
                Some((key(anthropic), get(&metrics, metric)?.clone()))
            })
            .collect::<Fields>(),
        |mut usage, (name, value)| {
            set_pair(&mut usage, name, value);
            usage
        },
    );
    set(&mut fields, "usage", Value::Dict(usage));
    Ok(fields)
}

/// `AmazonAnthropicClaudeMessagesConfig._promote_message_stop_usage`, on the decoded chunks.
#[derive(Default)]
struct UsagePromoter {
    pending_delta: Option<Fields>,
    start_usage: Option<Fields>,
}

impl UsagePromoter {
    /// Port of Python's `_merge_message_start_cache_into_delta_usage`.
    fn merge_message_start_cache_into_delta_usage(&self, delta_usage: &mut Fields) {
        let Some(start) = self.start_usage.as_ref().filter(|start| !start.is_empty()) else {
            return;
        };
        for field in CACHE_FIELDS.iter().chain(["cache_creation"].iter()) {
            if get(delta_usage, field).is_none()
                && let Some(value) = get(start, field).filter(|value| **value != Value::None)
            {
                set(delta_usage, field, value.clone());
            }
        }
    }

    fn completed(&self, mut delta: Fields, stop: Option<&Fields>) -> Fields {
        let mut delta_usage = dict_or_empty(get(&delta, "usage").filter(|usage| truthy(usage)));
        if let Some(stop) = stop {
            for field in CACHE_FIELDS {
                if let Some(value) = get(stop, field) {
                    set(&mut delta_usage, field, value.clone());
                }
            }
            if let Some(raw) = get(stop, "input_tokens").filter(|raw| **raw != Value::None) {
                let tokens = match raw {
                    Value::Int(_) | Value::Bool(_) => raw.clone(),
                    _ => Value::from(0),
                };
                set(&mut delta_usage, "input_tokens", tokens);
            }
        }
        self.merge_message_start_cache_into_delta_usage(&mut delta_usage);
        if !delta_usage.is_empty() {
            set(&mut delta, "usage", Value::Dict(delta_usage));
        }
        delta
    }

    fn feed(&mut self, chunk: Fields) -> Vec<Fields> {
        if is_type(&chunk, "message_start") {
            let message = dict_or_empty(get(&chunk, "message").filter(|message| truthy(message)));
            if let Some(Value::Dict(usage)) = get(&message, "usage") {
                self.start_usage = Some(usage.clone());
            }
            return self
                .pending_delta
                .take()
                .into_iter()
                .chain([chunk])
                .collect();
        }
        if is_type(&chunk, "message_delta") {
            self.pending_delta = Some(chunk);
            return Vec::new();
        }
        if is_type(&chunk, "message_stop")
            && let Some(delta) = self.pending_delta.take()
        {
            let stop_usage = dict_or_empty(get(&chunk, "usage").filter(|usage| truthy(usage)));
            return vec![self.completed(delta, Some(&stop_usage)), chunk];
        }
        self.pending_delta
            .take()
            .into_iter()
            .chain([chunk])
            .collect()
    }

    fn finish(&mut self) -> Option<Fields> {
        let delta = self.pending_delta.take()?;
        Some(self.completed(delta, None))
    }
}

fn dumped(value: &Value) -> Result<String, Error> {
    json::dumps(value)
        .map_err(|error| Error::InvalidResponse(ErrorDetail::failed("SSE event", error)))
}

/// `BaseAnthropicMessagesStreamingIterator._convert_chunk_to_sse_format` for a decoded chunk.
fn sse_frame(chunk: Fields) -> Result<Bytes, Error> {
    let event_type = get(&chunk, "type").map_or_else(|| "message".to_string(), repr::to_str);
    let data = dumped(&Value::Dict(chunk))?;
    Ok(Bytes::from(format!(
        "event: {event_type}\ndata: {data}\n\n"
    )))
}

/// Port of Python's `AmazonAnthropicClaudeMessagesStreamDecoder` hold: from the first
/// `message_delta` on, chunks wait in `tail` until the stream ends or fails, since Bedrock reports
/// a guardrail block only on a later event, and the held `message_delta` then stops on `refusal`.
struct Relay<S> {
    chunks: S,
    guardrail: InvokeStreamGuardrail,
    tail: Vec<Fields>,
    promoter: UsagePromoter,
    queued: VecDeque<Bytes>,
    failed: Option<Error>,
    finished: bool,
}

impl<S: Stream<Item = Result<Value, Error>> + Unpin> Relay<S> {
    fn queue(&mut self, chunks: impl IntoIterator<Item = Fields>) -> Result<(), Error> {
        for chunk in chunks {
            self.queued.push_back(sse_frame(chunk)?);
        }
        Ok(())
    }

    fn promote(&mut self, chunks: impl IntoIterator<Item = Fields>) -> Result<(), Error> {
        for chunk in chunks {
            let promoted = self.promoter.feed(chunk);
            self.queue(promoted)?;
        }
        Ok(())
    }

    fn release_tail(&mut self) -> Result<(), Error> {
        let tail = std::mem::take(&mut self.tail);
        if self.guardrail.blocked() {
            return self.promote(tail.into_iter().map(refused));
        }
        self.promote(tail)
    }

    fn relay(&mut self, chunk: Value) -> Result<(), Error> {
        self.guardrail = std::mem::take(&mut self.guardrail).with_event(&chunk);
        let parsed = chunk_parser(chunk)?;
        if self.tail.is_empty() && !is_type(&parsed, "message_delta") {
            return self.promote([parsed]);
        }
        self.tail.push(parsed);
        Ok(())
    }

    async fn advance(&mut self) -> Result<(), Error> {
        let Some(chunk) = self.chunks.next().await else {
            self.finished = true;
            self.release_tail()?;
            let flushed = self.promoter.finish();
            return self.queue(flushed);
        };
        if let Err(error) = chunk.and_then(|chunk| self.relay(chunk)) {
            self.failed = Some(error);
            return self.release_tail();
        }
        Ok(())
    }
}

/// Bedrock InvokeModel's event stream as the SSE frames Python's Bedrock Messages stream yields:
/// each decoded chunk re-serialized with `json.dumps`, with usage promoted onto `message_delta`
/// and a guardrail block reported as a `refusal` stop.
pub fn get_async_streaming_response_iterator(
    bytes: ByteStream,
    headers: Vec<(String, String)>,
) -> FrameStream {
    let relay = Relay {
        chunks: Box::pin(aiter_bytes(bytes, headers)),
        guardrail: InvokeStreamGuardrail::default(),
        tail: Vec::new(),
        promoter: UsagePromoter::default(),
        queued: VecDeque::new(),
        failed: None,
        finished: false,
    };
    stream::unfold(Some(relay), |relay| async move {
        let mut relay = relay?;
        loop {
            if let Some(frame) = relay.queued.pop_front() {
                return Some((Ok(frame), Some(relay)));
            }
            if let Some(error) = relay.failed.take() {
                return Some((Err(error), None));
            }
            if relay.finished {
                return None;
            }
            if let Err(error) = relay.advance().await {
                return Some((Err(error), None));
            }
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use base64::{Engine, engine::general_purpose::STANDARD};
    use futures_util::TryStreamExt;
    use rstest::rstest;

    use super::*;

    fn aws_frame(chunk: &str) -> Vec<u8> {
        let payload = format!(r#"{{"bytes": "{}"}}"#, STANDARD.encode(chunk));
        let message = Message::new(Bytes::from(payload))
            .add_header(Header::new(
                ":event-type",
                HeaderValue::String("chunk".into()),
            ))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ));
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        wire
    }

    fn in_pieces(chunks: &[&str]) -> ByteStream {
        let wire: Vec<u8> = chunks.iter().flat_map(|chunk| aws_frame(chunk)).collect();
        stream::iter(
            wire.chunks(7)
                .map(|piece| Ok(Bytes::copy_from_slice(piece)))
                .collect::<Vec<_>>(),
        )
        .boxed()
    }

    async fn relayed(chunks: &[&str]) -> Result<String, Error> {
        let frames: Vec<Bytes> =
            get_async_streaming_response_iterator(in_pieces(chunks), Vec::new())
                .try_collect()
                .await?;
        Ok(frames
            .iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect())
    }

    const MESSAGE_STOP: &str = "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";

    #[rstest]
    #[case::stop_sequence_null_is_kept(
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}"#,
        "event: message_delta\ndata: {\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\", \"stop_sequence\": null}, \"usage\": {\"output_tokens\": 4}}\n\n"
    )]
    #[case::an_absent_stop_sequence_stays_absent(
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        "event: message_delta\ndata: {\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\"}, \"usage\": {\"output_tokens\": 4}}\n\n"
    )]
    #[case::unknown_keys_and_non_ascii_are_dumped_as_python_does(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hé"},"extra":[1.0,2]}"#,
        "event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"text_delta\", \"text\": \"h\\u00e9\"}, \"extra\": [1.0, 2]}\n\n"
    )]
    #[tokio::test]
    async fn each_chunk_is_relayed_as_json_dumps_writes_it(
        #[case] chunk: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            relayed(&[chunk, r#"{"type":"message_stop"}"#])
                .await
                .unwrap(),
            format!("{expected}{MESSAGE_STOP}")
        );
    }

    #[tokio::test]
    async fn a_chunk_without_a_type_is_named_message() {
        assert_eq!(
            relayed(&[r#"{"message":"x"}"#, r#"{"type":"message_stop"}"#])
                .await
                .unwrap(),
            format!("event: message\ndata: {{\"message\": \"x\"}}\n\n{MESSAGE_STOP}")
        );
    }

    #[rstest]
    #[case::cache_fields_on_message_stop(
        r#"{"input_tokens": 10, "output_tokens": 0}"#,
        r#"{"output_tokens": 5}"#,
        r#"{"input_tokens": 3, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}"#,
        r#"{"output_tokens": 5, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 100, "input_tokens": 3}"#
    )]
    #[case::cache_only_on_message_start(
        r#"{"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 4}}"#,
        r#"{"output_tokens": 5}"#,
        r#"{"input_tokens": 10}"#,
        r#"{"output_tokens": 5, "input_tokens": 10, "cache_creation_input_tokens": 4, "cache_read_input_tokens": 80, "cache_creation": {"ephemeral_5m_input_tokens": 4}}"#
    )]
    #[case::a_float_input_count_is_zeroed(
        r#"{"input_tokens": 10}"#,
        r#"{"output_tokens": 5}"#,
        r#"{"input_tokens": 3.5}"#,
        r#"{"output_tokens": 5, "input_tokens": 0}"#
    )]
    #[tokio::test]
    async fn message_delta_usage_is_completed_from_stop_then_start(
        #[case] start: &str,
        #[case] delta: &str,
        #[case] stop: &str,
        #[case] expected: &str,
    ) {
        let start = format!(r#"{{"type": "message_start", "message": {{"usage": {start}}}}}"#);
        let delta = format!(r#"{{"type": "message_delta", "usage": {delta}}}"#);
        let stop = format!(r#"{{"type": "message_stop", "usage": {stop}}}"#);

        let relayed = relayed(&[&start, &delta, &stop]).await.unwrap();

        assert!(
            relayed.contains(&format!(
                "event: message_delta\ndata: {{\"type\": \"message_delta\", \"usage\": {expected}}}\n\n"
            )),
            "{relayed}"
        );
    }

    #[rstest]
    #[case::metrics_fill_missing_usage(
        r#"{"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "outputTokenCount": 9}}"#,
        r#"{"type": "message_stop", "usage": {"input_tokens": 3, "output_tokens": 9}}"#
    )]
    #[case::the_chunks_own_usage_wins_in_place(
        r#"{"type": "message_stop", "usage": {"input_tokens": 1}, "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "cacheReadInputTokenCount": 40}, "amazon-bedrock-trace": {}}"#,
        r#"{"type": "message_stop", "usage": {"input_tokens": 1, "cache_read_input_tokens": 40}, "amazon-bedrock-trace": {}}"#
    )]
    #[case::empty_metrics_are_dropped(
        r#"{"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}"#,
        r#"{"type": "message_stop"}"#
    )]
    #[tokio::test]
    async fn invocation_metrics_become_anthropic_usage(
        #[case] chunk: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            relayed(&[chunk]).await.unwrap(),
            format!("event: message_stop\ndata: {expected}\n\n")
        );
    }

    #[rstest]
    #[case::no_message_stop(
        &[r#"{"type":"message_start","message":{}}"#],
        "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {}}\n\n"
    )]
    #[case::a_held_delta_is_flushed(
        &[r#"{"type":"message_delta","usage":{"output_tokens":1}}"#],
        "event: message_delta\ndata: {\"type\": \"message_delta\", \"usage\": {\"output_tokens\": 1}}\n\n"
    )]
    #[tokio::test]
    async fn a_stream_without_message_stop_ends_with_its_own_frames(
        #[case] chunks: &[&str],
        #[case] expected: &str,
    ) {
        assert_eq!(relayed(chunks).await.unwrap(), expected);
    }

    #[tokio::test]
    async fn a_provider_error_event_is_terminal() {
        let error = r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#;

        assert_eq!(
            relayed(&[error]).await.unwrap(),
            "event: error\ndata: {\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"busy\"}}\n\n"
        );
    }

    const GUARDRAILED_EVENTS: [&str; 5] = [
        r#"{"type": "message_start", "message": {"id": "bedrock-guardrails-19a36670", "type": "message", "role": "assistant", "content": [], "model": "claude-haiku-5-5", "stop_reason": null, "usage": {"input_tokens": 0, "output_tokens": 0}}}"#,
        r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
        r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sorry, the model cannot answer this question."}}"#,
        r#"{"type": "content_block_stop", "index": 0}"#,
        r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}}"#,
    ];
    const INTERVENED: &str = r#", "amazon-bedrock-guardrailAction": "INTERVENED""#;
    const MODEL_RAN_METRICS: &str = r#", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 11, "outputTokenCount": 7, "invocationLatency": 412, "firstByteLatency": 380}"#;
    const MASKED_TRACE: &str = r#", "amazon-bedrock-trace": {"guardrail": {"input": {"6hbgpnnqoeb1": {"sensitiveInformationPolicy": {"piiEntities": [{"type": "US_SOCIAL_SECURITY_NUMBER", "match": "123-45-6789", "action": "ANONYMIZED", "detected": true}]}}}}}"#;
    const BLOCKED_AND_MASKED_TRACE: &str = r#", "amazon-bedrock-trace": {"guardrail": {"input": {"6hbgpnnqoeb1": {"wordPolicy": {"customWords": [{"match": "hexblade", "action": "BLOCKED", "detected": true}]}, "sensitiveInformationPolicy": {"piiEntities": [{"type": "US_SOCIAL_SECURITY_NUMBER", "match": "123-45-6789", "action": "ANONYMIZED", "detected": true}]}}}}}"#;

    fn sse_events(relayed: &str) -> Vec<serde_json::Value> {
        relayed
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    #[rstest]
    #[case::blocked(&[INTERVENED], "refusal")]
    #[case::masked(&[INTERVENED, MODEL_RAN_METRICS], "end_turn")]
    #[case::masked_per_trace(&[INTERVENED, MASKED_TRACE], "end_turn")]
    #[case::blocked_and_masked_per_trace(&[INTERVENED, MODEL_RAN_METRICS, BLOCKED_AND_MASKED_TRACE], "refusal")]
    #[case::guardrail_passed(&[r#", "amazon-bedrock-guardrailAction": "NONE""#, MODEL_RAN_METRICS], "end_turn")]
    #[case::no_guardrail(&[], "end_turn")]
    #[tokio::test]
    async fn a_guardrail_block_on_message_stop_turns_the_held_delta_into_a_refusal(
        #[case] guardrail_fields: &[&str],
        #[case] expected_stop_reason: &str,
    ) {
        let message_stop = format!(r#"{{"type": "message_stop"{}}}"#, guardrail_fields.concat());
        let chunks: Vec<&str> = GUARDRAILED_EVENTS
            .into_iter()
            .chain([message_stop.as_str()])
            .collect();

        let events = sse_events(&relayed(&chunks).await.unwrap());

        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "message_delta")
                .map(|event| event["delta"].clone())
                .collect::<Vec<_>>(),
            vec![serde_json::json!({"stop_reason": expected_stop_reason, "stop_sequence": null})]
        );
        assert_eq!(
            events[events.len() - 2..]
                .iter()
                .map(|event| event["type"].clone())
                .collect::<Vec<_>>(),
            vec!["message_delta", "message_stop"]
        );
    }

    #[tokio::test]
    async fn a_failing_stream_releases_the_held_tail_before_its_error() {
        let message_stop = format!(r#"{{"type": "message_stop"{INTERVENED}}}"#);
        let chunks = [GUARDRAILED_EVENTS[4], message_stop.as_str(), "[1]"];

        let items: Vec<Result<Bytes, Error>> =
            get_async_streaming_response_iterator(in_pieces(&chunks), Vec::new())
                .collect()
                .await;

        let (error, frames) = items.split_last().unwrap();
        let frames: String = frames
            .iter()
            .map(|frame| String::from_utf8(frame.as_ref().unwrap().to_vec()).unwrap())
            .collect();
        assert_eq!(
            frames,
            format!(
                "event: message_delta\ndata: {{\"type\": \"message_delta\", \"delta\": {{\"stop_reason\": \"refusal\", \"stop_sequence\": null}}}}\n\nevent: message_stop\ndata: {message_stop_dumped}\n\n",
                message_stop_dumped =
                    r#"{"type": "message_stop", "amazon-bedrock-guardrailAction": "INTERVENED"}"#
            )
        );
        assert_eq!(
            litellm_error::find::<BedrockStreamRaise>(error.as_ref().unwrap_err()),
            Some(&BedrockStreamRaise::NotAnObject("[1]".into()))
        );
    }

    #[rstest]
    #[case::a_list(r#"[1]"#, BedrockStreamRaise::NotAnObject("[1]".into()))]
    #[case::not_json(r#"{"type": "#, BedrockStreamRaise::InvalidJson(r#"{"type": "#.into()))]
    #[tokio::test]
    async fn a_payload_python_cannot_parse_raises_what_python_raises(
        #[case] chunk: &str,
        #[case] expected: BedrockStreamRaise,
    ) {
        let error = relayed(&[chunk]).await.unwrap_err();

        assert_eq!(
            litellm_error::find::<BedrockStreamRaise>(&error),
            Some(&expected)
        );
    }
}
