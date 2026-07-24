use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_types::event_stream::HeaderValue;
use base64::Engine;
use bytes::BytesMut;
use serde_json::{Value, json};

use crate::error::{CoreError, CoreResult};

#[derive(Clone, Debug, PartialEq)]
pub struct BedrockMessageEvent {
    pub event_type: String,
    pub chunk: Value,
}

pub struct BedrockEventStreamDecoder {
    buffer: BytesMut,
    decoder: MessageFrameDecoder,
}

impl BedrockEventStreamDecoder {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::new(),
            decoder: MessageFrameDecoder::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> CoreResult<Vec<BedrockMessageEvent>> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            let frame = self
                .decoder
                .decode_frame(&mut self.buffer)
                .map_err(|error| {
                    CoreError::InvalidResponse(format!(
                        "invalid Bedrock event stream frame: {error}"
                    ))
                })?;
            let DecodedFrame::Complete(message) = frame else {
                break;
            };
            let message_type = message.headers().iter().find_map(|header| {
                (header.name().as_str() == ":message-type").then(|| match header.value() {
                    HeaderValue::String(value) => value.as_str().to_string(),
                    _ => String::new(),
                })
            });
            if matches!(message_type.as_deref(), Some("error" | "exception")) {
                let exception_type = message.headers().iter().find_map(|header| {
                    (header.name().as_str() == ":exception-type").then(|| match header.value() {
                        HeaderValue::String(value) => value.as_str().to_string(),
                        _ => "unknown".to_string(),
                    })
                });
                let payload = String::from_utf8_lossy(message.payload());
                let payload = payload.chars().take(512).collect::<String>();
                return Err(CoreError::InvalidResponse(format!(
                    "Bedrock event stream {message_type:?} ({}){}",
                    exception_type.as_deref().unwrap_or("unknown"),
                    if payload.is_empty() {
                        String::new()
                    } else {
                        format!(": {payload}")
                    }
                )));
            }
            let envelope: Value = serde_json::from_slice(message.payload()).map_err(|error| {
                CoreError::InvalidResponse(format!("invalid Bedrock event payload: {error}"))
            })?;
            let encoded = envelope
                .get("bytes")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CoreError::InvalidResponse("Bedrock chunk has no bytes".to_string())
                })?;
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| {
                    CoreError::InvalidResponse(format!("invalid Bedrock chunk bytes: {error}"))
                })?;
            let mut chunk: Value = serde_json::from_slice(&decoded).map_err(|error| {
                CoreError::InvalidResponse(format!("invalid Anthropic stream chunk: {error}"))
            })?;
            let event_type = chunk
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CoreError::InvalidResponse(
                        "Bedrock Anthropic chunk has no event type".to_string(),
                    )
                })?
                .to_string();
            if let Some(metrics) = chunk
                .as_object_mut()
                .and_then(|object| object.remove("amazon-bedrock-invocationMetrics"))
            {
                let object = chunk.as_object_mut().ok_or_else(|| {
                    CoreError::InvalidResponse(
                        "Bedrock Anthropic chunk must be an object".to_string(),
                    )
                })?;
                let usage = object.entry("usage").or_insert_with(|| json!({}));
                if !usage.is_object() {
                    *usage = json!({});
                }
                let usage = usage.as_object_mut().ok_or_else(|| {
                    CoreError::InvalidResponse("Anthropic usage must be an object".to_string())
                })?;
                if let Some(input) = metrics.get("inputTokenCount") {
                    usage.insert("input_tokens".to_string(), input.clone());
                }
                if let Some(output) = metrics.get("outputTokenCount") {
                    usage.insert("output_tokens".to_string(), output.clone());
                }
            }
            events.push(BedrockMessageEvent { event_type, chunk });
        }
        Ok(events)
    }
}

impl Default for BedrockEventStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn serialize_sse(event: &BedrockMessageEvent) -> CoreResult<Vec<u8>> {
    let data = serde_json::to_string(&event.chunk)
        .map_err(|error| CoreError::InvalidResponse(format!("invalid stream chunk: {error}")))?;
    Ok(format!("event: {}\ndata: {data}\n\n", event.event_type).into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, Message};
    use bytes::BytesMut;

    fn frame(payload: Value, error: bool) -> Vec<u8> {
        let mut message = Message::new(serde_json::to_vec(&payload).expect("payload"));
        message = message.add_header(Header::new(
            ":event-type",
            HeaderValue::String("chunk".into()),
        ));
        if error {
            message = message.add_header(Header::new(
                ":message-type",
                HeaderValue::String("error".into()),
            ));
        }
        let mut bytes = BytesMut::new();
        write_message_to(&message, &mut bytes).expect("frame");
        bytes.to_vec()
    }

    #[test]
    fn decoder_handles_every_split_boundary_and_metrics() {
        let chunk = serde_json::json!({
            "type": "content_block_delta",
            "usage": {"cache_read_input_tokens": 3},
            "amazon-bedrock-invocationMetrics": {
                "outputTokenCount": 7
            }
        });
        let encoded = base64::engine::general_purpose::STANDARD
            .encode(serde_json::to_vec(&chunk).expect("chunk"));
        let bytes = frame(serde_json::json!({"bytes": encoded}), false);
        for split in 1..bytes.len() {
            let mut decoder = BedrockEventStreamDecoder::new();
            let mut events = Vec::new();
            for part in bytes[..split].chunks(1) {
                events.extend(decoder.push(part).expect("partial frame"));
            }
            events.extend(decoder.push(&bytes[split..]).expect("final frame"));
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].event_type, "content_block_delta");
            assert!(
                events[0]
                    .chunk
                    .get("amazon-bedrock-invocationMetrics")
                    .is_none()
            );
            assert_eq!(events[0].chunk["usage"]["cache_read_input_tokens"], 3);
            assert!(events[0].chunk["usage"].get("input_tokens").is_none());
            assert_eq!(events[0].chunk["usage"]["output_tokens"], 7);
        }
    }

    #[test]
    fn decoder_surfaces_error_frames() {
        let error = frame(serde_json::json!({"message": "bad"}), true);
        let result = BedrockEventStreamDecoder::new().push(&error);
        assert!(matches!(result, Err(CoreError::InvalidResponse(_))));
    }

    #[test]
    fn decoder_surfaces_exception_details() {
        let mut message =
            Message::new(serde_json::to_vec(&json!({"message": "throttled"})).expect("payload"));
        message = message
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("exception".into()),
            ))
            .add_header(Header::new(
                ":exception-type",
                HeaderValue::String("ThrottlingException".into()),
            ));
        let mut bytes = BytesMut::new();
        write_message_to(&message, &mut bytes).expect("frame");
        let result = BedrockEventStreamDecoder::new().push(&bytes);
        match result {
            Err(CoreError::InvalidResponse(message)) => {
                assert!(message.contains("ThrottlingException"));
                assert!(message.contains("throttled"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }
}
