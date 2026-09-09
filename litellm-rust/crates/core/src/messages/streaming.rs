use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::Error;
use crate::integrations::types::Usage;
use crate::sse::{SseEvent, SseEventObserver};

#[derive(Default)]
pub(super) struct AnthropicMessagesObserver {
    message: Value,
    inputs: BTreeMap<usize, String>,
    stopped: bool,
}

impl SseEventObserver for AnthropicMessagesObserver {
    fn on_event(&mut self, event: &SseEvent) -> Result<(), Error> {
        let Some(data) = event.data.as_deref() else {
            return Ok(());
        };
        if data.is_empty() {
            return Ok(());
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|_| Error::InvalidResponse("invalid Messages SSE JSON".into()))?;
        match value["type"].as_str() {
            Some("message_start") => {
                if !self.message.is_null() {
                    return Err(Error::InvalidResponse(
                        "duplicate Messages message_start".into(),
                    ));
                }
                self.message = value["message"].clone();
                if !self.message.is_object() {
                    return Err(Error::InvalidResponse(
                        "missing Messages message_start body".into(),
                    ));
                }
                self.message["stream"] = json!(true);
            }
            Some("content_block_start") => {
                let index = Self::index(&value)?;
                let content = self.message["content"]
                    .as_array_mut()
                    .ok_or_else(|| Error::InvalidResponse("content before message_start".into()))?;
                if index != content.len() {
                    return Err(Error::InvalidResponse(
                        "invalid Messages content block index".into(),
                    ));
                }
                if !value["content_block"].is_object() {
                    return Err(Error::InvalidResponse(
                        "invalid Messages content block".into(),
                    ));
                }
                content.push(value["content_block"].clone());
            }
            Some("content_block_delta") => {
                let index = Self::index(&value)?;
                let block = self.message["content"]
                    .get_mut(index)
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| Error::InvalidResponse("delta without content block".into()))?;
                let delta = &value["delta"];
                match delta["type"].as_str() {
                    Some("input_json_delta") => {
                        self.inputs
                            .entry(index)
                            .or_default()
                            .push_str(delta["partial_json"].as_str().unwrap_or_default());
                    }
                    Some("text_delta" | "thinking_delta" | "signature_delta") => {
                        let field = match delta["type"].as_str() {
                            Some("text_delta") => "text",
                            Some("thinking_delta") => "thinking",
                            _ => "signature",
                        };
                        let text = block
                            .entry(field)
                            .or_insert_with(|| Value::String(String::new()));
                        let Value::String(text) = text else {
                            return Err(Error::InvalidResponse(
                                "invalid Messages text field".into(),
                            ));
                        };
                        let delta = delta[field].as_str().ok_or_else(|| {
                            Error::InvalidResponse("missing Messages text delta".into())
                        })?;
                        text.push_str(delta);
                    }
                    Some("citations_delta") => {
                        if block.get("citations").is_none() {
                            block.insert("citations".into(), json!([]));
                        }
                        if let Some(citations) =
                            block.get_mut("citations").and_then(Value::as_array_mut)
                        {
                            citations.push(delta["citation"].clone());
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let index = Self::index(&value)?;
                if let Some(input) = self.inputs.remove(&index) {
                    let input = serde_json::from_str(&input).map_err(|_| {
                        Error::InvalidResponse("incomplete Messages tool input JSON".into())
                    })?;
                    let block = self.message["content"]
                        .get_mut(index)
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| {
                            Error::InvalidResponse("stop without content block".into())
                        })?;
                    block.insert("input".into(), input);
                }
            }
            Some("message_delta") => {
                let message = self
                    .message
                    .as_object_mut()
                    .ok_or_else(|| Error::InvalidResponse("delta before message_start".into()))?;
                if let Some(delta) = value["delta"].as_object() {
                    message.extend(delta.clone());
                }
                if let Some(usage) = value["usage"].as_object() {
                    let merged = message.entry("usage").or_insert_with(|| json!({}));
                    if let Some(merged) = merged.as_object_mut() {
                        merged.extend(usage.clone());
                    }
                }
            }
            Some("message_stop") => {
                if !self.message.is_object() {
                    return Err(Error::InvalidResponse("stop before message_start".into()));
                }
                if !self.inputs.is_empty() {
                    return Err(Error::InvalidResponse(
                        "unclosed Messages tool input".into(),
                    ));
                }
                self.stopped = true;
            }
            Some("error") => {
                return Err(Error::InvalidResponse(format!(
                    "Messages stream error: {}",
                    value["error"]
                )));
            }
            _ => {}
        }
        Ok(())
    }

    fn usage(&self) -> Usage {
        let usage = &self.message["usage"];
        let prompt_tokens = usage["input_tokens"]
            .as_u64()
            .unwrap_or_default()
            .saturating_add(
                usage["cache_creation_input_tokens"]
                    .as_u64()
                    .unwrap_or_default(),
            )
            .saturating_add(
                usage["cache_read_input_tokens"]
                    .as_u64()
                    .unwrap_or_default(),
            );
        let completion_tokens = usage["output_tokens"].as_u64().unwrap_or_default();
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }

    fn projection(&self) -> Value {
        self.message.clone()
    }
    fn finished(&self) -> bool {
        self.stopped
    }
    fn finish(&self) -> Result<(), Error> {
        if self.stopped {
            Ok(())
        } else {
            Err(Error::InvalidResponse(
                "Provider stream ended before emitting a message_stop event".into(),
            ))
        }
    }
}

impl AnthropicMessagesObserver {
    fn index(value: &Value) -> Result<usize, Error> {
        value["index"]
            .as_u64()
            .and_then(|index| index.try_into().ok())
            .ok_or_else(|| Error::InvalidResponse("missing Messages content block index".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::StreamingObserver;
    use crate::sse::SseObserver;
    use bytes::Bytes;

    #[test]
    fn fragmented_events_reconstruct_content_and_cache_usage() {
        let events = [
            json!({"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"test","content":[],"usage":{"input_tokens":2,"cache_read_input_tokens":10,"cache_creation_input_tokens":5}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"héllo"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"reason"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sig"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"weather","input":{}}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}),
            json!({"type":"content_block_stop","index":2}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}),
            json!({"type":"message_stop"}),
        ];
        for separator in ["\n", "\r\n", "\r"] {
            let mut observer = SseObserver::new(AnthropicMessagesObserver::default());
            for event in &events {
                let encoded = format!(
                    "event: {}{separator}data: {event}{separator}{separator}",
                    event["type"].as_str().unwrap()
                );
                for byte in encoded.as_bytes() {
                    observer.observe(&Bytes::copy_from_slice(&[*byte])).unwrap();
                }
            }
            observer.finish().unwrap();
            let result = observer.projection();
            assert_eq!(result["content"][0]["text"], "héllo");
            assert_eq!(
                result["content"][1],
                json!({"type":"thinking","thinking":"reason","signature":"sig"})
            );
            assert_eq!(result["content"][2]["input"], json!({"city":"Paris"}));
            assert_eq!(result["stop_reason"], "tool_use");
            assert_eq!(result["usage"]["cache_read_input_tokens"], 10);
            assert_eq!(
                observer.usage(),
                Usage {
                    prompt_tokens: 17,
                    completion_tokens: 8,
                    total_tokens: 25
                }
            );
        }
    }

    #[test]
    fn truncated_and_provider_error_streams_fail() {
        let mut observer = AnthropicMessagesObserver::default();
        observe_events(&mut observer, &[json!({"type":"ping"})]).unwrap();
        assert!(observer.finish().is_err());
        assert!(
            observe_events(
                &mut observer,
                &[json!({"type":"error","error":{"type":"overloaded_error"}})]
            )
            .is_err()
        );
        assert!(!observer.finished());
    }

    fn observe_events(
        observer: &mut AnthropicMessagesObserver,
        events: &[Value],
    ) -> Result<(), Error> {
        events
            .iter()
            .try_for_each(|event| observer.on_event(&SseEvent::default().data(event.to_string())))
    }

    #[test]
    fn empty_events_are_ignored_and_json_type_controls_dispatch() {
        let mut observer = AnthropicMessagesObserver::default();
        observer
            .on_event(&SseEvent::default().event("message_stop"))
            .unwrap();
        observer.on_event(&SseEvent::default().data("")).unwrap();
        assert!(observer.projection().is_null());
        assert!(!observer.finished());
        observer
            .on_event(
                &SseEvent::default()
                    .event("message_stop")
                    .data(json!({"type":"message_start","message":{"content":[]}}).to_string()),
            )
            .unwrap();
        assert!(!observer.finished());
        observe_events(&mut observer, &[json!({"type":"message_stop"})]).unwrap();
        observer.finish().unwrap();
    }

    #[test]
    fn every_chunk_boundary_preserves_multiline_sse_and_utf8() {
        for newline in ["\n", "\r\n", "\r"] {
            let wire = format!(
                ": heartbeat{newline}data: {{\"type\":\"message_start\",{newline}data: \"message\":{{\"content\":[],\"id\":\"hé🦀\"}}}}{newline}{newline}data: {{\"type\":\"message_stop\"}}{newline}{newline}"
            );
            for split in 0..=wire.len() {
                let mut observer = SseObserver::new(AnthropicMessagesObserver::default());
                observer
                    .observe(&Bytes::copy_from_slice(&wire.as_bytes()[..split]))
                    .unwrap();
                observer
                    .observe(&Bytes::copy_from_slice(&wire.as_bytes()[split..]))
                    .unwrap();
                observer.finish().unwrap();
                assert_eq!(observer.projection()["id"], "hé🦀");
            }
        }
    }

    #[test]
    fn malformed_blocks_are_errors_instead_of_panics() {
        for block in [json!(3), json!([]), json!("bad"), Value::Null] {
            let mut observer = AnthropicMessagesObserver::default();
            let result = observe_events(
                &mut observer,
                &[
                    json!({"type":"message_start","message":{"content":[]}}),
                    json!({"type":"content_block_start","index":0,"content_block":block}),
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}),
                ],
            );
            assert!(result.is_err());
            assert!(observer.finish().is_err());
        }
    }

    #[test]
    fn stop_without_start_and_duplicate_start_fail() {
        for events in [
            vec![json!({"type":"message_stop"})],
            vec![
                json!({"type":"message_start","message":{"content":[]}}),
                json!({"type":"message_start","message":{"content":[]}}),
            ],
        ] {
            let mut observer = AnthropicMessagesObserver::default();
            assert!(observe_events(&mut observer, &events).is_err());
        }
    }

    #[test]
    fn terminal_state_is_independent_of_chunking() {
        for terminal in [
            json!({"type":"message_stop"}),
            json!({"type":"error","error":"first"}),
        ] {
            let start = json!({"type":"message_start","message":{"id":"original","content":[]}});
            let trailing =
                json!({"type":"message_start","message":{"id":"overwritten","content":[]}});
            for coalesced in [false, true] {
                let mut observer = SseObserver::new(AnthropicMessagesObserver::default());
                observer
                    .observe(&Bytes::from(format!("data: {start}\n\n")))
                    .unwrap();
                let result = if coalesced {
                    observer.observe(&Bytes::from(format!(
                        "data: {terminal}\n\ndata: {trailing}\n\n"
                    )))
                } else {
                    let result = observer.observe(&Bytes::from(format!("data: {terminal}\n\n")));
                    let _ = observer.observe(&Bytes::from(format!("data: {trailing}\n\n")));
                    result
                };
                assert_eq!(observer.projection()["id"], "original");
                assert_eq!(observer.finished(), terminal["type"] == "message_stop");
                assert_eq!(result.is_err(), terminal["type"] == "error");
                assert_eq!(observer.finish().is_err(), terminal["type"] == "error");
            }
        }
    }

    #[test]
    fn incomplete_and_invalid_tool_json_fail() {
        for partial in ["{", "not json"] {
            let mut observer = AnthropicMessagesObserver::default();
            let result = observe_events(
                &mut observer,
                &[
                    json!({"type":"message_start","message":{"content":[]}}),
                    json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","input":{}}}),
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":partial}}),
                    json!({"type":"content_block_stop","index":0}),
                    json!({"type":"message_stop"}),
                ],
            );
            assert!(result.is_err());
            assert!(!observer.finished());
        }
    }

    #[test]
    fn large_text_reconstruction_preserves_all_deltas() {
        let mut observer = AnthropicMessagesObserver::default();
        observe_events(
            &mut observer,
            &[
                json!({"type":"message_start","message":{"content":[]}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"prefix"}}),
            ],
        ).unwrap();
        for _ in 0..4096 {
            observe_events(
                &mut observer,
                &[
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hé🦀"}}),
                ],
            ).unwrap();
        }
        observe_events(
            &mut observer,
            &[
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"message_stop"}),
            ],
        )
        .unwrap();
        observer.finish().unwrap();
        assert_eq!(
            observer.projection()["content"][0]["text"],
            format!("prefix{}", "hé🦀".repeat(4096))
        );
    }
    #[test]
    fn oversized_usage_cannot_panic_or_wrap_during_terminal_delivery() {
        let mut observer = AnthropicMessagesObserver::default();
        observe_events(
            &mut observer,
            &[
                json!({"type":"message_start","message":{"content":[],"usage":{"input_tokens":u64::MAX,"cache_read_input_tokens":1,"cache_creation_input_tokens":1,"output_tokens":1}}}),
                json!({"type":"message_stop"}),
            ],
        ).unwrap();
        observer.finish().unwrap();
        assert_eq!(
            observer.usage(),
            Usage {
                prompt_tokens: u64::MAX,
                completion_tokens: 1,
                total_tokens: u64::MAX
            }
        );
    }

    #[test]
    fn signature_delta_can_initialize_an_absent_signature() {
        let mut observer = AnthropicMessagesObserver::default();
        observe_events(
            &mut observer,
            &[
                json!({"type":"message_start","message":{"content":[]}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}),
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"message_stop"}),
            ],
        ).unwrap();
        observer.finish().unwrap();
        assert_eq!(observer.projection()["content"][0]["signature"], "sig");
    }

    #[test]
    #[ignore = "manual streaming throughput benchmark"]
    fn streaming_hot_path_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        for count in [4096, 16384] {
            let start = concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
            );
            let delta = format!(
                "data: {}\n\n",
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x".repeat(128)}})
            );
            let coalesced = delta.repeat(count);
            let large_event = format!(
                "data: {}\n\n",
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x".repeat(count)}})
            );
            for mode in ["deltas", "coalesced", "fragmented"] {
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let mut observer = SseObserver::new(AnthropicMessagesObserver::default());
                    observer
                        .observe(&Bytes::from_static(start.as_bytes()))
                        .unwrap();
                    let began = Instant::now();
                    match mode {
                        "deltas" => {
                            for _ in 0..count {
                                observer
                                    .observe(&Bytes::copy_from_slice(black_box(delta.as_bytes())))
                                    .unwrap();
                            }
                        }
                        "coalesced" => observer
                            .observe(&Bytes::copy_from_slice(black_box(coalesced.as_bytes())))
                            .unwrap(),
                        _ => {
                            for byte in large_event.as_bytes() {
                                observer
                                    .observe(&Bytes::copy_from_slice(black_box(
                                        std::slice::from_ref(byte),
                                    )))
                                    .unwrap();
                            }
                        }
                    }
                    samples.push(began.elapsed());
                    black_box(observer.projection());
                }
                samples.sort();
                eprintln!("{mode} count={count}: {:?}", samples[2]);
            }
        }
    }
}
