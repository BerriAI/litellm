use bytes::Bytes;
use litellm_types::llms::anthropic_messages::anthropic_response::{
    AnthropicMessagesResponse, AnthropicResponseContentBlock, AnthropicUsage, KnownContentBlock,
};
use serde::Serialize;
use serde_json::{Value, json};
use strum::IntoStaticStr;

use super::streaming_iterator::AnthropicContentBlockDelta;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StopDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, IntoStaticStr)]
#[serde(tag = "type", rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum FakeStreamEvent {
    MessageStart {
        message: Box<AnthropicMessagesResponse>,
    },
    ContentBlockStart {
        index: usize,
        content_block: AnthropicResponseContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: StopDelta,
        usage: AnthropicUsage,
    },
    MessageStop,
}

impl FakeStreamEvent {
    pub fn sse_frame(&self) -> Bytes {
        let name: &'static str = self.into();
        let data = serde_json::to_string(self).expect("stream events serialize to JSON");
        Bytes::from(format!("event: {name}\ndata: {data}\n\n"))
    }
}

pub fn fake_anthropic_messages_stream(
    message: &AnthropicMessagesResponse,
) -> impl Iterator<Item = FakeStreamEvent> + '_ {
    let usage = message.usage.clone().unwrap_or_default();
    let start = FakeStreamEvent::MessageStart {
        message: Box::new(AnthropicMessagesResponse {
            id: message.id.clone(),
            message_type: message.message_type.clone(),
            role: message.role.clone(),
            model: message.model.clone(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: Some(AnthropicUsage {
                input_tokens: Some(usage.input_tokens.unwrap_or(0)),
                output_tokens: Some(0),
                extra: usage.extra,
            }),
            container: message.container.clone(),
            extra: message.extra.clone(),
        }),
    };
    let end = [
        FakeStreamEvent::MessageDelta {
            delta: StopDelta {
                stop_reason: message.stop_reason.clone(),
                stop_sequence: message.stop_sequence.clone(),
            },
            usage: message.usage.clone().unwrap_or(AnthropicUsage {
                output_tokens: Some(0),
                ..AnthropicUsage::default()
            }),
        },
        FakeStreamEvent::MessageStop,
    ];
    std::iter::once(start)
        .chain(
            message
                .content
                .iter()
                .enumerate()
                .flat_map(|(index, block)| block_events(index, block)),
        )
        .chain(end)
}

fn block_events(
    index: usize,
    block: &AnthropicResponseContentBlock,
) -> impl Iterator<Item = FakeStreamEvent> {
    let (content_block, deltas) = match block {
        AnthropicResponseContentBlock::Known(KnownContentBlock::Text { text, extra }) => (
            AnthropicResponseContentBlock::Known(KnownContentBlock::Text {
                text: String::new(),
                extra: extra.clone(),
            }),
            vec![AnthropicContentBlockDelta::TextDelta { text: text.clone() }],
        ),
        AnthropicResponseContentBlock::Known(KnownContentBlock::Thinking {
            thinking,
            signature,
            extra,
        }) => (
            AnthropicResponseContentBlock::Known(KnownContentBlock::Thinking {
                thinking: String::new(),
                signature: None,
                extra: extra.clone(),
            }),
            std::iter::once(AnthropicContentBlockDelta::ThinkingDelta {
                thinking: thinking.clone(),
            })
            .chain(match signature {
                Some(Value::String(signature)) if !signature.is_empty() => {
                    Some(AnthropicContentBlockDelta::SignatureDelta {
                        signature: signature.clone(),
                    })
                }
                _ => None,
            })
            .collect(),
        ),
        AnthropicResponseContentBlock::Known(KnownContentBlock::ToolUse { input, extra }) => (
            AnthropicResponseContentBlock::Known(KnownContentBlock::ToolUse {
                input: json!({}),
                extra: extra.clone(),
            }),
            vec![input_delta(input)],
        ),
        AnthropicResponseContentBlock::Known(KnownContentBlock::ServerToolUse { input, extra }) => {
            (
                AnthropicResponseContentBlock::Known(KnownContentBlock::ServerToolUse {
                    input: json!({}),
                    extra: extra.clone(),
                }),
                vec![input_delta(input)],
            )
        }
        AnthropicResponseContentBlock::Known(KnownContentBlock::RedactedThinking { .. })
        | AnthropicResponseContentBlock::Other(_) => (block.clone(), Vec::new()),
    };
    std::iter::once(FakeStreamEvent::ContentBlockStart {
        index,
        content_block,
    })
    .chain(
        deltas
            .into_iter()
            .map(move |delta| FakeStreamEvent::ContentBlockDelta { index, delta }),
    )
    .chain([FakeStreamEvent::ContentBlockStop { index }])
}

fn input_delta(input: &Value) -> AnthropicContentBlockDelta {
    AnthropicContentBlockDelta::InputJsonDelta {
        partial_json: input.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::*;

    fn message(
        content: Vec<AnthropicResponseContentBlock>,
        usage: Option<AnthropicUsage>,
    ) -> AnthropicMessagesResponse {
        AnthropicMessagesResponse {
            id: "msg_1".to_string(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude".to_string(),
            content,
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage,
            container: None,
            extra: Map::new(),
        }
    }

    fn usage(input: u64, output: u64, extra: Value) -> AnthropicUsage {
        AnthropicUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            extra: fields(extra),
        }
    }

    fn fields(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("expected a JSON object, got {other}"),
        }
    }

    fn known(block: KnownContentBlock) -> AnthropicResponseContentBlock {
        AnthropicResponseContentBlock::Known(block)
    }

    fn text(text: &str, extra: Value) -> AnthropicResponseContentBlock {
        known(KnownContentBlock::Text {
            text: text.to_string(),
            extra: fields(extra),
        })
    }

    fn thinking(signature: Option<Value>) -> AnthropicResponseContentBlock {
        known(KnownContentBlock::Thinking {
            thinking: "hm".to_string(),
            signature,
            extra: Map::new(),
        })
    }

    fn events(message: &AnthropicMessagesResponse) -> Vec<Value> {
        fake_anthropic_messages_stream(message)
            .map(|event| serde_json::to_value(event).unwrap())
            .collect()
    }

    #[rstest]
    #[case::with_usage(
        Some(usage(12, 8, json!({"cache_read_input_tokens": 7}))),
        json!({"input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 7}),
    )]
    #[case::without_usage(None, json!({"input_tokens": 0, "output_tokens": 0}))]
    fn message_start_zeroes_output_tokens(
        #[case] usage: Option<AnthropicUsage>,
        #[case] expected: Value,
    ) {
        let all = events(&message(vec![], usage));
        assert_eq!(all[0]["message"]["usage"], expected);
        assert_eq!(all[0]["message"]["content"], json!([]));
        assert_eq!(all[0]["message"]["stop_reason"], Value::Null);
    }

    #[test]
    fn message_delta_reports_response_usage_or_zero_output() {
        let with = events(&message(vec![], Some(usage(1, 2, json!({})))));
        assert_eq!(
            with[1]["usage"],
            json!({"input_tokens": 1, "output_tokens": 2})
        );
        let without = events(&message(vec![], None));
        assert_eq!(without[1]["usage"], json!({"output_tokens": 0}));
        assert_eq!(
            without[1]["delta"],
            json!({"stop_reason": "end_turn", "stop_sequence": null})
        );
    }

    #[test]
    fn message_stop_is_bare() {
        let all = events(&message(vec![], None));
        assert_eq!(all.last(), Some(&json!({"type": "message_stop"})));
    }

    #[test]
    fn thinking_start_strips_the_signature() {
        let all = events(&message(vec![thinking(Some(json!("sig")))], None));
        assert_eq!(
            all[1]["content_block"],
            json!({"type": "thinking", "thinking": ""})
        );
    }

    #[rstest]
    #[case::missing(None, 1)]
    #[case::empty(Some(json!("")), 1)]
    #[case::non_string(Some(json!(42)), 1)]
    #[case::valid(Some(json!("signed")), 2)]
    fn signature_delta_only_for_a_nonempty_string(
        #[case] signature: Option<Value>,
        #[case] deltas: usize,
    ) {
        let all = events(&message(vec![thinking(signature)], None));
        let count = all
            .iter()
            .filter(|event| event["type"] == "content_block_delta")
            .count();
        assert_eq!(count, deltas);
    }

    #[test]
    fn tool_use_streams_its_input_as_compact_json() {
        let block = known(KnownContentBlock::ServerToolUse {
            input: json!({"q": "x"}),
            extra: fields(json!({"id": "s1", "name": "web_search"})),
        });
        let all = events(&message(vec![block], None));
        assert_eq!(all[1]["content_block"]["input"], json!({}));
        assert_eq!(
            all[2]["delta"],
            json!({"type": "input_json_delta", "partial_json": "{\"q\":\"x\"}"})
        );
    }

    #[test]
    fn text_keeps_extras_and_streams_one_delta() {
        let all = events(&message(
            vec![text("hi", json!({"citations": [{"a": 1}]}))],
            None,
        ));
        assert_eq!(
            all[1]["content_block"],
            json!({"type": "text", "text": "", "citations": [{"a": 1}]})
        );
        assert_eq!(all[2]["delta"], json!({"type": "text_delta", "text": "hi"}));
    }

    #[rstest]
    #[case::redacted_thinking(known(KnownContentBlock::RedactedThinking {
        extra: fields(json!({"data": "opaque"})),
    }))]
    #[case::other(AnthropicResponseContentBlock::Other(json!({"type": "web_search_tool_result", "content": []})))]
    #[case::non_object(AnthropicResponseContentBlock::Other(json!("stray")))]
    fn blocks_without_a_delta_pass_through_verbatim(#[case] block: AnthropicResponseContentBlock) {
        let all = events(&message(vec![block.clone()], None));
        assert_eq!(
            all[1]["content_block"],
            serde_json::to_value(&block).unwrap()
        );
        assert_eq!(all[2]["type"], "content_block_stop");
    }

    #[test]
    fn sse_frame_event_name_matches_the_data_type() {
        let all = message(
            vec![
                text("hi", json!({})),
                known(KnownContentBlock::ToolUse {
                    input: json!({}),
                    extra: Map::new(),
                }),
            ],
            None,
        );
        for event in fake_anthropic_messages_stream(&all) {
            let frame = String::from_utf8(event.sse_frame().to_vec()).unwrap();
            let (head, data) = frame.split_once("\ndata: ").unwrap();
            let data: Value = serde_json::from_str(data.trim_end()).unwrap();
            assert_eq!(head, format!("event: {}", data["type"].as_str().unwrap()));
        }
    }
}
