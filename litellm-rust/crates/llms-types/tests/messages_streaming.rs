use litellm_llms_types::formats::messages::{
    ContentBlock,
    streaming::{MessagesContentBlock, MessagesStreamEvent},
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::start(json!({
    "type": "message_start",
    "message": {
        "id": "msg", "type": "message", "role": "assistant", "model": "test-model",
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "future_usage": {"count": 9}},
        "future_message": [1, 2]
    }
}))]
#[case::block(json!({
    "type": "content_block_start", "index": 0,
    "content_block": {"type": "future", "payload": {"keep": true}}
}))]
#[case::delta(json!({
    "type": "message_delta", "delta": {"stop_reason": "end_turn", "future_delta": 42},
    "usage": {"output_tokens": 7, "future_usage": true}
}))]
#[case::error(json!({
    "type": "error", "error": {"type": "future_error", "message": "failed", "future": 42}
}))]
fn stream_events_preserve_extensible_fields(#[case] wire: Value) {
    let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(event).unwrap(), wire);
}

#[rstest]
#[case::invalid_index(json!({"type": "content_block_stop", "index": "zero"}))]
#[case::missing_delta(json!({"type": "content_block_delta", "index": 0}))]
#[case::unknown_event(json!({"type": "future_event"}))]
fn malformed_or_unrecognized_events_remain_rejected(#[case] wire: Value) {
    assert!(serde_json::from_value::<MessagesStreamEvent>(wire).is_err());
}

#[rstest]
#[case::text(json!({"type":"content_block_delta","index":0,"future_event":null,"delta":{"type":"text_delta","text":"hi","future_delta":[1,null]}}))]
#[case::json(json!({"type":"content_block_delta","index":0,"future_event":true,"delta":{"type":"input_json_delta","partial_json":"{","future_delta":true}}))]
#[case::citations(json!({"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"doc","document_title":null,"start_page_number":1,"end_page_number":2},"future_delta":true}}))]
#[case::thinking(json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning","future_delta":true}}))]
#[case::signature(json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed","future_delta":true}}))]
#[case::compaction(json!({"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"summary","future_delta":true}}))]
#[case::block_start(json!({"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null},"future_event":true}))]
#[case::block_stop(json!({"type":"content_block_stop","index":0,"future_event":true}))]
#[case::message_delta(json!({"type":"message_delta","delta":{"stop_details":null,"safeguard_results":[{"future":null}]},"context_management":{"applied_edits":[]},"usage":{"iterations":[{"type":"message","input_tokens":2,"output_tokens":1}]},"future_event":true}))]
#[case::message_stop(json!({"type":"message_stop","future_event":true}))]
#[case::ping(json!({"type":"ping","future_event":true}))]
#[case::error(json!({"type":"error","error":{"type":"future","message":"error"},"future_event":true}))]
fn stream_events_and_deltas_preserve_extensions(#[case] wire: Value) {
    let parsed: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::text(json!({"type":"text_delta","text":17}))]
#[case::thinking(json!({"type":"thinking_delta","thinking":null}))]
#[case::future(json!({"type":"future_delta","payload":true}))]
fn malformed_and_unknown_deltas_remain_rejected(#[case] delta: Value) {
    assert!(
        serde_json::from_value::<MessagesStreamEvent>(
            json!({"type":"content_block_delta","index":0,"delta":delta})
        )
        .is_err()
    );
}

#[rstest]
#[case::wrong_known_field(json!({"type":"text","text":17}), false)]
#[case::missing_type(json!({"text":"hi"}), false)]
#[case::opaque_tool_id(json!({"type":"future","tool_use_id":17}), true)]
#[case::opaque_cache_control(json!({"type":"future","cache_control":17}), true)]
#[case::nullable_fields(json!({"type":"tool_use","id":null,"name":null,"input":null,"text":null}), true)]
fn stream_block_validation_keeps_its_existing_boundary(
    #[case] block: Value,
    #[case] accepted: bool,
) {
    let wire = json!({"type":"content_block_start","index":0,"content_block":block});
    let parsed = serde_json::from_value::<MessagesStreamEvent>(wire.clone());
    assert_eq!(parsed.is_ok(), accepted);
    if let Ok(parsed) = parsed {
        assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
    }
}

#[rstest]
#[case::nullable_text_fields(json!({
    "type":"text","text":null,"thinking":null,"signature":null,"data":null,
    "id":null,"name":null
}), true)]
#[case::nested_content(json!({
    "type":"document",
    "source":{"type":"content","content":[{"type":"text","text":"document","future":null}]},
    "content":[{"type":"tool_reference","tool_name":"lookup"},17],
    "citations":[{"type":"char_location","cited_text":"document","start_char_index":0,"end_char_index":8}],
    "caller":{"type":"code_execution_20250825","tool_id":"tool_1"}
}), true)]
#[case::tool_fields(json!({
    "type":"tool_use","input":{"query":"q","nested":{"content":null}},
    "provider_specific_fields":{"signature":"sig"},"is_error":false,
    "file_id":"file_1","title":"title","context":"context","tool_name":"lookup",
    "url":"https://example.com","page_age":"recent","encrypted_content":"ciphertext",
    "snippet":"snippet","prompt_cache_breakpoint":{"mode":"explicit"}
}), true)]
#[case::execution_fields(json!({
    "type":"code_execution_result","stdout":"output","stderr":"error","return_code":-1,
    "encrypted_stdout":"ciphertext","error_code":"failed","error_message":"failure",
    "retrieved_at":"timestamp","server_name":"server",
    "tool_references":[{"type":"tool_reference","tool_name":"lookup"},null],
    "file_type":"text","num_lines":2,"start_line":1,"total_lines":2,"is_file_update":true,
    "lines":["before","after"],"new_lines":1,"new_start":2,"old_lines":1,"old_start":2
}), true)]
#[case::opaque_nested_fields(json!({
    "type":"future","content":17,"input":false,"source":{"type":"future","data":null},
    "citations":false,"caller":{"type":"future"},"is_error":"future","num_lines":-1,
    "lines":[17],"tool_references":"future"
}), true)]
#[case::wrong_text(json!({"type":"text","text":17}), false)]
#[case::wrong_id(json!({"type":"tool_use","id":false}), false)]
#[case::wrong_thinking(json!({"type":"thinking","thinking":[]}), false)]
fn content_and_stream_blocks_share_payload_value_semantics(
    #[case] wire: Value,
    #[case] accepted: bool,
) {
    let content = serde_json::from_value::<ContentBlock>(wire.clone());
    let streaming = serde_json::from_value::<MessagesContentBlock>(wire.clone());
    assert_eq!(content.is_ok(), accepted);
    assert_eq!(streaming.is_ok(), accepted);
    if let (Ok(content), Ok(streaming)) = (content, streaming) {
        assert_eq!(content.payload, streaming.payload);
        assert!(content.extra.is_empty());
        assert_eq!(serde_json::to_value(content).unwrap(), wire);
        assert_eq!(serde_json::to_value(streaming).unwrap(), wire);
    }
}

#[rstest]
fn flattened_payload_preserves_unknown_fields_named_payload() {
    let wire = json!({"type":"future","payload":{"nested":null},"future":[1,null]});
    let content: ContentBlock = serde_json::from_value(wire.clone()).unwrap();
    let streaming: MessagesContentBlock = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(content.extra.get("payload"), wire.get("payload"));
    assert_eq!(streaming.extra.get("future"), wire.get("future"));
    assert_eq!(serde_json::to_value(content).unwrap(), wire);
    assert_eq!(serde_json::to_value(streaming).unwrap(), wire);
}

#[rstest]
#[case::wrong_count(json!({"input_tokens":"many"}), false)]
#[case::nullable_counts(json!({"input_tokens":null,"output_tokens":null,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}), true)]
#[case::unknown_iterations(json!({"iterations":[{"type":"future","input_tokens":17}]}), true)]
fn stream_usage_validation_keeps_its_existing_boundary(
    #[case] usage: Value,
    #[case] accepted: bool,
) {
    let wire = json!({"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":null},"usage":usage});
    let parsed = serde_json::from_value::<MessagesStreamEvent>(wire.clone());
    assert_eq!(parsed.is_ok(), accepted);
    if let Ok(parsed) = parsed {
        assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
    }
}
