use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, future::ready, stream::BoxStream};
use litellm_framer::{
    SseError, frames,
    sse::{SseBlockCodec, decode_all},
};
use litellm_llms_types::formats::responses::{
    ResponsesApiResponse,
    streaming::{ResponsesStreamEvent, ResponsesStreamEventType},
};
use serde_json::{Map, Value, json};

use crate::Error;
pub use crate::base_llm::base_model_iterator::ByteStream;

const DONE: &str = "[DONE]";

fn invalid(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::InvalidResponse(crate::ErrorDetail::invalid("Responses stream event", error))
}

fn carries_event(block: &Bytes) -> bool {
    decode_all(block).is_ok_and(|events| events.iter().any(|event| event.data != DONE))
}

/// The provider's SSE bytes regrouped so each item is exactly one event's bytes, as the wire
/// carried them. The `[DONE]` terminator some OpenAI-compatible hosts append is no event and
/// is dropped.
pub fn openai_sse_blocks(bytes: ByteStream) -> BoxStream<'static, Result<Bytes, SseError>> {
    frames(bytes, SseBlockCodec::default())
        .try_filter(|block| ready(carries_event(block)))
        .boxed()
}

/// The typed events in a complete Responses SSE wire, however the bytes were chunked.
pub fn decode_openai_sse(wire: &[u8]) -> Result<Vec<ResponsesStreamEvent>, Error> {
    decode_all(wire)
        .map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::failed("stream framing", error))
        })?
        .iter()
        .filter(|event| event.data != DONE)
        .map(|event| serde_json::from_str(&event.data).map_err(invalid))
        .collect()
}

pub fn encode_openai_sse(event: &ResponsesStreamEvent) -> Result<Bytes, Error> {
    let data = serde_json::to_string(event).map_err(invalid)?;
    Ok(Bytes::from(format!(
        "event: {}\ndata: {data}\n\n",
        event.event_type
    )))
}

/// The complete response a finished Responses stream described: the one its terminal event
/// carried, or `None` when the stream ended without one.
pub fn assemble_response(
    events: impl IntoIterator<Item = ResponsesStreamEvent>,
) -> Result<Option<ResponsesApiResponse>, Error> {
    events
        .into_iter()
        .filter(|event| event.event_type.is_terminal())
        .last()
        .and_then(|mut event| event.data.remove("response"))
        .map(|response| serde_json::from_value(response).map_err(invalid))
        .transpose()
}

fn event(event_type: ResponsesStreamEventType, data: Value) -> ResponsesStreamEvent {
    ResponsesStreamEvent {
        event_type,
        data: match data {
            Value::Object(map) => map,
            _ => Map::new(),
        },
    }
}

fn text(object: &Map<String, Value>, field: &str) -> String {
    object
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn part_events(item_id: &str, output_index: usize, content_index: usize, part: &Value) -> Vec<ResponsesStreamEvent> {
    let position = |extra: Value| {
        let mut data = json!({"item_id": item_id, "output_index": output_index, "content_index": content_index});
        if let (Value::Object(data), Value::Object(extra)) = (&mut data, extra) {
            data.extend(extra);
        }
        data
    };
    let object = part.as_object().cloned().unwrap_or_default();
    let mut events = vec![event(
        ResponsesStreamEventType::ContentPartAdded,
        position(json!({"part": part})),
    )];
    if object.get("type").and_then(Value::as_str) == Some("output_text") {
        let content = text(&object, "text");
        events.push(event(
            ResponsesStreamEventType::OutputTextDelta,
            position(json!({"delta": content})),
        ));
        let annotations = object
            .get("annotations")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        events.extend(annotations.iter().enumerate().map(|(index, annotation)| {
            event(
                ResponsesStreamEventType::OutputTextAnnotationAdded,
                position(json!({"annotation_index": index, "annotation": annotation})),
            )
        }));
        events.push(event(
            ResponsesStreamEventType::OutputTextDone,
            position(json!({"text": content})),
        ));
    }
    events.push(event(
        ResponsesStreamEventType::ContentPartDone,
        position(json!({"part": part})),
    ));
    events
}

fn item_events(response_id: &str, output_index: usize, item: &Value) -> Vec<ResponsesStreamEvent> {
    let object = item.as_object().cloned().unwrap_or_default();
    let item_id = object
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or(response_id)
        .to_owned();
    let mut events = vec![event(
        ResponsesStreamEventType::OutputItemAdded,
        json!({"output_index": output_index, "item": item}),
    )];
    match object.get("type").and_then(Value::as_str) {
        Some("message") => {
            let parts = object
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            events.extend(
                parts
                    .iter()
                    .enumerate()
                    .flat_map(|(index, part)| part_events(&item_id, output_index, index, part)),
            );
        }
        Some("function_call") => {
            let arguments = text(&object, "arguments");
            events.push(event(
                ResponsesStreamEventType::FunctionCallArgumentsDelta,
                json!({"item_id": item_id, "output_index": output_index, "delta": arguments}),
            ));
            events.push(event(
                ResponsesStreamEventType::FunctionCallArgumentsDone,
                json!({"item_id": item_id, "output_index": output_index, "arguments": arguments}),
            ));
        }
        _ => {}
    }
    events.push(event(
        ResponsesStreamEventType::OutputItemDone,
        json!({"output_index": output_index, "item": item}),
    ));
    events
}

/// The event stream a client would have seen for a complete response, the way the Python
/// synthetic stream replays one: the response created and in progress with no output, every
/// output item added, spelled out and done, then the complete response.
pub fn disassemble_response(response: &ResponsesApiResponse) -> Result<Vec<ResponsesStreamEvent>, Error> {
    let complete = serde_json::to_value(response).map_err(invalid)?;
    let mut opened = complete.clone();
    if let Value::Object(opened) = &mut opened {
        opened.insert("status".into(), json!("in_progress"));
        opened.insert("output".into(), json!([]));
    }
    let items = response
        .output
        .iter()
        .enumerate()
        .flat_map(|(index, item)| item_events(&response.id, index, item));
    Ok([
        event(ResponsesStreamEventType::Created, json!({"response": opened})),
        event(ResponsesStreamEventType::InProgress, json!({"response": opened})),
    ]
    .into_iter()
    .chain(items)
    .chain([event(
        ResponsesStreamEventType::Completed,
        json!({"response": complete}),
    )])
    .collect())
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    const COMPLETED: &str = r#"{"type":"response.completed","response":{"id":"resp_1","model":"gpt","output":[],"status":"completed"}}"#;

    fn in_pieces(wire: &[u8], size: usize) -> ByteStream {
        let pieces: Vec<Bytes> = wire.chunks(size).map(Bytes::copy_from_slice).collect();
        stream::iter(pieces.into_iter().map(Ok)).boxed()
    }

    fn response(output: Vec<Value>) -> ResponsesApiResponse {
        serde_json::from_value(json!({
            "id": "resp_1",
            "object": "response",
            "model": "gpt-test",
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        }))
        .unwrap()
    }

    #[rstest]
    #[case::one_piece(usize::MAX)]
    #[case::split_inside_events(5)]
    #[tokio::test]
    async fn blocks_are_one_event_each_without_comments_or_the_done_marker(#[case] size: usize) {
        let wire = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\"}\n\n",
            ": keepalive\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: [DONE]\n\n",
        );
        let blocks: Vec<Bytes> = openai_sse_blocks(in_pieces(wire.as_bytes(), size.min(wire.len())))
            .try_collect()
            .await
            .unwrap();
        assert_eq!(
            blocks,
            vec![
                Bytes::from("event: response.created\ndata: {\"type\":\"response.created\"}\n\n"),
                Bytes::from(
                    "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"
                ),
            ]
        );
    }

    #[rstest]
    fn decoding_skips_the_done_marker_and_types_every_event() {
        let wire = format!("event: response.completed\ndata: {COMPLETED}\n\ndata: [DONE]\n\n");
        let events = decode_openai_sse(wire.as_bytes()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, ResponsesStreamEventType::Completed);
    }

    #[rstest]
    #[case::completed("response.completed", Some("resp_1"))]
    #[case::incomplete("response.incomplete", Some("resp_1"))]
    #[case::failed("response.failed", Some("resp_1"))]
    #[case::no_terminal("response.in_progress", None)]
    fn the_terminal_event_carries_the_assembled_response(#[case] terminal: &str, #[case] id: Option<&str>) {
        let wire = format!(
            "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"early\",\"model\":\"gpt\",\"output\":[]}}}}\n\nevent: {terminal}\ndata: {}\n\n",
            COMPLETED.replace("response.completed", terminal)
        );
        let assembled = assemble_response(decode_openai_sse(wire.as_bytes()).unwrap()).unwrap();
        assert_eq!(assembled.map(|response| response.id), id.map(str::to_owned));
    }

    #[rstest]
    fn a_malformed_terminal_response_is_an_error_not_a_guess() {
        let wire = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt\"}}\n\n";
        assert!(assemble_response(decode_openai_sse(wire.as_bytes()).unwrap()).is_err());
    }

    #[rstest]
    #[case::message(json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "hello", "annotations": [{"type": "url_citation", "url": "https://x.invalid"}]}]}))]
    #[case::function_call(json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}", "status": "completed"}))]
    #[case::passthrough(json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]}))]
    fn a_response_survives_the_event_round_trip(#[case] item: Value) {
        let original = response(vec![item, json!({"type": "message", "id": "msg_2", "role": "assistant", "status": "completed", "content": []})]);
        let events = disassemble_response(&original).unwrap();
        let wire: Vec<u8> = events
            .iter()
            .map(|event| encode_openai_sse(event).unwrap())
            .flat_map(|bytes| bytes.to_vec())
            .collect();
        let rebuilt = assemble_response(decode_openai_sse(&wire).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_value(&rebuilt).unwrap(),
            serde_json::to_value(&original).unwrap()
        );
    }

    #[rstest]
    fn the_replay_opens_empty_spells_out_each_item_and_ends_complete() {
        let events = disassemble_response(&response(vec![
            json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "hello", "annotations": []}]}),
            json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": "{}", "status": "completed"}),
        ]))
        .unwrap();
        let shapes: Vec<Value> = events.iter().map(|event| serde_json::to_value(event).unwrap()).collect();
        let types: Vec<&str> = shapes.iter().map(|shape| shape["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(shapes[0]["response"]["status"], "in_progress");
        assert_eq!(shapes[0]["response"]["output"], json!([]));
        assert_eq!(shapes[4]["delta"], "hello");
        assert_eq!(shapes[4]["item_id"], "msg_1");
        assert_eq!(shapes[9]["delta"], "{}");
        assert_eq!(shapes[12]["response"]["status"], "completed");
        assert_eq!(shapes[12]["response"]["output"].as_array().unwrap().len(), 2);
    }

    #[rstest]
    fn an_event_is_named_by_its_type_on_the_wire() {
        let encoded = encode_openai_sse(&event(ResponsesStreamEventType::OutputTextDelta, json!({"delta": "x"}))).unwrap();
        assert_eq!(
            encoded,
            Bytes::from("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n")
        );
    }
}
