use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

#[test]
fn streams_match_serde_including_escaping_and_base64_boundaries() {
    let text = format!(
        "{}{}é💙\\\"\n",
        "x".repeat(JSON_BODY_CHUNK_BYTES - 1),
        (0_u8..32).map(char::from).collect::<String>()
    );
    for length in [0, 1, 2, 3, 49151, 49152, 49153, 131072] {
        let payload = JsonPayload::Object(BTreeMap::from([
            ("image".into(), JsonPayload::String(text.clone().into())),
            (
                "audio".into(),
                JsonPayload::Base64(vec![123; length].into()),
            ),
            (
                "nested".into(),
                serde_json::json!([null, true, 2.5, {"key": "value"}]).into(),
            ),
        ]));
        let expected = serde_json::to_vec(&payload).unwrap();
        let body = PreparedJsonBody::streamed(payload).unwrap();
        let actual: Vec<u8> = body.chunks().flat_map(|bytes| bytes.to_vec()).collect();
        assert_eq!(actual, expected);
        assert_eq!(body.content_length(), expected.len() as u64);
        assert_eq!(body.sha256(), format!("{:x}", Sha256::digest(&expected)));
        assert!(
            body.chunks()
                .all(|chunk| chunk.len() <= JSON_BODY_CHUNK_BYTES)
        );
        assert_eq!(body.chunks().flatten().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn shared_payload_slices_and_replays_retain_the_owner_without_copying() {
    struct Owner(Arc<Vec<u8>>);
    impl AsRef<[u8]> for Owner {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }
    let owner = Arc::new(vec![b'A'; 1024 * 1024]);
    let pointer = owner.as_ptr();
    let bytes = Bytes::from_owner(Owner(owner.clone()));
    let text = SharedText::new(bytes).unwrap();
    let slice = text.slice(1..text.bytes().len()).unwrap();
    assert_eq!(slice.bytes().as_ptr(), pointer.wrapping_add(1));
    let body = PreparedJsonBody::streamed(JsonPayload::String(text)).unwrap();
    drop(slice);
    assert_eq!(Arc::strong_count(&owner), 2);
    for _ in 0..2 {
        let mut chunks = body.chunks();
        assert_eq!(chunks.next().unwrap(), "\"");
        assert_eq!(chunks.next().unwrap().as_ptr(), pointer);
    }
    drop(body);
    assert_eq!(Arc::strong_count(&owner), 1);
}

#[test]
fn non_media_remains_buffered_and_data_uris_stream() {
    assert!(
        !PreparedJsonBody::new(serde_json::json!({"text":"hello"}).into())
            .unwrap()
            .is_streamed()
    );
    assert!(
        PreparedJsonBody::new(
            serde_json::json!({"document_url":"data:application/pdf;base64,AA=="}).into()
        )
        .unwrap()
        .is_streamed()
    );
}

#[test]
fn serde_escaping_stays_bounded_for_long_control_runs_and_unicode_splits() {
    for text in [
        "\u{0001}".repeat(JSON_BODY_CHUNK_BYTES * 3),
        format!("{}é💙\n\\\"", "a".repeat(JSON_BODY_CHUNK_BYTES - 1)),
    ] {
        let expected = serde_json::to_vec(&text).unwrap();
        let body = PreparedJsonBody::streamed(JsonPayload::String(text.into())).unwrap();
        assert_eq!(body.content_length(), expected.len() as u64);
        assert!(
            body.chunks()
                .all(|chunk| chunk.len() <= JSON_BODY_CHUNK_BYTES)
        );
        assert_eq!(body.chunks().flatten().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn typed_messages_keep_nested_media_in_outgoing_slices() {
    use crate::messages::transformation::AnthropicMessagesProviderConfig;
    use crate::messages::types::{AnthropicMessagesRequest, MessageContent};
    use crate::providers::azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG;
    let bytes = Bytes::from(vec![b'A'; JSON_BODY_CHUNK_BYTES * 3]);
    let mut request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({"model":"model","messages":[{"role":"system","content":[{"type":"tool_result","content":[]}]}]})).unwrap();
    let MessageContent::Blocks(blocks) = &mut request.messages[0].content else {
        panic!("blocks")
    };
    blocks[0].extra.insert(
        "content".into(),
        JsonPayload::Array(vec![JsonPayload::object([
            ("type", "image".into()),
            (
                "source",
                JsonPayload::object([
                    ("type", "base64".into()),
                    (
                        "data",
                        JsonPayload::String(SharedText::new(bytes.clone()).unwrap()),
                    ),
                ]),
            ),
        ])]),
    );
    let request = AZURE_ANTHROPIC_MESSAGES_CONFIG
        .transform_request(request)
        .unwrap();
    let expected = serde_json::to_vec(&request).unwrap();
    let body = PreparedJsonBody::new(request.into_payload().unwrap()).unwrap();
    let emitted = body.chunks().flatten().collect::<Vec<_>>();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&emitted).unwrap(),
        serde_json::from_slice::<serde_json::Value>(&expected).unwrap()
    );
    assert!(
        body.chunks()
            .any(|chunk| chunk.as_ptr() == bytes.as_ptr() && chunk.len() == JSON_BODY_CHUNK_BYTES)
    );
}
