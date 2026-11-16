use bytes::Bytes;
use futures_util::{StreamExt, stream};
use litellm_inference_responses::{
    encoded_stream::encode_stream,
    ids::{IdScope, encrypted_item_id},
};
use serde_json::{Value, json};

fn scope() -> IdScope {
    IdScope {
        provider: Some("azure".to_owned()),
        model_id: Some("deploy-1".to_owned()),
        encrypted_content_affinity: true,
    }
}

async fn encoded(chunks: Vec<&'static str>) -> String {
    let chunks = stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok::<_, std::io::Error>(Bytes::from_static(chunk.as_bytes()))),
    )
    .boxed();
    let out: Vec<Bytes> = encode_stream(chunks, scope())
        .map(Result::unwrap)
        .collect()
        .await;
    out.iter()
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect()
}

#[tokio::test]
async fn an_event_split_across_chunks_is_encoded_and_the_others_keep_their_bytes() {
    let delta = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
    let wire = encoded(vec![
        delta,
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",",
        "\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
    ])
    .await;
    let (kept, rewritten) = wire.split_at(delta.len());
    let data: Value = serde_json::from_str(
        rewritten
            .strip_prefix("event: response.output_item.added\ndata: ")
            .unwrap()
            .trim_end(),
    )
    .unwrap();
    assert_eq!(
        (kept, data),
        (
            delta,
            json!({"type": "response.output_item.added", "item": {"type": "reasoning", "id": encrypted_item_id("deploy-1", "rs_1")}})
        )
    );
}

#[tokio::test]
async fn an_unfinished_event_at_the_end_passes_through() {
    assert_eq!(encoded(vec!["data: {\"type\""]).await, "data: {\"type\"");
}
