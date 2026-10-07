use litellm_llms_types::formats::chat_completions::{ChatLogprobs, ChatMessageContent};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T>(wire: Value)
where
    T: DeserializeOwned + Serialize,
{
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::text(json!("plain text"))]
#[case::parts(json!([
    {"type":"text","text":"hello","cache_control":{"type":"ephemeral"}},
    {"type":"image_url","image_url":{"url":"https://example.test/image","detail":"high"}},
    {"type":"video_url","video_url":"https://example.test/video"},
    {"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}},
    {"type":"file","file":{"file_id":"file_1","video_metadata":{"fps":1,"start_offset":"1s"}}},
    {"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true}},
    {"type":"refusal","refusal":"refused"}
]))]
fn message_content_round_trips(#[case] wire: Value) {
    round_trip::<ChatMessageContent>(wire);
}

#[rstest]
fn logprobs_round_trip_with_tokens_and_alternatives() {
    round_trip::<ChatLogprobs>(json!({
        "content":[{"token":"hi","logprob":-1,"bytes":[104,105],"top_logprobs":[{"token":"hey","logprob":-2.5,"bytes":[104]}]}],
        "refusal":[{"token":"refused","logprob":-3}]
    }));
}
