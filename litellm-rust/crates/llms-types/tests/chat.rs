use litellm_llms_types::formats::chat_completions::{ChatContentPart, ChatLogprobs, ChatMediaUrl};
use litellm_llms_types::formats::messages::{Citations, ContentSource};
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
#[case::parts(json!([
    {"type":"text","text":"hello","cache_control":{"type":"ephemeral"}},
    {"type":"image_url","image_url":{"url":"https://example.test/image","detail":"high"}},
    {"type":"video_url","video_url":"https://example.test/video"},
    {"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}},
    {"type":"file","file":{"file_id":"file_1","video_metadata":{"fps":1,"start_offset":"1s"}}},
    {"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true}},
    {"type":"refusal","refusal":"refused"}
]))]
fn content_parts_round_trip(#[case] wire: Value) {
    let parts: Vec<ChatContentPart> = serde_json::from_value(wire.clone()).unwrap();
    let [
        ChatContentPart::Text {
            text,
            cache_control,
            ..
        },
        ChatContentPart::ImageUrl {
            image_url: ChatMediaUrl::Parameters(image),
            ..
        },
        ChatContentPart::VideoUrl {
            video_url: ChatMediaUrl::Url(video),
            ..
        },
        ChatContentPart::InputAudio { input_audio, .. },
        ChatContentPart::File { file, .. },
        ChatContentPart::Document {
            source,
            citations: Some(Citations::Config(citations)),
            ..
        },
        ChatContentPart::Refusal { refusal, .. },
    ] = parts.as_slice()
    else {
        panic!("expected typed content parts");
    };
    assert_eq!(text, "hello");
    assert_eq!(
        cache_control.as_ref().unwrap().cache_type.as_deref(),
        Some("ephemeral")
    );
    assert_eq!(image.url.as_deref(), Some("https://example.test/image"));
    assert_eq!(image.detail.as_deref(), Some("high"));
    assert_eq!(video, "https://example.test/video");
    assert_eq!(input_audio.data.as_deref(), Some("AA=="));
    assert_eq!(input_audio.format.as_deref(), Some("wav"));
    assert_eq!(file.file_id.as_deref(), Some("file_1"));
    assert_eq!(file.video_metadata.as_ref().unwrap().fps, Some(1.into()));
    let ContentSource::Text {
        data, media_type, ..
    } = source.as_ref()
    else {
        panic!("expected document text source");
    };
    assert_eq!(data, "doc");
    assert_eq!(media_type, "text/plain");
    assert_eq!(citations.enabled, Some(true));
    assert_eq!(refusal, "refused");
    assert_eq!(serde_json::to_value(parts).unwrap(), wire);
}

#[rstest]
fn logprobs_round_trip_with_tokens_and_alternatives() {
    round_trip::<ChatLogprobs>(json!({
        "content":[{"token":"hi","logprob":-1,"bytes":[104,105],"top_logprobs":[{"token":"hey","logprob":-2.5,"bytes":[104]}]}],
        "refusal":[{"token":"refused","logprob":-3}]
    }));
}
