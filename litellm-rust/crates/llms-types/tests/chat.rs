use litellm_llms_types::formats::chat_completions::{ChatContentPart, ChatLogprobs, ChatMediaUrl};
use litellm_llms_types::formats::messages::{Citations, ContentSource};
use rstest::rstest;
use serde_json::{Value, json};

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
    let wire = json!({
        "content":[{"token":"hi","logprob":-1,"bytes":[104,105],"top_logprobs":[{"token":"hey","logprob":-2.5,"bytes":[104]}]}],
        "refusal":[{"token":"refused","logprob":-3}]
    });
    let logprobs: ChatLogprobs = serde_json::from_value(wire.clone()).unwrap();
    let token = &logprobs.content.as_ref().unwrap()[0];
    assert_eq!(token.token.as_deref(), Some("hi"));
    assert_eq!(token.logprob, Some((-1).into()));
    assert_eq!(token.bytes.as_deref(), Some([104, 105].as_slice()));
    let alternative = &token.top_logprobs.as_ref().unwrap()[0];
    assert_eq!(alternative.token.as_deref(), Some("hey"));
    assert_eq!(alternative.bytes.as_deref(), Some([104].as_slice()));
    assert_eq!(
        logprobs.refusal.as_ref().unwrap()[0].token.as_deref(),
        Some("refused")
    );
    assert_eq!(serde_json::to_value(logprobs).unwrap(), wire);
}

#[rstest]
#[case::text(json!({"type":"text","text":false}))]
#[case::missing_image(json!({"type":"image_url"}))]
#[case::audio_shape(json!({"type":"input_audio","input_audio":{"data":7}}))]
#[case::file_metadata(json!({"type":"file","file":{"video_metadata":{"fps":"fast"}}}))]
#[case::document_source(json!({"type":"document","source":{"type":"url","url":false}}))]
#[case::unknown_tag(json!({"type":"future"}))]
fn content_parts_reject_malformed_typed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ChatContentPart>(wire).is_err());
}

#[rstest]
fn partial_audio_preserves_extensions_and_omits_null_optionals() {
    let wire = json!({"type":"input_audio","input_audio":{"data":null,"extension":[1,null]},"future":true});
    let part: ChatContentPart = serde_json::from_value(wire).unwrap();
    let ChatContentPart::InputAudio { input_audio, extra } = &part else {
        panic!("expected audio")
    };
    assert!(input_audio.data.is_none());
    assert!(input_audio.format.is_none());
    assert_eq!(extra["future"], json!(true));
    assert_eq!(
        serde_json::to_value(part).unwrap(),
        json!({"type":"input_audio","input_audio":{"extension":[1,null]},"future":true})
    );
}
