use litellm_llms_types::{
    formats::chat_completions::{
        ChatContentPart, ChatLogprobs, ChatMediaUrl, ChatMessageContent, ReasoningEffort,
    },
    recognized::Recognized,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text(json!({"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}))]
#[case::image(json!({"type":"image_url","image_url":{"url":"https://example.test/image","detail":"high"}}))]
#[case::video(json!({"type":"video_url","video_url":"https://example.test/video"}))]
#[case::audio(json!({"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}))]
#[case::file(json!({"type":"file","file":{"file_id":"file_1","video_metadata":{"fps":1,"start_offset":"1s"}}}))]
#[case::document(json!({"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true}}))]
#[case::refusal(json!({"type":"refusal","refusal":"refused"}))]
fn known_content_parts_are_typed_and_lossless(#[case] wire: Value) {
    let parsed: Recognized<ChatContentPart> = serde_json::from_value(wire.clone()).unwrap();
    assert!(parsed.known().is_some());
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
fn image_parameters_are_accessible_without_parsing_json() {
    let parsed: Recognized<ChatContentPart> = serde_json::from_value(
        json!({"type":"image_url","image_url":{"url":"https://example.test","detail":"high"}}),
    )
    .unwrap();
    let Some(ChatContentPart::ImageUrl {
        image_url: Recognized::Known(ChatMediaUrl::Parameters(image)),
        ..
    }) = parsed.known()
    else {
        panic!("expected image")
    };
    assert_eq!(
        image.url,
        Some(Recognized::Known("https://example.test".into()))
    );
    assert_eq!(image.detail, Some(Recognized::Known("high".into())));
}

#[rstest]
#[case::plain(json!({"type":"text","text":"hi"}), Some("hi"), Some("hi"))]
#[case::extra(json!({"type":"text","text":"hi","extension":null}), Some("hi"), None)]
#[case::cached(json!({"type":"text","text":"hi","cache_control":null}), Some("hi"), None)]
#[case::unknown_text(json!({"type":"future","text":"hi"}), Some("hi"), None)]
#[case::untyped_text(json!({"text":"hi"}), Some("hi"), None)]
#[case::malformed(json!({"type":"text","text":17}), None, None)]
#[case::scalar(json!(17), None, None)]
fn text_access_preserves_partial_payloads_and_exact_plain_text_shape(
    #[case] wire: Value,
    #[case] text: Option<&str>,
    #[case] plain: Option<&str>,
) {
    let parsed: ChatMessageContent = serde_json::from_value(json!([wire.clone()])).unwrap();
    let ChatMessageContent::Parts(parts) = &parsed else {
        panic!("expected parts")
    };
    assert_eq!(parts[0].text(), text);
    assert_eq!(parts[0].plain_text(), plain);
    assert_eq!(serde_json::to_value(parsed).unwrap(), json!([wire]));
}

#[rstest]
fn logprobs_expose_tokens_bytes_and_alternatives_without_changing_numeric_forms() {
    let wire = json!({"content":[{"token":"hi","logprob":-1,"bytes":[104,105],"top_logprobs":[{"token":"hey","logprob":-2.5,"bytes":null}]}],"refusal":null,"future":true});
    let parsed: ChatLogprobs = serde_json::from_value(wire.clone()).unwrap();
    let token = parsed.content.as_ref().unwrap().known().unwrap()[0]
        .known()
        .unwrap();
    assert_eq!(token.token, Some(Recognized::Known("hi".into())));
    assert_eq!(token.bytes, Some(Recognized::Known(vec![104, 105])));
    assert_eq!(
        token.logprob,
        Some(Recognized::Known(serde_json::Number::from(-1)))
    );
    assert_eq!(
        token.top_logprobs.as_ref().unwrap().known().unwrap()[0]
            .known()
            .unwrap()
            .token,
        Some(Recognized::Known("hey".into()))
    );
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
fn reasoning_effort_names_match_the_wire_and_parse_back(
    #[values(
        ReasoningEffort::None,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
        ReasoningEffort::Max
    )]
    effort: ReasoningEffort,
) {
    assert_eq!(
        serde_json::to_value(effort).unwrap(),
        Value::String(effort.as_str().to_string())
    );
    assert_eq!(ReasoningEffort::parse(effort.as_str()), Some(effort));
    assert!(ReasoningEffort::ALL.contains(&effort));
}

#[rstest]
#[case::unknown("ultra")]
#[case::uppercase("HIGH")]
#[case::empty("")]
fn reasoning_effort_parse_rejects(#[case] value: &str) {
    assert_eq!(ReasoningEffort::parse(value), None);
}
