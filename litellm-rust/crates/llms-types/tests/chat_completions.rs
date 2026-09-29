use litellm_llms_types::{
    formats::chat_completions::{
        ChatCompletionsRequest, ChatContentPart, ChatMessageContent, ChatVideoUrl, ReasoningEffort,
    },
    recognized::Recognized,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
fn request_exposes_text_and_media_without_losing_extensions() {
    let wire = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe", "cache_control": null},
            {"type": "image_url", "image_url": {"url": "https://example.test/image", "detail": "auto", "future": null}},
            {"type": "video_url", "video_url": {"url": "https://example.test/video", "fps": 2}},
            {"type": "future_part", "payload": [null, {"x": true}]}
        ], "provider_option": null}],
        "reasoning_effort": "future-effort",
        "metadata": {"provider_option": [1, null]},
        "stream": false
    });
    let request: ChatCompletionsRequest = serde_json::from_value(wire.clone()).unwrap();
    let Some(ChatMessageContent::Parts(parts)) = &request.messages[0].content else {
        panic!("expected content parts");
    };
    assert_eq!(
        parts[0].known().and_then(ChatContentPart::text),
        Some("describe")
    );
    let Some(ChatContentPart::ImageUrl { image_url, .. }) = parts[1].known() else {
        panic!("expected typed image URL");
    };
    assert_eq!(image_url.url, "https://example.test/image");
    let Some(ChatContentPart::VideoUrl {
        video_url: ChatVideoUrl::Options(video),
        ..
    }) = parts[2].known()
    else {
        panic!("expected typed video options");
    };
    assert_eq!(video.url, "https://example.test/video");
    assert!(matches!(parts[3], Recognized::Unrecognized(_)));
    assert_eq!(serde_json::to_value(request).unwrap(), wire);
}

#[rstest]
#[case::text(json!({"type": "text", "text": "hello", "future": null}), true)]
#[case::image(json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}), true)]
#[case::video_string(json!({"type": "video_url", "video_url": "https://example.test/video"}), true)]
#[case::video_options(json!({"type": "video_url", "video_url": {"url": "https://example.test/video", "fps": null}}), true)]
#[case::unknown(json!({"type": "new_kind", "text": "preserved"}), false)]
#[case::missing_type(json!({"text": "preserved"}), false)]
#[case::missing_text(json!({"type": "text"}), false)]
#[case::null_text(json!({"type": "text", "text": null}), false)]
#[case::wrong_text_shape(json!({"type": "text", "text": {"nested": true}}), false)]
#[case::wrong_url_shape(json!({"type": "image_url", "image_url": {"url": 7}}), false)]
#[case::scalar(json!(7), false)]
#[case::null(json!(null), false)]
fn content_parts_recognize_supported_shapes_and_preserve_every_value(
    #[case] wire: Value,
    #[case] known: bool,
) {
    let part: Recognized<ChatContentPart> = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(part.known().is_some(), known);
    assert_eq!(serde_json::to_value(part).unwrap(), wire);
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
