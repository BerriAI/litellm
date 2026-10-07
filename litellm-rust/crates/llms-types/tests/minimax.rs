use litellm_llms_types::{
    formats::messages::ContentBlock,
    providers::minimax::{MinimaxMediaDetail, MinimaxMediaSourceType, MinimaxMessagesContentBlock},
    recognized::Recognized,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
fn video_source_options_are_typed_in_the_provider_contract_and_pass_through_messages() {
    let wire = json!({"type":"video","source":{"type":"url","url":"https://example.test/video","detail":"high","fps":1,"max_long_side_pixel":1024,"future":null},"cache_control":null});
    let parsed: MinimaxMessagesContentBlock = serde_json::from_value(wire.clone()).unwrap();
    let MinimaxMessagesContentBlock::Video(video) = &parsed else {
        panic!("expected video")
    };
    let source = video.source.known().unwrap();
    assert_eq!(source.source_type, MinimaxMediaSourceType::Url);
    assert_eq!(
        source.detail,
        Some(Recognized::Known(MinimaxMediaDetail::High))
    );
    assert_eq!(
        source.fps,
        Some(Recognized::Known(serde_json::Number::from(1)))
    );
    assert_eq!(source.max_long_side_pixel, Some(Recognized::Known(1024)));
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
    let shared: ContentBlock = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(shared).unwrap(), wire);
}

#[rstest]
#[case::base64_image(json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA==","detail":"low"}}))]
#[case::mid_conversation_system(json!({"type":"mid_conv_system","text":"instruction","future":null}))]
#[case::malformed_source_field(json!({"type":"video","source":{"type":"url","url":null,"fps":"future","detail":"future","max_long_side_pixel":null}}))]
fn provider_extensions_preserve_known_and_unknown_data(#[case] wire: Value) {
    let parsed: MinimaxMessagesContentBlock = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}
