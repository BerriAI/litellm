use litellm_llms_types::providers::minimax::{
    MinimaxMediaDetail, MinimaxMediaSourceType, MinimaxMessagesContentBlock,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::image(json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA==","detail":"low"}}))]
#[case::video(json!({"type":"video","source":{"type":"url","url":"https://example.test/video","detail":"high","fps":1,"max_long_side_pixel":1024,"future":null},"cache_control":{"type":"ephemeral"}}))]
#[case::mid_conversation_system(json!({"type":"mid_conv_system","text":"instruction","future":null}))]
fn provider_content_blocks_round_trip(#[case] wire: Value) {
    let block: MinimaxMessagesContentBlock = serde_json::from_value(wire.clone()).unwrap();
    match &block {
        MinimaxMessagesContentBlock::Image(image) => {
            assert_eq!(image.source.source_type, MinimaxMediaSourceType::Base64);
            assert_eq!(image.source.detail, Some(MinimaxMediaDetail::Low));
            assert_eq!(image.source.data.as_deref(), Some("AA=="));
        }
        MinimaxMessagesContentBlock::Video(video) => {
            assert_eq!(video.source.source_type, MinimaxMediaSourceType::Url);
            assert_eq!(video.source.detail, Some(MinimaxMediaDetail::High));
            assert_eq!(
                video.source.url.as_deref(),
                Some("https://example.test/video")
            );
            assert_eq!(video.source.fps, Some(1.into()));
            assert_eq!(video.source.max_long_side_pixel, Some(1024));
            assert_eq!(video.source.extra["future"], Value::Null);
            assert_eq!(
                video.cache_control.as_ref().unwrap().cache_type.as_deref(),
                Some("ephemeral")
            );
        }
        MinimaxMessagesContentBlock::MidConvSystem { text, extra } => {
            assert_eq!(text, "instruction");
            assert_eq!(extra["future"], Value::Null);
        }
    }
    assert_eq!(serde_json::to_value(block).unwrap(), wire);
}

#[rstest]
#[case::missing_source(json!({"type":"video"}))]
#[case::bad_source_tag(json!({"type":"image","source":{"type":"future"}}))]
#[case::bad_detail(json!({"type":"image","source":{"type":"url","detail":7}}))]
#[case::bad_fps(json!({"type":"video","source":{"type":"url","fps":"fast"}}))]
#[case::missing_text(json!({"type":"mid_conv_system"}))]
fn provider_content_rejects_malformed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<MinimaxMessagesContentBlock>(wire).is_err());
}

#[rstest]
fn partial_media_source_omits_null_optionals() {
    let block: MinimaxMessagesContentBlock = serde_json::from_value(
        json!({"type":"video","source":{"type":"url","url":null,"future":null}}),
    )
    .unwrap();
    let MinimaxMessagesContentBlock::Video(video) = &block else {
        panic!("expected video")
    };
    assert!(video.source.url.is_none());
    assert!(video.cache_control.is_none());
    assert_eq!(
        serde_json::to_value(block).unwrap(),
        json!({"type":"video","source":{"type":"url","future":null}})
    );
}
