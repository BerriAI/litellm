use serde_json::{Map, Value};

use crate::formats::messages::CacheControl;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MinimaxMessagesContentBlock {
    Image(MinimaxMediaBlock),
    Video(MinimaxMediaBlock),
    MidConvSystem {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct MinimaxMediaBlock {
    pub source: MinimaxMediaSource,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MinimaxMediaSource {
    Base64 {
        media_type: String,
        data: String,
        #[serde(flatten)]
        options: MinimaxMediaOptions,
    },
    Url {
        url: String,
        #[serde(flatten)]
        options: MinimaxMediaOptions,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MinimaxMediaOptions {
    pub detail: Option<MinimaxMediaDetail>,
    pub fps: Option<serde_json::Number>,
    pub max_long_side_pixel: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum MinimaxMediaDetail {
    Low,
    Default,
    High,
}

#[cfg(test)]
mod tests {
    use crate::providers::minimax::{
        MinimaxMediaDetail, MinimaxMediaSource, MinimaxMessagesContentBlock,
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
                let MinimaxMediaSource::Base64 {
                    media_type,
                    data,
                    options,
                } = &image.source
                else {
                    panic!("expected base64 image source");
                };
                assert_eq!((media_type.as_str(), data.as_str()), ("image/png", "AA=="));
                assert_eq!(options.detail, Some(MinimaxMediaDetail::Low));
                assert!(options.extra.is_empty());
            }
            MinimaxMessagesContentBlock::Video(video) => {
                let MinimaxMediaSource::Url { url, options } = &video.source else {
                    panic!("expected URL video source");
                };
                assert_eq!(url, "https://example.test/video");
                assert_eq!(options.detail, Some(MinimaxMediaDetail::High));
                assert_eq!(options.fps, Some(1.into()));
                assert_eq!(options.max_long_side_pixel, Some(1024));
                assert_eq!(options.extra["future"], Value::Null);
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
    #[case::url_without_url(json!({"type":"video","source":{"type":"url"}}))]
    #[case::base64_without_data(json!({"type":"image","source":{"type":"base64","media_type":"image/png"}}))]
    #[case::bad_detail(json!({"type":"image","source":{"type":"url","url":"u","detail":7}}))]
    #[case::bad_fps(json!({"type":"video","source":{"type":"url","url":"u","fps":"fast"}}))]
    #[case::missing_text(json!({"type":"mid_conv_system"}))]
    fn provider_content_rejects_malformed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<MinimaxMessagesContentBlock>(wire).is_err());
    }

    #[rstest]
    fn partial_media_options_omit_null_optionals() {
        let block: MinimaxMessagesContentBlock = serde_json::from_value(json!({
            "type":"video",
            "source":{"type":"url","url":"u","detail":null,"fps":null,"future":null},
            "cache_control":null
        }))
        .unwrap();
        let MinimaxMessagesContentBlock::Video(video) = &block else {
            panic!("expected video")
        };
        let MinimaxMediaSource::Url { options, .. } = &video.source else {
            panic!("expected URL source");
        };
        assert!(options.detail.is_none());
        assert!(options.fps.is_none());
        assert!(video.cache_control.is_none());
        assert_eq!(
            serde_json::to_value(block).unwrap(),
            json!({"type":"video","source":{"type":"url","url":"u","future":null}})
        );
    }
}
