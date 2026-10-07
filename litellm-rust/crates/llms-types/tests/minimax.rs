use litellm_llms_types::providers::minimax::MinimaxMessagesContentBlock;
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
#[case::image(json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA==","detail":"low"}}))]
#[case::video(json!({"type":"video","source":{"type":"url","url":"https://example.test/video","detail":"high","fps":1,"max_long_side_pixel":1024,"future":null},"cache_control":{"type":"ephemeral"}}))]
#[case::mid_conversation_system(json!({"type":"mid_conv_system","text":"instruction","future":null}))]
fn provider_content_blocks_round_trip(#[case] wire: Value) {
    round_trip::<MinimaxMessagesContentBlock>(wire);
}
