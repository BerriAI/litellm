use litellm_llms::{
    base_llm::messages::transformation::BaseMessagesConfig,
    edenai::messages::transformation::EDENAI_MESSAGES_CONFIG,
};
use litellm_llms_types::formats::messages::MessagesResponse;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::number(json!(0.025), Some(0.025))]
#[case::free(json!(0), Some(0.0))]
#[case::text(json!("0.75"), Some(0.75))]
#[case::absent(Value::Null, None)]
#[case::malformed(json!("not-a-cost"), None)]
#[case::negative(json!(-1), None)]
#[case::boolean(json!(true), None)]
#[case::infinite(json!("inf"), None)]
#[case::nan(json!("NaN"), None)]
fn reported_cost_is_a_finite_nonnegative_amount(
    #[case] cost: Value,
    #[case] expected: Option<f64>,
) {
    let response: MessagesResponse = serde_json::from_value(json!({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "native-test",
        "content": [], "stop_reason": "end_turn", "stop_sequence": null, "cost": cost,
    }))
    .unwrap();
    assert_eq!(
        EDENAI_MESSAGES_CONFIG
            .reported_cost(&response)
            .and_then(|cost| cost.as_f64()),
        expected
    );
}
