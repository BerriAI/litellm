use litellm_llms_types::billing::BilledAmount;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::number(json!(0.025), 0.025)]
#[case::free(json!(0), 0.0)]
#[case::integer(json!(3), 3.0)]
#[case::text(json!("0.75"), 0.75)]
fn a_billed_amount_reads_numbers_and_numeric_strings(#[case] wire: Value, #[case] expected: f64) {
    let amount: BilledAmount = serde_json::from_value(wire).unwrap();
    assert_eq!(amount.as_f64(), expected);
    assert_eq!(serde_json::to_value(amount).unwrap(), json!(expected));
}

#[rstest]
#[case::malformed(json!("not-a-cost"))]
#[case::negative(json!(-1))]
#[case::negative_text(json!("-0.5"))]
#[case::boolean(json!(true))]
#[case::object(json!({"total": 1}))]
#[case::null(Value::Null)]
#[case::infinite(json!("inf"))]
#[case::nan(json!("NaN"))]
fn anything_else_is_not_an_amount(#[case] wire: Value) {
    assert!(serde_json::from_value::<BilledAmount>(wire).is_err());
}
