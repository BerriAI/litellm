#![cfg(feature = "serde-compat")]

use litellm_python_compat::serde_compat::{FiniteF64, LaxI64};
use rstest::rstest;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use serde_with::serde_as;

#[serde_as]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct Numbers {
    #[serde_as(deserialize_as = "Option<Vec<LaxI64>>")]
    integers: Option<Vec<i64>>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    float: Option<f64>,
}

#[rstest]
fn adapters_compose_and_serialize_as_numbers() {
    let numbers: Numbers = serde_json::from_value(json!({
        "integers": ["9007199254740993.0", "1_000", " +2.000 ", 3.0, true],
        "float": " 1.5 "
    }))
    .unwrap();
    assert_eq!(
        serde_json::to_value(numbers).unwrap(),
        json!({"integers": [9_007_199_254_740_993_i64, 1000, 2, 3, 1], "float": 1.5})
    );
}

#[rstest]
#[case::missing(json!({}))]
#[case::null(json!({"integers": null, "float": null}))]
fn optional_adapters_accept_missing_and_null_fields(#[case] input: Value) {
    assert_eq!(
        serde_json::from_value::<Numbers>(input).unwrap(),
        Numbers {
            integers: None,
            float: None
        }
    );
}

#[rstest]
#[case::minimum(json!(i64::MIN), i64::MIN)]
#[case::maximum(json!(i64::MAX), i64::MAX)]
#[case::maximum_string(json!(i64::MAX.to_string()), i64::MAX)]
fn integers_preserve_bounds(#[case] input: Value, #[case] expected: i64) {
    let numbers: Numbers = serde_json::from_value(json!({"integers": [input]})).unwrap();
    assert_eq!(numbers.integers, Some(vec![expected]));
}

#[rstest]
#[case::unsigned_maximum(json!(u64::MAX))]
#[case::above_maximum(json!(9_223_372_036_854_775_808_u64))]
#[case::float_above_maximum(json!(9_223_372_036_854_775_808.0))]
#[case::below_minimum(json!("-9223372036854775809"))]
#[case::precise_fraction(json!("1.0000000000000001"))]
#[case::exponent(json!("1e3"))]
#[case::missing_fraction(json!("2."))]
#[case::missing_integer(json!(".0"))]
#[case::leading_separator(json!("_2"))]
#[case::repeated_separator(json!("2__0"))]
#[case::fraction(json!(2.5))]
#[case::null(json!(null))]
#[case::object(json!({}))]
fn integers_reject_invalid_values(#[case] input: Value) {
    assert!(serde_json::from_value::<Numbers>(json!({"integers": [input]})).is_err());
}

#[rstest]
#[case::nan(json!("NaN"))]
#[case::positive_infinity(json!("inf"))]
#[case::negative_infinity(json!("-inf"))]
#[case::overflow(json!("1e999"))]
#[case::array(json!([]))]
fn floats_reject_nonfinite_and_invalid_values(#[case] input: Value) {
    assert!(serde_json::from_value::<Numbers>(json!({"float": input})).is_err());
}

#[rstest]
#[case::integer(json!(2), 2.0)]
#[case::float(json!(2.5), 2.5)]
#[case::boolean(json!(true), 1.0)]
fn floats_accept_finite_numbers(#[case] input: Value, #[case] expected: f64) {
    let numbers: Numbers = serde_json::from_value(json!({"float": input})).unwrap();
    assert_eq!(numbers.float, Some(expected));
}
