//! `repr`, `str`, `==`, `int()`, `float()`, `sum` and `%` formatting over the values
//! LiteLLM's rate limiters read and build their messages from. Expected values are CPython 3.14.

use litellm_python_compat::{
    format::percent,
    json::{loaded_equals, loaded_str},
    number::{ConversionError, float_of, float_to_int, int_of, sum},
    repr::str_repr,
    unicode::is_latin1,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::plain("plain", "'plain'")]
#[case::apostrophe("it's", "\"it's\"")]
#[case::double_quote("a\"b", "'a\"b'")]
#[case::both_quotes("both ' and \"", "'both \\' and \"'")]
#[case::backslash("back\\slash", "'back\\\\slash'")]
#[case::tab("tab\t", "'tab\\t'")]
fn str_repr_matches_repr(#[case] text: &str, #[case] expected: &str) {
    assert_eq!(str_repr(text), expected);
}

#[rstest]
#[case::int(json!(5), "5")]
#[case::bool(json!(true), "True")]
#[case::null(Value::Null, "None")]
#[case::list(json!([1, "a", null, 1.5]), "[1, 'a', None, 1.5]")]
#[case::dict(json!({"b": 1, "a": "it's"}), "{'b': 1, 'a': \"it's\"}")]
#[case::text(json!("plain"), "plain")]
fn loaded_str_matches_str_of_json_loads(#[case] value: Value, #[case] expected: &str) {
    assert_eq!(loaded_str(&value), expected);
}

#[rstest]
#[case::int_float(json!(1), json!(1.0), true)]
#[case::bool_int(json!(true), json!(1), true)]
#[case::text_int(json!("1"), json!(1), false)]
#[case::null(Value::Null, Value::Null, true)]
#[case::half(json!(0.5), json!(0), false)]
fn loaded_equals_matches_python_equality(
    #[case] left: Value,
    #[case] right: Value,
    #[case] expected: bool,
) {
    assert_eq!(loaded_equals(&left, &right), expected);
}

#[rstest]
#[case::tenths(vec![0.1; 10], 1.0)]
#[case::thirds(vec![0.3, 0.3, 0.3, 0.1], 1.0)]
#[case::mixed(vec![0.6, 8.0 / 30.0, 0.3], 1.166_666_666_666_666_5)]
fn sum_compensates_like_python(#[case] values: Vec<f64>, #[case] expected: f64) {
    assert_eq!(sum(values).to_bits(), expected.to_bits());
}

#[rstest]
#[case(0.8, 1, "80.0%")]
#[case(0.123_456, 2, "12.35%")]
#[case(1.0 / 3.0, 1, "33.3%")]
#[case(0.0, 2, "0.00%")]
#[case(1.25, 1, "125.0%")]
#[case(0.125, 2, "12.50%")]
#[case(f64::NAN, 1, "nan%")]
fn percent_matches_format(#[case] value: f64, #[case] precision: usize, #[case] expected: &str) {
    assert_eq!(percent(value, precision), expected);
}

#[rstest]
#[case::ascii("x-litellm-priority", true)]
#[case::latin1("café ÿ", true)]
#[case::beyond("€", false)]
fn is_latin1_matches_encode(#[case] text: &str, #[case] expected: bool) {
    assert_eq!(is_latin1(text), expected);
}

#[rstest]
#[case::int(json!(7), Ok(7))]
#[case::float_truncates(json!(7.9), Ok(7))]
#[case::negative_float_truncates(json!(-7.9), Ok(-7))]
#[case::text(json!(" 1_000 "), Ok(1000))]
#[case::bool(json!(true), Ok(1))]
#[case::bad_text(json!("7.5"), Err("invalid literal for int() with base 10: '7.5'"))]
#[case::none(
    Value::Null,
    Err("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'")
)]
#[case::list(json!([1]), Err("int() argument must be a string, a bytes-like object or a real number, not 'list'"))]
#[case::too_big(json!(1e30), Err("1e+30 does not fit in a 64-bit int"))]
#[case::too_big_text(json!("99999999999999999999"), Err("99999999999999999999 does not fit in a 64-bit int"))]
fn int_of_matches_int(#[case] value: Value, #[case] expected: Result<i64, &str>) {
    assert_eq!(
        int_of(&value).map_err(|error| error.to_string()),
        expected.map_err(str::to_owned)
    );
}

#[rstest]
#[case::int(json!(7), Ok(7.0))]
#[case::float(json!(7.5), Ok(7.5))]
#[case::text(json!(" 1_0.5\n"), Ok(10.5))]
#[case::bool(json!(false), Ok(0.0))]
#[case::bad_text(json!("x"), Err("could not convert string to float: 'x'"))]
#[case::dict(json!({}), Err("float() argument must be a string or a real number, not 'dict'"))]
fn float_of_matches_float(#[case] value: Value, #[case] expected: Result<f64, &str>) {
    assert_eq!(
        float_of(&value).map_err(|error| error.to_string()),
        expected.map_err(str::to_owned)
    );
}

#[rstest]
#[case::nan(
    f64::NAN,
    ConversionError::NotANumber,
    "cannot convert float NaN to integer"
)]
#[case::infinity(
    f64::INFINITY,
    ConversionError::Infinite,
    "cannot convert float infinity to integer"
)]
#[case::negative_infinity(
    f64::NEG_INFINITY,
    ConversionError::Infinite,
    "cannot convert float infinity to integer"
)]
fn non_finite_floats_refuse_int(
    #[case] value: f64,
    #[case] expected: ConversionError,
    #[case] message: &str,
) {
    let refused = float_to_int(value).unwrap_err();
    assert_eq!(refused.to_string(), message);
    assert_eq!(refused, expected);
}
