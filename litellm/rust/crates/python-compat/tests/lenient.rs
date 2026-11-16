use litellm_python_compat::lenient::{IsTrue, Lenient, Truthy};
use rstest::rstest;
use serde::Deserialize;
use serde_json::{Value, json};
use serde_with::serde_as;

#[serde_as]
#[derive(Deserialize)]
struct Flag(#[serde_as(as = "IsTrue")] bool);

/// Expected values from CPython: `value is True` for the JSON value loaded with `json.loads`.
#[rstest]
#[case::true_(json!(true), true)]
#[case::false_(json!(false), false)]
#[case::string_true(json!("true"), false)]
#[case::one(json!(1), false)]
#[case::null(json!(null), false)]
#[case::list_with_true(json!([true]), false)]
fn is_true_is_only_the_boolean_true(#[case] value: Value, #[case] expected: bool) {
    assert_eq!(serde_json::from_value::<Flag>(value).unwrap().0, expected);
}

#[serde_as]
#[derive(Deserialize)]
struct Bool(#[serde_as(as = "Truthy")] bool);

/// Expected values from CPython: `bool(value)` for the JSON value loaded with `json.loads`.
#[rstest]
#[case::true_(json!(true), true)]
#[case::false_(json!(false), false)]
#[case::zero(json!(0), false)]
#[case::one(json!(1), true)]
#[case::zero_float(json!(0.0), false)]
#[case::half(json!(0.5), true)]
#[case::empty_string(json!(""), false)]
#[case::string_false(json!("false"), true)]
#[case::null(json!(null), false)]
#[case::empty_list(json!([]), false)]
#[case::list_of_false(json!([false]), true)]
#[case::empty_mapping(json!({}), false)]
#[case::mapping(json!({"a": null}), true)]
fn truthy_is_python_bool(#[case] value: Value, #[case] expected: bool) {
    assert_eq!(serde_json::from_value::<Bool>(value).unwrap().0, expected);
}

#[rstest]
#[case::list(json!(["/a", "/b/*"]), Lenient::Valid(["/a".to_owned(), "/b/*".to_owned()].into()))]
#[case::empty(json!([]), Lenient::Valid([].into()))]
#[case::bare_string(json!("/a/*"), Lenient::Malformed)]
#[case::number(json!(5), Lenient::Malformed)]
#[case::mapping(json!({"a": 1}), Lenient::Malformed)]
#[case::non_string_entry(json!(["/a", 5, "/b/*"]), Lenient::Malformed)]
#[case::null_entry(json!(["/a", null]), Lenient::Malformed)]
fn a_value_of_another_shape_is_malformed(
    #[case] value: Value,
    #[case] expected: Lenient<Box<[String]>>,
) {
    assert_eq!(
        serde_json::from_value::<Lenient<Box<[String]>>>(value).unwrap(),
        expected
    );
}

#[rstest]
#[case::listed(Lenient::Valid(["/a".to_owned()].into()), &["/a"][..])]
#[case::malformed(Lenient::Malformed, &[][..])]
fn items_of_a_malformed_list_are_none(
    #[case] list: Lenient<Box<[String]>>,
    #[case] expected: &[&str],
) {
    assert_eq!(list.items(), expected);
}
