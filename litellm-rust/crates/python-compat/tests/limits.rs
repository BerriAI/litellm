use std::time::{Duration, Instant};

use litellm_python_compat::{Error, MAX_DEPTH, Value, json, literal::literal_eval, pickle};
use rstest::{fixture, rstest};

/// The bracket pair of one container shape, as `(open, close)`.
#[fixture]
fn shapes() -> [(&'static str, &'static str); 3] {
    [("[", "]"), ("{'a': ", "}"), ("(", ",)")]
}

fn nested_text(open: &str, close: &str, depth: usize) -> String {
    format!("{}1{}", open.repeat(depth), close.repeat(depth))
}

fn nested_list(depth: usize) -> Value {
    (0..depth).fold(Value::from(1), |value, _| Value::List(vec![value]))
}

/// A protocol 3 pickle of `depth` nested lists around `1`: `EMPTY_LIST` per level, then
/// `BININT1 1`, then `APPEND` per level. Written by hand because `dumps` refuses the depth.
fn nested_list_pickle(depth: usize) -> Vec<u8> {
    let mut data = vec![0x80, 3];
    data.extend(std::iter::repeat_n(b']', depth));
    data.extend([b'K', 1]);
    data.extend(std::iter::repeat_n(b'a', depth));
    data.push(b'.');
    data
}

#[rstest]
fn literal_eval_accepts_the_limit_and_rejects_past_it(shapes: [(&'static str, &'static str); 3]) {
    for (open, close) in shapes {
        assert!(literal_eval(&nested_text(open, close, MAX_DEPTH)).is_ok());
        assert!(matches!(
            literal_eval(&nested_text(open, close, MAX_DEPTH + 1)),
            Err(Error::TooDeep)
        ));
    }
}

/// A backtracking parser (the `py_literal` grammar this replaced) doubles per nested level
/// and takes minutes here; the bound is loose enough to survive a slow debug build.
#[rstest]
fn literal_eval_stays_linear_in_depth(shapes: [(&'static str, &'static str); 3]) {
    for (open, close) in shapes {
        let text = nested_text(open, close, MAX_DEPTH);
        let start = Instant::now();
        assert!(literal_eval(&text).is_ok());
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(50),
            "{open} nested {MAX_DEPTH} deep took {elapsed:?}"
        );
    }
}

#[rstest]
fn literal_eval_ignores_brackets_inside_strings() {
    let text = format!("'{}'", "[".repeat(MAX_DEPTH + 1));
    assert!(matches!(literal_eval(&text), Ok(Value::Str(_))));
}

#[rstest]
fn pickle_nesting_is_bounded_in_both_directions() {
    assert_eq!(
        pickle::loads(&nested_list_pickle(MAX_DEPTH)).unwrap(),
        nested_list(MAX_DEPTH)
    );
    assert!(matches!(
        pickle::loads(&nested_list_pickle(MAX_DEPTH + 1)),
        Err(Error::InvalidPickle(_))
    ));
    assert!(pickle::dumps(&nested_list(MAX_DEPTH)).is_ok());
    assert!(matches!(
        pickle::dumps(&nested_list(MAX_DEPTH + 2)),
        Err(Error::TooDeep)
    ));
}

#[rstest]
#[case("{1, 2}", "set")]
#[case("1+2j", "complex")]
fn pickle_dumps_refuses_types_it_would_change(#[case] source: &str, #[case] type_name: &str) {
    let value = literal_eval(source).expect("source is a literal");
    assert!(matches!(pickle::dumps(&value), Err(Error::NotPicklable(name)) if name == type_name));
}

#[rstest]
#[case(b"\x80\x02c__builtin__\ncomplex\nq\x00.".to_vec(), "a class reference")]
#[case({ let mut data = pickle::dumps(&Value::from(1)).unwrap(); data.push(b'.'); data }, "trailing data")]
fn pickle_loads_rejects(#[case] data: Vec<u8>, #[case] what: &str) {
    assert!(
        matches!(pickle::loads(&data), Err(Error::InvalidPickle(_))),
        "{what} must not decode"
    );
}

#[rstest]
#[case(Value::Bytes(b"x".to_vec()), "Object of type bytes is not JSON serializable")]
#[case(Value::Set(vec![Value::from(1)]), "Object of type set is not JSON serializable")]
#[case(Value::Complex { re: 1.0, im: 2.0 }, "Object of type complex is not JSON serializable")]
#[case(
    Value::Float(f64::NAN),
    "Out of range float values are not JSON compliant"
)]
fn to_json_reports_what_python_json_dumps_would_reject(
    #[case] value: Value,
    #[case] message: &str,
) {
    let error = json::to_json(&value).expect_err("value has no serde_json form");
    assert_eq!(error.to_string(), message);
}

/// `json.dumps` writes the non-finite floats that `to_json` cannot represent.
#[rstest]
#[case(f64::NAN, "NaN")]
#[case(f64::INFINITY, "Infinity")]
#[case(f64::NEG_INFINITY, "-Infinity")]
fn json_dumps_writes_non_finite_floats(#[case] value: f64, #[case] text: &str) {
    assert_eq!(json::dumps(&Value::Float(value)).unwrap(), text);
}
