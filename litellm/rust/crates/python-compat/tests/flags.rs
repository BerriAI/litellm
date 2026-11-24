//! Replays `generated/flags.json`, which LiteLLM's own helpers wrote with `scripts/generate_flag_fixtures.py`.

use litellm_python_compat::{
    Value,
    flags::{
        coerces_to_true, lax_bool, lower_is_true, redis_str_to_bool, str_to_bool,
        str_to_bool_or_truthy,
    },
    json,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    rows: Vec<Row>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Row {
    input: String,
    str_to_bool: Option<bool>,
    redis_str_to_bool: Option<bool>,
    lower_is_true: Option<bool>,
    coerces_to_true: bool,
    str_to_bool_or_truthy: bool,
    lax_bool: Option<bool>,
}

fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/flags.json"
    )))
    .expect("flags.json matches the fixture schema")
}

fn replay(input: &str) -> Row {
    let value = json::loads(input).expect("inputs are json.dumps output");
    let text = match &value {
        Value::Str(text) => Some(text.as_str()),
        _ => None,
    };
    Row {
        input: input.to_owned(),
        str_to_bool: text.and_then(str_to_bool),
        redis_str_to_bool: text.map(redis_str_to_bool),
        lower_is_true: text.map(lower_is_true),
        coerces_to_true: coerces_to_true(&value),
        str_to_bool_or_truthy: str_to_bool_or_truthy(&value),
        lax_bool: lax_bool(&value),
    }
}

#[test]
fn flags_match_litellm_helpers() {
    let mismatches: Vec<(Row, Row)> = fixtures()
        .rows
        .into_iter()
        .map(|expected| (replay(&expected.input), expected))
        .filter(|(actual, expected)| actual != expected)
        .collect();
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}
