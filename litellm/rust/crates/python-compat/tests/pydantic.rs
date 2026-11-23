//! Replays `generated/pydantic.json`, which pydantic wrote with `scripts/generate_pydantic_fixtures.py`.

use litellm_python_compat::{
    json,
    pydantic::{lax_bool, lax_int},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    input: String,
    bool: Option<bool>,
    int: Option<String>,
}

fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/pydantic.json"
    )))
    .expect("pydantic.json matches the fixture schema")
}

#[test]
fn lax_bool_and_int_match_pydantic() {
    let mismatches: Vec<String> = fixtures()
        .rows
        .iter()
        .flat_map(|row| {
            let value = json::loads(&row.input).expect("inputs are json.dumps output");
            let as_bool = lax_bool(&value);
            let as_int = lax_int(&value).map(|number| number.to_string());
            let input = shown(&row.input);
            [
                (as_bool != row.bool)
                    .then(|| format!("bool {input}: {as_bool:?} != {:?}", row.bool)),
                (as_int != row.int).then(|| {
                    format!(
                        "int {input}: {:?} != {:?}",
                        as_int.as_deref().map(shown),
                        row.int.as_deref().map(shown)
                    )
                }),
            ]
        })
        .flatten()
        .collect();
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// Long rows sit at pydantic's 4300-character cap, so a mismatch shows their length instead.
fn shown(text: &str) -> String {
    match text.len() {
        ..=40 => text.to_owned(),
        length => format!("{}... ({length} chars)", &text[..40]),
    }
}
