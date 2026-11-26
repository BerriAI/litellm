//! Replays `generated/pydantic.json`, which pydantic wrote with `scripts/generate_pydantic_fixtures.py`.

use litellm_python_compat::{
    datetime::IsoDateTime,
    json,
    pydantic::{lax_bool, lax_datetime, lax_int},
};
use rstest::rstest;
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

#[rstest]
#[case::naive(r#""2026-12-01T00:00:00""#, Some("2026-12-01T00:00:00"))]
#[case::naive_with_a_fraction(r#""2026-12-01T00:00:00.5""#, Some("2026-12-01T00:00:00.500000"))]
#[case::a_space_for_the_t(r#""2026-12-01 00:00:00""#, Some("2026-12-01T00:00:00"))]
#[case::a_date_alone(r#""2026-12-01""#, Some("2026-12-01T00:00:00"))]
#[case::ahead_of_utc(r#""2026-12-01T00:00:00+05:30""#, Some("2026-12-01T00:00:00+05:30"))]
#[case::behind_utc(r#""2026-12-01T00:00:00-08:00""#, Some("2026-12-01T00:00:00-08:00"))]
#[case::a_zero_offset(r#""2026-12-01T00:00:00+00:00""#, Some("2026-12-01T00:00:00Z"))]
#[case::a_z(r#""2026-12-01T00:00:00Z""#, Some("2026-12-01T00:00:00Z"))]
#[case::epoch_seconds("1764547200", Some("2025-12-01T00:00:00Z"))]
#[case::epoch_seconds_with_a_fraction("1764547200.5", Some("2025-12-01T00:00:00.500000Z"))]
#[case::before_the_epoch("-5", Some("1969-12-31T23:59:55Z"))]
#[case::epoch_milliseconds("20000000001", Some("1970-08-20T11:33:20.001000Z"))]
#[case::epoch_seconds_as_text(r#""1764547200""#, Some("2025-12-01T00:00:00Z"))]
#[case::the_last_millisecond_of_year_9999("253402300799999", Some("9999-12-31T23:59:59.999000Z"))]
#[case::year_10000("253402300800000", None)]
#[case::the_first_millisecond_of_year_1("-62135596800000", Some("0001-01-01T00:00:00Z"))]
#[case::year_0("-62135596801000", None)]
#[case::a_bool("true", None)]
#[case::no_time(r#""soon""#, None)]
fn lax_datetime_reads_and_writes_what_pydantic_does(
    #[case] input: &str,
    #[case] dumped: Option<&str>,
) {
    assert_eq!(
        lax_datetime(&json::loads(input).unwrap()).map(IsoDateTime::isoformat),
        dumped.map(str::to_owned)
    );
}
