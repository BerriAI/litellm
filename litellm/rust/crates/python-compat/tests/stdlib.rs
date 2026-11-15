//! Replays `generated/stdlib.json`, which CPython wrote with `scripts/generate_stdlib_fixtures.py`.

use litellm_python_compat::{datetime::UtcDateTime, urllib::quote};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    quote: Vec<Quote>,
    utc: Vec<Utc>,
}

#[derive(Deserialize)]
struct Quote {
    text: String,
    safe: String,
    quoted: String,
}

#[derive(Deserialize)]
struct Utc {
    seconds: i64,
    micros: u32,
    fields: [i64; 7],
}

fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/stdlib.json"
    )))
    .expect("stdlib.json matches the fixture schema")
}

#[test]
fn quote_matches_urllib() {
    for row in fixtures().quote {
        assert_eq!(
            quote(&row.text, &row.safe),
            row.quoted,
            "{:?} safe={:?}",
            row.text,
            row.safe
        );
    }
}

#[test]
fn utc_fields_match_datetime() {
    for row in fixtures().utc {
        let moment = UtcDateTime::from_timestamp(row.seconds, row.micros);
        let fields = [
            moment.year,
            i64::from(moment.month),
            i64::from(moment.day),
            i64::from(moment.hour),
            i64::from(moment.minute),
            i64::from(moment.second),
            i64::from(moment.microsecond),
        ];
        assert_eq!(fields, row.fields, "{} {}", row.seconds, row.micros);
    }
}
