//! Replays `generated/stdlib.json`, which CPython wrote with `scripts/generate_stdlib_fixtures.py`.

use litellm_python_compat::{
    datetime::{IsoDateTime, UtcDateTime, fromisoformat},
    format::fixed,
    number::{self, PyNumber},
    unicode::isspace,
    urllib::quote,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    quote: Vec<Quote>,
    utc: Vec<Utc>,
    fromisoformat: Vec<FromIsoFormat>,
    fixed: Vec<Fixed>,
    number_str: Vec<NumberStr>,
    isspace: Vec<u32>,
}

#[derive(Deserialize)]
struct FromIsoFormat {
    text: String,
    naive: Option<i64>,
    aware: Option<i64>,
}

/// A finite `float`, or the `repr` of a non-finite one, which JSON cannot hold.
#[derive(Deserialize)]
#[serde(untagged)]
enum Float {
    Finite(f64),
    Repr(String),
}

impl Float {
    fn value(&self) -> f64 {
        match self {
            Self::Finite(value) => *value,
            Self::Repr(text) => number::float(text).expect("a float repr"),
        }
    }
}

#[derive(Deserialize)]
struct Fixed {
    value: Float,
    precision: usize,
    text: String,
}

#[derive(Deserialize)]
struct NumberStr {
    value: Option<PyNumber>,
    float: Option<String>,
    str: String,
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

#[test]
fn a_clock_reading_has_the_fields_of_its_timestamp() {
    for row in fixtures().utc {
        let offset = std::time::Duration::from_secs(row.seconds.unsigned_abs());
        let whole = if row.seconds < 0 {
            std::time::UNIX_EPOCH - offset
        } else {
            std::time::UNIX_EPOCH + offset
        };
        let at = whole
            + std::time::Duration::from_micros(u64::from(row.micros))
            + std::time::Duration::from_nanos(999);
        assert_eq!(
            UtcDateTime::from_system_time(at),
            UtcDateTime::from_timestamp(row.seconds, row.micros),
            "{} {}",
            row.seconds,
            row.micros
        );
    }
}

#[test]
fn fromisoformat_matches_datetime() {
    for row in fixtures().fromisoformat {
        let expected = match (row.naive, row.aware) {
            (Some(micros), None) => Some(IsoDateTime::Naive(micros)),
            (None, Some(micros)) => Some(IsoDateTime::Aware(micros)),
            _ => None,
        };
        assert_eq!(fromisoformat(&row.text), expected, "{:?}", row.text);
    }
}

#[test]
fn fixed_matches_format() {
    for row in fixtures().fixed {
        let value = row.value.value();
        assert_eq!(
            fixed(value, row.precision),
            row.text,
            "{value:?} .{}f",
            row.precision
        );
    }
}

#[test]
fn number_text_matches_str() {
    for row in fixtures().number_str {
        let number = match (row.value, row.float) {
            (Some(number), None) => number,
            (None, Some(text)) => PyNumber::Float(number::float(&text).expect("a float repr")),
            _ => panic!("a number_str row holds a value or a float repr"),
        };
        assert_eq!(number.text(), row.str);
    }
}

#[test]
fn isspace_matches_str_isspace() {
    let ours: Vec<u32> = (0..=0x0010_ffff_u32)
        .filter_map(char::from_u32)
        .filter(|c| isspace(*c))
        .map(u32::from)
        .collect();
    assert_eq!(ours, fixtures().isspace);
}
