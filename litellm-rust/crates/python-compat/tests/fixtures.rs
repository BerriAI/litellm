//! Replays `generated/values.json`, which CPython wrote with `scripts/generate_fixtures.py`.

use std::{collections::BTreeMap, fs::File, io::Write};

use litellm_python_compat::{
    Value, json,
    literal::literal_eval,
    pickle,
    repr::{repr, to_str},
    truthy::truthy,
};
use rstest::{fixture, rstest};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    rows: Vec<Row>,
    sources: Vec<Source>,
}

/// `ast.literal_eval(source)`: the `repr` of its result, or the exception class it raised.
#[derive(Deserialize)]
struct Source {
    name: String,
    source: String,
    repr: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct Row {
    name: String,
    source: String,
    literal: bool,
    plain: bool,
    repr: String,
    str: String,
    truthy: bool,
    json: Option<String>,
    json_error: Option<String>,
    pickle: Option<BTreeMap<String, String>>,
    view: Option<String>,
}

/// Parsed once for the whole test binary.
#[fixture]
#[once]
fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/values.json"
    )))
    .expect("values.json matches the fixture schema")
}

/// Accepted differences from CPython: `(fixture source, check prefix, reason)`. Each entry
/// must still differ, so a dependency fix that removes one fails the test until it is deleted.
const KNOWN: &[(&str, &str, &str)] = &[
    (
        "...",
        "literal_eval source",
        "`Ellipsis` is not part of the data model",
    ),
    (
        r"'\ud800'",
        "literal_eval source",
        "a Rust `String` cannot hold a lone surrogate",
    ),
    (
        r"'\N{BULLET}'",
        "literal_eval source",
        "`\\N{NAME}` needs the Unicode name table",
    ),
    (
        "nested_150",
        "literal_eval",
        "deeper than MAX_DEPTH: rejected for stack safety, where CPython still parses it",
    ),
    (
        "nested_150",
        "pickle.loads",
        "deeper than MAX_DEPTH: rejected for stack safety, where CPython has no limit",
    ),
    (
        "2**64",
        "pickle.loads",
        "serde-pickle's serde interface stops at i64",
    ),
    (
        "-(2**70)",
        "pickle.loads",
        "serde-pickle's serde interface stops at i64",
    ),
    (
        "2**64",
        "pickle.dumps",
        "serde-pickle's serde interface stops at i64",
    ),
    (
        "-(2**70)",
        "pickle.dumps",
        "serde-pickle's serde interface stops at i64",
    ),
    (
        "2**64",
        "to_json",
        "serde_json has no exact form for integers beyond u64",
    ),
    (
        "-(2**70)",
        "to_json",
        "serde_json has no exact form for integers beyond i64",
    ),
    (
        "b''",
        "pickle.loads protocol 0",
        "protocols 0-2 pickle `b''` as a `bytes()` call",
    ),
    (
        "b''",
        "pickle.loads protocol 1",
        "protocols 0-2 pickle `b''` as a `bytes()` call",
    ),
    (
        "b''",
        "pickle.loads protocol 2",
        "protocols 0-2 pickle `b''` as a `bytes()` call",
    ),
];

/// Collects every mismatch so one run reports the whole divergence set.
#[derive(Default)]
struct Mismatches {
    unexpected: Vec<String>,
    known_seen: Vec<usize>,
    known_scope: Vec<usize>,
}

impl Mismatches {
    fn known(source: &str, what: &str) -> Option<usize> {
        KNOWN
            .iter()
            .position(|(known, prefix, _)| *known == source && what.starts_with(prefix))
    }

    fn check(&mut self, row: &Row, what: &str, expected: &str, actual: &str) {
        self.check_source(&row.name, what, expected, actual);
    }

    /// `name` identifies the fixture row in reports and in [`KNOWN`].
    fn check_source(&mut self, name: &str, what: &str, expected: &str, actual: &str) {
        let known = Self::known(name, what);
        if let Some(index) = known {
            self.known_scope.push(index);
        }
        if expected == actual {
            return;
        }
        match known {
            Some(index) => self.known_seen.push(index),
            None => self.unexpected.push(format!(
                "{name:?} [{what}]\n  python: {expected}\n  rust:   {actual}"
            )),
        }
    }

    fn finish(self) {
        let resolved: Vec<_> = self
            .known_scope
            .iter()
            .filter(|index| !self.known_seen.contains(index))
            .map(|&index| format!("{} [{}]", KNOWN[index].0, KNOWN[index].1))
            .collect();
        assert!(
            self.unexpected.is_empty() && resolved.is_empty(),
            "{} mismatches with CPython:\n{}\nknown divergences that now match (delete them \
             from KNOWN): {resolved:?}",
            self.unexpected.len(),
            self.unexpected.join("\n"),
        );
    }
}

fn check_value(mismatches: &mut Mismatches, row: &Row, value: &Value) {
    mismatches.check(row, "repr", &row.repr, &repr(value));
    mismatches.check(row, "str", &row.str, &to_str(value));
    mismatches.check(
        row,
        "bool",
        &row.truthy.to_string(),
        &truthy(value).to_string(),
    );
    let expected = row.json.clone().or_else(|| {
        row.json_error
            .clone()
            .map(|error| format!("error: {error}"))
    });
    let actual = match json::dumps(value) {
        Ok(text) => text,
        Err(error) => format!("error: {error}"),
    };
    mismatches.check(
        row,
        "json.dumps",
        expected.as_deref().unwrap_or(""),
        &actual,
    );
}

#[rstest]
fn literal_rows_match_python_repr_str_bool_and_json(fixtures: &Fixtures) {
    let mut mismatches = Mismatches::default();
    for row in fixtures.rows.iter().filter(|row| row.literal) {
        match literal_eval(&row.repr) {
            Ok(value) => check_value(&mut mismatches, row, &value),
            Err(error) => mismatches.check(row, "literal_eval", &row.repr, &error.to_string()),
        }
    }
    mismatches.finish();
}

/// Errors compare by outcome only: CPython's exception class is not part of the contract.
#[rstest]
fn literal_eval_matches_python_on_source_texts(fixtures: &Fixtures) {
    let mut mismatches = Mismatches::default();
    for case in &fixtures.sources {
        let expected = match (&case.repr, &case.error) {
            (Some(repr), None) => repr.clone(),
            (None, Some(_)) => "an error".to_owned(),
            _ => panic!("{:?}: a source records a repr or an error", case.source),
        };
        let actual = match literal_eval(&case.source) {
            Ok(value) => repr(&value),
            Err(_) => "an error".to_owned(),
        };
        mismatches.check_source(&case.name, "literal_eval source", &expected, &actual);
    }
    mismatches.finish();
}

#[rstest]
fn pickle_loads_matches_python_at_every_protocol(fixtures: &Fixtures) {
    let mut mismatches = Mismatches::default();
    for row in &fixtures.rows {
        let Some(pickles) = &row.pickle else { continue };
        for (protocol, data) in pickles {
            let data = hex::decode(data).expect("fixture pickle is hex");
            let what = format!("pickle.loads protocol {protocol}");
            match (pickle::loads(&data), row.plain) {
                (Ok(value), true) => {
                    let view = row.view.as_deref().expect("picklable rows have a view");
                    mismatches.check(row, &what, view, &repr(&value));
                }
                (Err(error), true) => {
                    mismatches.check(row, &what, "a value", &format!("error: {error}"))
                }
                (Ok(value), false) => {
                    mismatches.check(row, &what, "a class-reference error", &repr(&value))
                }
                (Err(pickle_error), false) => assert!(
                    matches!(pickle_error, litellm_python_compat::Error::InvalidPickle(_)),
                    "{}: {pickle_error}",
                    row.source
                ),
            }
        }
    }
    mismatches.finish();
}

/// Non-finite floats have no literal form; pickle is how Rust receives them.
#[rstest]
fn values_reached_only_through_pickle_match_python(fixtures: &Fixtures) {
    let mut mismatches = Mismatches::default();
    for row in fixtures
        .rows
        .iter()
        .filter(|row| !row.literal && row.plain && row.view.as_deref() == Some(&row.repr))
    {
        let data = hex::decode(&row.pickle.as_ref().expect("plain rows pickle")["5"])
            .expect("fixture pickle is hex");
        let value = pickle::loads(&data).expect("plain pickle decodes");
        check_value(&mut mismatches, row, &value);
    }
    mismatches.finish();
}

/// Byte equality with CPython is not the contract: CPython adds memo opcodes and picks the
/// smallest integer opcode. `scripts/verify_rust_pickles.py` checks that CPython reads these
/// back; set `PYTHON_COMPAT_RUST_PICKLES` to a file path to export them.
#[rstest]
fn pickle_dumps_round_trips_every_plain_literal(fixtures: &Fixtures) {
    // Truncate up front: the verifier must read this run's rows and nothing else.
    let mut export = std::env::var_os("PYTHON_COMPAT_RUST_PICKLES")
        .map(|path| File::create(path).expect("the export path is writable"));
    let mut mismatches = Mismatches::default();
    for row in fixtures
        .rows
        .iter()
        .filter(|row| row.literal && row.plain && row.view.as_deref() == Some(&row.repr))
    {
        // Rows past `MAX_DEPTH` are covered by the literal test's own KNOWN entry.
        let Ok(value) = literal_eval(&row.repr) else {
            continue;
        };
        let data = match pickle::dumps(&value) {
            Ok(data) => data,
            Err(error) => {
                mismatches.check(row, "pickle.dumps", "a pickle", &format!("error: {error}"));
                continue;
            }
        };
        let decoded = pickle::loads(&data).expect("rust pickle decodes");
        mismatches.check(row, "pickle round trip", &row.repr, &repr(&decoded));
        if let Some(file) = &mut export {
            writeln!(file, "{}\t{}", hex::encode(&data), repr(&value))
                .expect("the export file is writable");
        }
    }
    mismatches.finish();
}

#[rstest]
fn to_json_matches_python_json_round_trip(fixtures: &Fixtures) {
    let mut mismatches = Mismatches::default();
    for row in fixtures.rows.iter().filter(|row| row.literal) {
        let Some(expected) = &row.json else { continue };
        let Ok(value) = literal_eval(&row.repr) else {
            continue;
        };
        let expected: serde_json::Value =
            serde_json::from_str(expected).expect("python json.dumps output parses");
        match json::to_json(&value) {
            Ok(actual) => {
                mismatches.check(row, "to_json", &expected.to_string(), &actual.to_string())
            }
            Err(error) => {
                mismatches.check(row, "to_json", &expected.to_string(), &error.to_string())
            }
        }
    }
    mismatches.finish();
}
