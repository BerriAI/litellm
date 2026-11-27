//! Replays the rows and ledger columns `scripts/record_goldens.py` recorded from Python's own
//! `get_logging_payload` and `build_accounting_payload`, through the builder end to end.

use std::path::PathBuf;
use std::sync::Arc;

use litellm_callbacks_spend_log::{
    AccountingSavings, AccountingWindow, LedgerColumns, LedgerInput, RowConfig, RowInput,
    SpendLogRow, SpendLogRowBuilder,
};
use litellm_python_compat::json;
use litellm_redaction::TokenRedactor;
use rstest::rstest;
use serde::Deserialize;
use serde_json::Value;

const MINIMUM_CUSTOM_KEY_LENGTH: usize = 16;

/// What the recorder writes after the first character of each token-shaped string, so that no
/// committed golden matches a secret scanner.
const TOKEN_SPLIT: &str = "#~#";

/// A golden file as Python recorded it, with the recorder's token splits removed.
fn goldens(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .replace(TOKEN_SPLIT, "")
}

fn token_redactor() -> Arc<TokenRedactor> {
    let ranges: Vec<(u32, u32)> =
        serde_json::from_str(&goldens("token_word_ranges.json")).expect("word ranges parse");
    Arc::new(TokenRedactor::new(MINIMUM_CUSTOM_KEY_LENGTH, ranges))
}

/// The typed settings the recorder wrote, field for field.
#[derive(Deserialize)]
struct RecordedConfig {
    latest_turn_only: bool,
    redact_tokens: bool,
    drop_requester_ip_address: bool,
    omit_missing_session_id: bool,
    max_string_length_prompt_in_db: i64,
    disable_end_user_cost_tracking: bool,
    request_correlation_in_logs: bool,
    turn_off_message_logging: bool,
}

impl From<RecordedConfig> for RowConfig {
    fn from(config: RecordedConfig) -> Self {
        Self {
            latest_turn_only: config.latest_turn_only,
            redact_tokens: config.redact_tokens,
            drop_requester_ip_address: config.drop_requester_ip_address,
            omit_missing_session_id: config.omit_missing_session_id,
            max_string_length_prompt_in_db: config.max_string_length_prompt_in_db,
            disable_end_user_cost_tracking: config.disable_end_user_cost_tracking,
            request_correlation_in_logs: config.request_correlation_in_logs,
            turn_off_message_logging: config.turn_off_message_logging,
        }
    }
}

/// One `rows.jsonl` line. The input stays a Python-compatible value, since Python's JSON may
/// carry `NaN` where `serde_json` reads none.
struct RowRecord {
    case: String,
    config: Option<RecordedConfig>,
    input: Option<litellm_python_compat::Value>,
    output: Option<String>,
}

impl RowRecord {
    fn parse(line: &str) -> Self {
        let record = json::loads(line).expect("golden parses");
        let field = |name: &str| litellm_python_compat::dict::get(&record, name).cloned();
        Self {
            case: field("case")
                .map(|case| litellm_python_compat::repr::to_str(&case))
                .unwrap_or_default(),
            config: field("config").map(|config| {
                serde_json::from_value(json::to_json(&config).expect("config is JSON"))
                    .expect("config types")
            }),
            input: field("input"),
            output: match field("output") {
                Some(litellm_python_compat::Value::Str(output)) => Some(output),
                _ => None,
            },
        }
    }
}

#[derive(Deserialize)]
struct RecordedLedger {
    billable_cost: Option<f64>,
    delivery_id: String,
    entities: Vec<(String, Option<String>)>,
    model_access_groups: Vec<String>,
    windows: Vec<AccountingWindow>,
    savings: AccountingSavings,
}

#[derive(Deserialize)]
struct LedgerRecord {
    case: String,
    row: SpendLogRow,
    ledger: RecordedLedger,
    output: String,
}

/// A JSON value with every number read as a float, since Python's `0` and `0.0` are the same
/// amount to the accounting worker.
fn numbers_as_floats(value: Value) -> Value {
    match value {
        Value::Number(number) => serde_json::Number::from_f64(number.as_f64().unwrap_or_default())
            .map_or(Value::Null, Value::Number),
        Value::Array(items) => Value::Array(items.into_iter().map(numbers_as_floats).collect()),
        Value::Object(entries) => Value::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key, numbers_as_floats(value)))
                .collect(),
        ),
        other => other,
    }
}

fn mismatch_report(file: &str, total: usize, mismatches: &[String]) -> String {
    format!(
        "{} of {total} {file} goldens differ:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    )
}

fn row_mismatch(record: RowRecord, tokens: &Arc<TokenRedactor>) -> Option<String> {
    let (Some(config), Some(input), Some(expected)) = (record.config, record.input, record.output)
    else {
        return None;
    };
    let case = record.case;
    let builder = SpendLogRowBuilder::new(config.into(), Arc::clone(tokens));
    let mut row = match RowInput::from_value(&input).and_then(|input| builder.build_row(&input)) {
        Ok(row) => row,
        Err(error) => return Some(format!("{case}: {error}")),
    };
    // A NaN response cost is stored as is; JSON has no NaN, so both sides compare it as null.
    if row.spend.is_some_and(f64::is_nan) {
        row.spend = None;
    }
    let python: Value =
        serde_json::from_str(&expected.replace("\"spend\": NaN", "\"spend\": null"))
            .expect("python row parses");
    let rust = serde_json::to_value(&row).expect("row serializes");
    let same_columns = python
        .as_object()
        .map(|columns| columns.keys().collect::<Vec<_>>())
        == rust
            .as_object()
            .map(|columns| columns.keys().collect::<Vec<_>>());
    let typed: SpendLogRow = match serde_json::from_value(python) {
        Ok(typed) => typed,
        Err(error) => return Some(format!("{case}: python row does not type: {error}")),
    };
    (!same_columns || typed != row)
        .then(|| format!("{case}\n  python: {expected}\n  rust:   {rust}"))
}

fn ledger_mismatch(record: LedgerRecord) -> Option<String> {
    let case = record.case;
    let ledger = LedgerInput {
        billable_cost: record.ledger.billable_cost,
        org_id: None,
        team_id: None,
        delivery_id: record.ledger.delivery_id,
        entities: record.ledger.entities,
        model_access_groups: record.ledger.model_access_groups,
        windows: record.ledger.windows,
        savings: record.ledger.savings,
    };
    let columns = match record.row.ledger_columns(&ledger) {
        Ok(columns) => columns,
        Err(error) => return Some(format!("{case}: {error}")),
    };
    let python: LedgerColumns =
        serde_json::from_str(&record.output).expect("python ledger columns parse");
    let same = python.spend == columns.spend
        && python.accounting_delivery_id == columns.accounting_delivery_id
        && numbers_as_floats(python.accounting_payload.clone())
            == numbers_as_floats(columns.accounting_payload.clone());
    (!same).then(|| {
        format!(
            "{case}\n  python: {}\n  rust:   {}",
            record.output,
            serde_json::to_string(&columns).unwrap_or_default()
        )
    })
}

#[rstest]
fn rust_rows_match_every_python_row() {
    let tokens = token_redactor();
    let lines = goldens("rows.jsonl");
    let records: Vec<RowRecord> = lines.lines().map(RowRecord::parse).collect();
    let total = records
        .iter()
        .filter(|record| record.output.is_some())
        .count();
    assert!(total > 0, "rows.jsonl has no goldens");
    let mismatches: Vec<String> = records
        .into_iter()
        .filter_map(|record| row_mismatch(record, &tokens))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{}",
        mismatch_report("rows.jsonl", total, &mismatches)
    );
}

#[rstest]
fn rust_ledger_columns_match_every_python_ledger_row() {
    let lines = goldens("ledger_rows.jsonl");
    let records: Vec<LedgerRecord> = lines
        .lines()
        .map(|line| serde_json::from_str(line).expect("golden parses"))
        .collect();
    let total = records.len();
    assert!(total > 0, "ledger_rows.jsonl has no goldens");
    let mismatches: Vec<String> = records.into_iter().filter_map(ledger_mismatch).collect();
    assert!(
        mismatches.is_empty(),
        "{}",
        mismatch_report("ledger_rows.jsonl", total, &mismatches)
    );
}
