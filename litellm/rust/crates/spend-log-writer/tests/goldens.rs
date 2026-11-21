//! Replays `tests/goldens/queue.json`, recorded from the Python queue by
//! `scripts/record_goldens.py`: the bytes each row costs, for a row Python queued and for the same
//! row built in Rust, and the queue an overflow leaves.

use litellm_callbacks_spend_log::{LedgerColumns, SpendLogRow};
use litellm_spend_log_writer::{QueuedSpendLog, SpendLogQueue};
use rstest::{fixture, rstest};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Measured {
    row: String,
    bytes: usize,
    accounting_only_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct Queued {
    request_id: String,
    accounting_log_disabled: Option<bool>,
}

#[derive(Deserialize)]
struct Overflow {
    budget: usize,
    enqueued: Vec<String>,
    replayed_at_head: String,
    queue: Vec<Queued>,
}

#[derive(Deserialize)]
struct Golden {
    measured: Vec<Measured>,
    overflow: Overflow,
}

#[fixture]
fn golden() -> Golden {
    serde_json::from_str(include_str!("goldens/queue.json")).expect("golden parses")
}

/// The row as the Rust callback would build it: the builder's row and the ledger's columns.
fn built_in_rust(text: &str) -> QueuedSpendLog {
    let Value::Object(mut fields) = serde_json::from_str(text).expect("row parses") else {
        panic!("row is an object");
    };
    fields.remove("accounting_log_disabled");
    let delivery_id = fields.remove("accounting_delivery_id");
    let ledger = fields
        .remove("accounting_payload")
        .map(|payload| LedgerColumns {
            spend: fields["spend"].as_f64().expect("spend"),
            accounting_delivery_id: delivery_id.and_then(|id| id.as_str().map(str::to_owned)),
            accounting_payload: payload,
        });
    let row: SpendLogRow = serde_json::from_value(Value::Object(fields)).expect("row fits");
    QueuedSpendLog::new(row, ledger, true).expect("entry builds")
}

#[rstest]
fn rows_cost_what_python_measures(golden: Golden) {
    for measured in golden.measured {
        let queued = QueuedSpendLog::from_python(&measured.row).expect("row is queued");
        let built = built_in_rust(&measured.row);

        assert_eq!(queued.bytes(), measured.bytes, "{}", measured.row);
        assert_eq!(built.bytes(), measured.bytes, "{}", measured.row);
        assert_eq!(
            built.accounting_only().map(|charge| charge.bytes()),
            measured.accounting_only_bytes,
            "{}",
            measured.row
        );
    }
}

#[rstest]
fn an_overflow_leaves_the_queue_python_leaves(golden: Golden) {
    let overflow = golden.overflow;
    let queue = SpendLogQueue::default();
    for text in &overflow.enqueued {
        queue.enqueue(
            vec![QueuedSpendLog::from_python(text).expect("row is queued")],
            false,
            overflow.budget,
        );
    }
    queue.enqueue(
        vec![QueuedSpendLog::from_python(&overflow.replayed_at_head).expect("row is queued")],
        true,
        overflow.budget,
    );

    let left: Vec<(String, Option<bool>)> = queue
        .dequeue(usize::MAX)
        .iter()
        .map(|entry| {
            let disabled = match (entry.row(), entry.charge()) {
                (None, _) => Some(true),
                (Some(_), Some(_)) => Some(false),
                (Some(_), None) => None,
            };
            (entry.request_id().to_owned(), disabled)
        })
        .collect();
    let expected: Vec<(String, Option<bool>)> = overflow
        .queue
        .into_iter()
        .map(|queued| (queued.request_id, queued.accounting_log_disabled))
        .collect();
    assert_eq!(left, expected);
}
