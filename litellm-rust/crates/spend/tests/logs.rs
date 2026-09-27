use std::sync::Mutex;

use litellm_spend::{
    InsertError, LogFlushOutcome, LogQueue, LogSink, Outcome, SerializedJson, SpendLogRow,
    flush_logs,
};
use rstest::rstest;
use time::macros::datetime;

#[derive(Debug, thiserror::Error)]
#[error("injected failure")]
struct Injected;

fn row(request_id: &str, response_bytes: usize) -> SpendLogRow {
    SpendLogRow {
        request_id: request_id.to_owned(),
        call_type: "acompletion".to_owned(),
        api_key: "hashed".to_owned(),
        spend: 0.25,
        total_tokens: 30,
        prompt_tokens: 10,
        completion_tokens: 20,
        start_time: datetime!(2026-09-26 12:00 UTC),
        end_time: datetime!(2026-09-26 12:00:01 UTC),
        completion_start_time: None,
        request_duration_ms: Some(1_000),
        model: "model-a".to_owned(),
        model_id: None,
        model_group: None,
        custom_llm_provider: None,
        api_base: None,
        user: None,
        team_id: None,
        organization_id: None,
        end_user: None,
        agent_id: None,
        requester_ip_address: None,
        session_id: None,
        litellm_call_id: None,
        mcp_namespaced_tool_name: None,
        cache_hit: None,
        cache_key: None,
        request_tags: Vec::new(),
        outcome: Outcome::Success,
        metadata: SerializedJson::default(),
        messages: SerializedJson::default(),
        response: SerializedJson("x".repeat(response_bytes)),
        proxy_server_request: SerializedJson::default(),
    }
}

fn ids(rows: &[SpendLogRow]) -> Vec<&str> {
    rows.iter().map(|row| row.request_id.as_str()).collect()
}

struct Sink {
    poison: Vec<&'static str>,
    down: bool,
    inserted: Mutex<Vec<String>>,
}

impl Sink {
    fn new(poison: &[&'static str], down: bool) -> Self {
        Self {
            poison: poison.to_vec(),
            down,
            inserted: Mutex::default(),
        }
    }

    fn inserted(&self) -> Vec<String> {
        self.inserted.lock().unwrap().clone()
    }
}

impl LogSink for Sink {
    type Error = Injected;

    async fn insert(&self, rows: &[SpendLogRow]) -> Result<(), InsertError<Injected>> {
        if self.down {
            return Err(InsertError::Transient(Injected));
        }
        if rows
            .iter()
            .any(|row| self.poison.contains(&row.request_id.as_str()))
        {
            return Err(InsertError::Rejected(Injected));
        }
        self.inserted
            .lock()
            .unwrap()
            .extend(rows.iter().map(|row| row.request_id.clone()));
        Ok(())
    }
}

#[rstest]
#[case::under_budget(&[10, 10, 10], 100, 0, &["r0", "r1", "r2"])]
#[case::oldest_rows_go_first(&[40, 40, 40], 100, 1, &["r1", "r2"])]
#[case::a_row_larger_than_the_budget_is_still_kept(&[10, 500], 100, 1, &["r1"])]
fn the_queue_stays_within_its_byte_budget_by_evicting_the_oldest_rows(
    #[case] response_bytes: &[usize],
    #[case] max_bytes: usize,
    #[case] evicted: usize,
    #[case] kept: &[&str],
) {
    let queue = LogQueue::new(max_bytes);

    let total_evicted: usize = response_bytes
        .iter()
        .enumerate()
        .map(|(i, bytes)| queue.enqueue(row(&format!("r{i}"), *bytes)))
        .sum();

    assert_eq!(total_evicted, evicted);
    assert_eq!(ids(&queue.dequeue(usize::MAX)), kept);
}

#[test]
fn dequeued_bytes_are_given_back_to_the_budget() {
    let queue = LogQueue::new(100);
    queue.enqueue(row("r0", 60));
    queue.dequeue(1);

    assert_eq!(queue.enqueue(row("r1", 60)), 0);
}

#[tokio::test]
async fn a_rejected_batch_is_bisected_until_only_the_poison_rows_are_dropped() {
    let queue = LogQueue::new(usize::MAX);
    for i in 0..8 {
        queue.enqueue(row(&format!("r{i}"), 10));
    }
    let sink = Sink::new(&["r2", "r5"], false);

    let outcome = flush_logs(&queue, &sink, 100).await.unwrap();

    assert_eq!(
        outcome,
        LogFlushOutcome {
            inserted: 6,
            rejected: vec!["r2".to_owned(), "r5".to_owned()],
        }
    );
    assert_eq!(sink.inserted(), ["r0", "r1", "r3", "r4", "r6", "r7"]);
    assert!(queue.is_empty());
}

#[tokio::test]
async fn a_transient_failure_puts_the_rows_back_ahead_of_newer_ones_in_order() {
    let queue = LogQueue::new(usize::MAX);
    for i in 0..3 {
        queue.enqueue(row(&format!("r{i}"), 10));
    }

    let failure = flush_logs(&queue, &Sink::new(&[], true), 2)
        .await
        .unwrap_err();

    assert_eq!((failure.requeued, failure.evicted), (2, 0));
    assert_eq!(ids(&queue.dequeue(usize::MAX)), ["r0", "r1", "r2"]);
}

#[tokio::test]
async fn at_most_max_rows_leave_the_queue_per_flush() {
    let queue = LogQueue::new(usize::MAX);
    for i in 0..3 {
        queue.enqueue(row(&format!("r{i}"), 10));
    }
    let sink = Sink::new(&[], false);

    let outcome = flush_logs(&queue, &sink, 2).await.unwrap();

    assert_eq!(outcome.inserted, 2);
    assert_eq!(ids(&queue.dequeue(usize::MAX)), ["r2"]);
}
