//! The byte-bounded spend log queue, which keeps a long database outage from growing the proxy's
//! memory without bound. It ports `enqueue_spend_logs` and `dequeue_spend_logs`.

use std::{collections::VecDeque, sync::Mutex};

use crate::QueuedSpendLog;

#[derive(Debug, Default)]
struct State {
    rows: VecDeque<QueuedSpendLog>,
    bytes: usize,
}

/// The in-memory spend log queue, bounded by the bytes its rows dump to.
#[derive(Debug, Default)]
pub struct SpendLogQueue {
    state: Mutex<State>,
}

impl SpendLogQueue {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Adds `rows` at the head (a replay of rows the database refused) or the tail, then drops
    /// the oldest rows until the queue fits `max_bytes`, always keeping the newest row. A dropped
    /// row that carries a charge comes back at the head as its accounting-only row, so the
    /// request is still charged. Returns how many rows were dropped.
    pub fn enqueue(&self, rows: Vec<QueuedSpendLog>, at_head: bool, max_bytes: usize) -> usize {
        let mut state = self.state();
        state.bytes += rows.iter().map(QueuedSpendLog::bytes).sum::<usize>();
        if at_head {
            for row in rows.into_iter().rev() {
                state.rows.push_front(row);
            }
        } else {
            state.rows.extend(rows);
        }
        let mut dropped = 0;
        let mut rescued = Vec::new();
        while state.bytes > max_bytes && state.rows.len() > 1 {
            let Some(oldest) = state.rows.pop_front() else {
                break;
            };
            state.bytes -= oldest.bytes();
            dropped += 1;
            rescued.extend(oldest.accounting_only());
        }
        for charge in rescued.into_iter().rev() {
            state.bytes += charge.bytes();
            state.rows.push_front(charge);
        }
        drop(state);
        if dropped > 0 {
            tracing::error!(
                "Spend tracking - spend log queue is at its {} byte budget; dropped the {} oldest spend logs",
                max_bytes,
                dropped
            );
        }
        dropped
    }

    /// Takes up to `limit` rows off the head.
    pub fn dequeue(&self, limit: usize) -> Vec<QueuedSpendLog> {
        let mut state = self.state();
        let take = limit.min(state.rows.len());
        let popped: Vec<QueuedSpendLog> = state.rows.drain(..take).collect();
        state.bytes -= popped.iter().map(QueuedSpendLog::bytes).sum::<usize>();
        popped
    }

    pub fn len(&self) -> usize {
        self.state().rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        self.state().bytes
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::SpendLogQueue;
    use crate::{QueuedSpendLog, test_support::python_row};

    fn row(request_id: &str, charged: bool) -> QueuedSpendLog {
        let ledger = if charged {
            r#", "accounting_payload": {"spend": 1.0}"#
        } else {
            ""
        };
        QueuedSpendLog::from_python(&python_row(request_id, ledger)).expect("row is queued")
    }

    fn ids(rows: &[QueuedSpendLog]) -> Vec<(String, bool)> {
        rows.iter()
            .map(|row| (row.request_id().to_owned(), row.row().is_some()))
            .collect()
    }

    #[rstest]
    #[case::appends(false, ["old", "new"])]
    #[case::replays_at_head(true, ["new", "old"])]
    fn keeps_order_within_budget(#[case] at_head: bool, #[case] expected: [&str; 2]) {
        let queue = SpendLogQueue::default();
        queue.enqueue(vec![row("old", false)], false, usize::MAX);

        assert_eq!(
            queue.enqueue(vec![row("new", false)], at_head, usize::MAX),
            0
        );
        assert_eq!(
            ids(&queue.dequeue(10)),
            expected.map(|id| (id.to_owned(), true))
        );
        assert_eq!(queue.bytes(), 0);
    }

    #[rstest]
    #[case::drops_oldest_log_only(false, vec![("c", true)])]
    #[case::keeps_the_charge_of_a_dropped_row(true, vec![("a", false), ("c", true)])]
    fn past_the_budget_drops_the_oldest(
        #[case] charged: bool,
        #[case] expected: Vec<(&str, bool)>,
    ) {
        let queue = SpendLogQueue::default();
        let budget = row("c", false).bytes() + row("b", false).bytes() / 2;
        queue.enqueue(vec![row("a", charged), row("b", false)], false, usize::MAX);

        assert_eq!(queue.enqueue(vec![row("c", false)], false, budget), 2);
        let kept = queue.dequeue(10);
        assert_eq!(
            ids(&kept),
            expected
                .iter()
                .map(|(id, logged)| ((*id).to_owned(), *logged))
                .collect::<Vec<_>>()
        );
        assert_eq!(queue.bytes(), 0);
    }

    #[rstest]
    fn keeps_the_newest_row_whatever_it_costs() {
        let queue = SpendLogQueue::default();

        assert_eq!(queue.enqueue(vec![row("a", false)], false, 1), 0);
        assert_eq!(queue.len(), 1);
    }

    #[rstest]
    #[case::partial(1, 1, 1)]
    #[case::everything(5, 2, 0)]
    fn dequeues_the_oldest_rows(#[case] limit: usize, #[case] popped: usize, #[case] left: usize) {
        let queue = SpendLogQueue::default();
        queue.enqueue(vec![row("a", false), row("b", false)], false, usize::MAX);

        assert_eq!(queue.dequeue(limit).len(), popped);
        assert_eq!(queue.len(), left);
        assert_eq!(queue.bytes(), left * row("b", false).bytes());
    }
}
