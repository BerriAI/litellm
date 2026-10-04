use std::future::Future;

use litellm_traces::store::{CallCursor, CallRow, SpanCursor, SpanRow};
use serde::Serialize;

use crate::{
    MAX_GRAPH_BYTES, MAX_GRAPH_SPANS,
    store::{StoreError, StoreResult},
};

pub(crate) const PAGE_SIZE: u32 = 256;

pub(crate) trait Keyed: Serialize {
    type Cursor: Clone;

    fn cursor(&self) -> Self::Cursor;
}

impl Keyed for SpanRow {
    type Cursor = SpanCursor;

    fn cursor(&self) -> SpanCursor {
        SpanRow::cursor(self)
    }
}

impl Keyed for CallRow {
    type Cursor = CallCursor;

    fn cursor(&self) -> CallCursor {
        CallRow::cursor(self)
    }
}

#[derive(Default)]
struct ReadBudget {
    bytes: usize,
    rows: usize,
}

impl ReadBudget {
    fn reserve(&mut self, bytes: usize) -> bool {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes > MAX_GRAPH_BYTES || self.rows == MAX_GRAPH_SPANS {
            return false;
        }
        self.rows += 1;
        true
    }
}

/// Reads every row by following the keyset. A page storage rejects as too large is retried at
/// half the size, and the smaller page is kept for the rest of the read because row sizes within
/// one graph rarely shrink again. The read stops with `TooLarge` once it passes the graph budget
/// or a single row exceeds the response limit.
pub(crate) async fn read_all<R, E, F, Fut>(read: F) -> StoreResult<Vec<R>, E>
where
    R: Keyed,
    F: Fn(Option<R::Cursor>, u32) -> Fut,
    Fut: Future<Output = StoreResult<Vec<R>, E>>,
{
    let mut after = None;
    let mut limit = PAGE_SIZE;
    let mut rows = Vec::new();
    let mut budget = ReadBudget::default();
    loop {
        let page = match read(after.clone(), limit).await {
            Err(StoreError::TooLarge) if limit > 1 => {
                limit /= 2;
                continue;
            }
            result => result?,
        };
        let complete = page.len() < limit as usize;
        for row in &page {
            let bytes = serde_json::to_vec(row).map_or(usize::MAX, |json| json.len());
            if !budget.reserve(bytes) {
                return Err(StoreError::TooLarge);
            }
        }
        if let Some(last) = page.last() {
            after = Some(last.cursor());
        }
        rows.extend(page);
        if complete {
            return Ok(rows);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::byte_boundary(MAX_GRAPH_BYTES - 1, 0, 1, true)]
    #[case::byte_overflow(MAX_GRAPH_BYTES - 1, 0, 2, false)]
    #[case::integer_overflow(MAX_GRAPH_BYTES, 0, usize::MAX, false)]
    #[case::row_boundary(0, MAX_GRAPH_SPANS - 1, 1, true)]
    #[case::row_overflow(0, MAX_GRAPH_SPANS, 1, false)]
    fn accumulation_stops_at_the_graph_budget(
        #[case] bytes: usize,
        #[case] rows: usize,
        #[case] next: usize,
        #[case] accepted: bool,
    ) {
        let mut budget = ReadBudget { bytes, rows };
        assert_eq!(budget.reserve(next), accepted);
    }

    #[derive(Serialize)]
    struct Number(u32);

    impl Keyed for Number {
        type Cursor = u32;

        fn cursor(&self) -> u32 {
            self.0
        }
    }

    /// A table of `total` rows whose transport rejects any page larger than `largest_page`.
    struct Table {
        total: u32,
        largest_page: u32,
        requests: Mutex<Vec<u32>>,
    }

    impl Table {
        async fn page(&self, after: Option<u32>, limit: u32) -> StoreResult<Vec<Number>, String> {
            self.requests.lock().unwrap().push(limit);
            if limit > self.largest_page {
                return Err(StoreError::TooLarge);
            }
            let after = after.unwrap_or(0);
            let end = (after + limit).min(self.total);
            Ok((after + 1..=end).map(Number).collect())
        }
    }

    #[rstest]
    #[case::fits(1000, PAGE_SIZE, &[256, 256, 256, 256])]
    #[case::uniform_large_rows(1000, 100, &[256, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64])]
    #[tokio::test]
    async fn a_rejected_page_size_is_not_retried(
        #[case] total: u32,
        #[case] largest_page: u32,
        #[case] requests: &[u32],
    ) {
        let table = Table {
            total,
            largest_page,
            requests: Mutex::new(Vec::new()),
        };
        let rows = read_all(|after, limit| table.page(after, limit))
            .await
            .unwrap();
        assert_eq!(
            rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            (1..=total).collect::<Vec<_>>()
        );
        assert_eq!(table.requests.lock().unwrap().as_slice(), requests);
    }

    #[rstest]
    #[tokio::test]
    async fn a_single_oversized_row_fails_the_read() {
        let table = Table {
            total: 10,
            largest_page: 0,
            requests: Mutex::new(Vec::new()),
        };
        let result = read_all(|after, limit| table.page(after, limit)).await;
        assert!(matches!(result, Err(StoreError::TooLarge)));
        assert_eq!(
            table.requests.lock().unwrap().as_slice(),
            &[256, 128, 64, 32, 16, 8, 4, 2, 1]
        );
    }
}
