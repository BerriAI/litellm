//! The gateway request repository: adds the request counts the gateway folds at its edge onto
//! `LiteLLM_DailyGatewayRequests`, and the failed ones by status code onto
//! `LiteLLM_DailyGatewayFailedRequests`, the tables the admin UI reads successful gateway
//! requests from.
//!
//! The policy mirrors `build_gateway_requests_upsert`: a flush is one statement, the totals sum
//! every status code of a route, only a status code with failures gets a failure row, and
//! both sets of rows are written in the order of their conflict keys, the order Python sorts
//! them in, so a Rust and a Python writer lock shared rows in the same order and cannot deadlock.

use std::collections::BTreeMap;

use litellm_core_utils::billable_route::Category;
use litellm_db::Call;

use super::queries::{self, FailedColumns, TotalColumns};
use crate::{Database, Side};

const COMMIT_GATEWAY_REQUESTS: Call = Call {
    call_type: "commit_gateway_requests",
    table: Some("LiteLLM_DailyGatewayRequests"),
    operation: None,
};

/// The requests to one route of one category on one UTC date that answered `status_code`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatewayRequestCount {
    pub date: String,
    pub category: Category,
    pub route: String,
    pub status_code: i32,
    pub successful_requests: i64,
    pub failed_requests: i64,
}

/// Writes `LiteLLM_DailyGatewayRequests` and `LiteLLM_DailyGatewayFailedRequests`.
pub struct GatewayRequestRepository {
    database: Database,
}

impl GatewayRequestRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Adds `counts` onto the stored rows in one statement on the writer. No counts is no round
    /// trip.
    pub async fn increment(&self, counts: &[GatewayRequestCount]) -> Result<(), litellm_db::Error> {
        if counts.is_empty() {
            return Ok(());
        }
        let (failed, totals) = (failed_columns(counts), total_columns(counts));
        self.database
            .run(COMMIT_GATEWAY_REQUESTS, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::increment_gateway_requests(session.executor(), &failed, &totals).await
            })
            .await
    }
}

fn total_columns(counts: &[GatewayRequestCount]) -> TotalColumns {
    let totals = counts.iter().fold(BTreeMap::new(), |mut totals, count| {
        let (successful, failed) = totals
            .entry((&count.date, count.category, &count.route))
            .or_insert((0_i64, 0_i64));
        *successful += count.successful_requests;
        *failed += count.failed_requests;
        totals
    });
    totals.into_iter().fold(
        TotalColumns::default(),
        |mut columns, ((date, category, route), (successful, failed))| {
            columns.dates.push(date.clone());
            columns.categories.push(<&str>::from(category).to_owned());
            columns.routes.push(route.clone());
            columns.successful_requests.push(successful);
            columns.failed_requests.push(failed);
            columns
        },
    )
}

fn failed_columns(counts: &[GatewayRequestCount]) -> FailedColumns {
    let mut failed: Vec<(&String, Category, &String, i32, i64)> = counts
        .iter()
        .filter(|count| count.failed_requests > 0)
        .map(|count| {
            (
                &count.date,
                count.category,
                &count.route,
                count.status_code,
                count.failed_requests,
            )
        })
        .collect();
    failed.sort_by(|left, right| {
        (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
    });
    failed.into_iter().fold(
        FailedColumns::default(),
        |mut columns, (date, category, route, status_code, count)| {
            columns.dates.push(date.clone());
            columns.categories.push(<&str>::from(category).to_owned());
            columns.routes.push(route.clone());
            columns.status_codes.push(status_code);
            columns.failed_requests.push(count);
            columns
        },
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn count(route: &str, status_code: i32, successful: i64, failed: i64) -> GatewayRequestCount {
        GatewayRequestCount {
            date: "2026-11-23".to_owned(),
            category: Category::Llm,
            route: route.to_owned(),
            status_code,
            successful_requests: successful,
            failed_requests: failed,
        }
    }

    #[rstest]
    fn totals_sum_every_status_code_of_a_route_in_key_order() {
        let counts = [
            count("/v1/messages", 200, 3, 0),
            count("/ocr", 200, 1, 0),
            count("/v1/messages", 500, 0, 2),
            count("/v1/messages", 429, 1, 1),
        ];

        assert_eq!(
            total_columns(&counts),
            TotalColumns {
                dates: vec!["2026-11-23".to_owned(); 2],
                categories: vec!["llm".to_owned(); 2],
                routes: vec!["/ocr".to_owned(), "/v1/messages".to_owned()],
                successful_requests: vec![1, 4],
                failed_requests: vec![0, 3],
            }
        );
    }

    #[rstest]
    fn only_a_status_code_with_failures_gets_a_failure_row_in_key_order() {
        let counts = [
            count("/v1/messages", 500, 0, 2),
            count("/v1/messages", 200, 3, 0),
            count("/v1/messages", 429, 0, 1),
        ];

        assert_eq!(
            failed_columns(&counts),
            FailedColumns {
                dates: vec!["2026-11-23".to_owned(); 2],
                categories: vec!["llm".to_owned(); 2],
                routes: vec!["/v1/messages".to_owned(); 2],
                status_codes: vec![429, 500],
                failed_requests: vec![1, 2],
            }
        );
    }
}
