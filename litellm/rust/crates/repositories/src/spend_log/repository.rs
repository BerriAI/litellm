//! The spend log repository: the spend a key, team or user logged since a window start, read when a budget window
//! counter is cold and the maintained `LiteLLM_BudgetWindowSpend` row cannot answer

use chrono::{DateTime, Utc};
use litellm_db::Call;

use super::queries;
use crate::{Database, Side};

const WINDOW_SPEND: Call = Call {
    call_type: "group_by",
    table: Some("LiteLLM_SpendLogs"),
    operation: Some("group_by"),
};

/// Whose spend a window sum adds up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpendLogOwner {
    /// `api_key`, the hashed virtual key
    Key,
    Team,
    User,
}

/// Reads `LiteLLM_SpendLogs`.
pub struct SpendLogRepository {
    database: Database,
}

impl SpendLogRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The total spend `owner_id` logged at or after `since`, zero when it logged none.
    pub async fn window_spend(
        &self,
        owner: SpendLogOwner,
        owner_id: &str,
        since: DateTime<Utc>,
    ) -> Result<f64, litellm_db::Error> {
        let since = since.naive_utc();
        self.database
            .run(WINDOW_SPEND, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::window_spend(session.executor(), owner, owner_id, since).await
            })
            .await
    }
}
