//! The budget window spend repository: the spend recorded against one budget window, read by its primary key for
//! the counter reseed and the shared model budget windows.

use litellm_db::Call;

use super::queries::{self, BudgetWindowSpend};
use crate::{Database, Side};

const FIND_WINDOW_SPEND: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_BudgetWindowSpend"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_BudgetWindowSpend`.
pub struct BudgetWindowSpendRepository {
    database: Database,
}

impl BudgetWindowSpendRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The window row at (`entity_type`, `entity_id`, `window_duration`) read from `side`, or `None` when no window
    /// has been recorded.
    pub async fn find(
        &self,
        entity_type: &str,
        entity_id: &str,
        window_duration: &str,
        side: Side,
    ) -> Result<Option<BudgetWindowSpend>, litellm_db::Error> {
        self.database
            .run(FIND_WINDOW_SPEND, || async {
                let mut session = self.database.session(side).await?;
                queries::find_window_spend(
                    session.executor(),
                    entity_type,
                    entity_id,
                    window_duration,
                )
                .await
            })
            .await
    }
}
