//! The budget repository: `LiteLLM_BudgetTable` rows read by id, for an end user's default budget and a team's
//! default per-member budget

use litellm_db::Call;

use super::queries::{self, Budget};
use crate::{Database, Side};

const GET_DEFAULT_END_USER_BUDGET: Call = Call {
    call_type: "get_default_end_user_budget",
    table: None,
    operation: None,
};

const FIND_TEAM_MEMBER_DEFAULT_BUDGET: Call = Call {
    call_type: "get_team_member_default_budget",
    table: None,
    operation: None,
};

/// Reads `LiteLLM_BudgetTable`.
pub struct BudgetRepository {
    database: Database,
}

impl BudgetRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The budget row `budget_id`, reported as the end user default budget lookup it serves.
    pub async fn find_end_user_default(
        &self,
        budget_id: &str,
    ) -> Result<Option<Budget>, litellm_db::Error> {
        self.find_as(GET_DEFAULT_END_USER_BUDGET, budget_id, Side::Read)
            .await
    }

    /// The budget row `budget_id`, reported as the team member default budget lookup it serves. A member add
    /// reads it from `Side::Writer`, since it links the budget it finds.
    pub async fn find_team_member_default(
        &self,
        budget_id: &str,
        side: Side,
    ) -> Result<Option<Budget>, litellm_db::Error> {
        self.find_as(FIND_TEAM_MEMBER_DEFAULT_BUDGET, budget_id, side)
            .await
    }

    async fn find_as(
        &self,
        call: Call,
        budget_id: &str,
        side: Side,
    ) -> Result<Option<Budget>, litellm_db::Error> {
        self.database
            .run(call, || async {
                let mut session = self.database.session(side).await?;
                queries::find_budget(session.executor(), budget_id).await
            })
            .await
    }
}
