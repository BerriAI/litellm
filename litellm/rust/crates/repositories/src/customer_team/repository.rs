//! The customer team repository: one `LiteLLM_CustomerTeamTable` row with its budget and budget override, read when a
//! customer team spend counter reseeds.

use litellm_db::Call;

use super::queries::{self, CustomerTeam};

use crate::{Database, Side};

const FIND_CUSTOMER_TEAM: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_CustomerTeamTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_CustomerTeamTable` joined to the budget and budget override each team names.
pub struct CustomerTeamRepository {
    database: Database,
}

impl CustomerTeamRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The customer team with id `customer_team_id` and its budgets, or `None` when there is none.
    pub async fn find_with_budgets(
        &self,
        customer_team_id: &str,
    ) -> Result<Option<CustomerTeam>, litellm_db::Error> {
        self.database
            .run(FIND_CUSTOMER_TEAM, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_customer_team(session.executor(), customer_team_id).await
            })
            .await
    }
}
