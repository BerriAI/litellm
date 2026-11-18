//! The customer member repository: one `LiteLLM_CustomerMemberTable` row with its budget and budget override, read
//! when a customer member spend counter reseeds.

use litellm_db::Call;

use super::queries::{self, CustomerMember};

use crate::{Database, Side};

const FIND_CUSTOMER_MEMBER: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_CustomerMemberTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_CustomerMemberTable` joined to the budget and budget override each member names.
pub struct CustomerMemberRepository {
    database: Database,
}

impl CustomerMemberRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The customer member with id `customer_member_id` and its budgets, or `None` when there is none.
    pub async fn find_with_budgets(
        &self,
        customer_member_id: &str,
    ) -> Result<Option<CustomerMember>, litellm_db::Error> {
        self.database
            .run(FIND_CUSTOMER_MEMBER, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_customer_member(session.executor(), customer_member_id).await
            })
            .await
    }
}
