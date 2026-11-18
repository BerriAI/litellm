//! The app customer member repository: one `LiteLLM_AppCustomerMemberTable` row, read when an app customer member
//! spend counter reseeds.

use litellm_db::Call;

use super::queries::{self, AppCustomerMember};

use crate::{Database, Side};

const FIND_APP_CUSTOMER_MEMBER: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_AppCustomerMemberTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_AppCustomerMemberTable`.
pub struct AppCustomerMemberRepository {
    database: Database,
}

impl AppCustomerMemberRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Customer member `customer_member_id`'s row within app `app_id`, or `None` when there is none.
    pub async fn find(
        &self,
        app_id: &str,
        customer_member_id: &str,
    ) -> Result<Option<AppCustomerMember>, litellm_db::Error> {
        self.database
            .run(FIND_APP_CUSTOMER_MEMBER, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_app_customer_member(session.executor(), app_id, customer_member_id)
                    .await
            })
            .await
    }
}
