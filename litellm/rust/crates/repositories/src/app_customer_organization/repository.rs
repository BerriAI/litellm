//! The app customer organization repository: one `LiteLLM_AppCustomerOrgTable` row, read when an app customer
//! organization spend counter reseeds.

use litellm_db::Call;

use super::queries::{self, AppCustomerOrganization};

use crate::{Database, Side};

const FIND_APP_CUSTOMER_ORGANIZATION: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_AppCustomerOrgTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_AppCustomerOrgTable`.
pub struct AppCustomerOrganizationRepository {
    database: Database,
}

impl AppCustomerOrganizationRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Customer organization `customer_org_id`'s row within app `app_id`, or `None` when there is none.
    pub async fn find(
        &self,
        app_id: &str,
        customer_org_id: &str,
    ) -> Result<Option<AppCustomerOrganization>, litellm_db::Error> {
        self.database
            .run(FIND_APP_CUSTOMER_ORGANIZATION, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_app_customer_organization(session.executor(), app_id, customer_org_id)
                    .await
            })
            .await
    }
}
