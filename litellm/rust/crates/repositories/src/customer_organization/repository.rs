//! The customer organization repository: one `LiteLLM_CustomerOrganizationTable` row with its budget, read when a
//! customer organization spend counter reseeds.

use litellm_db::Call;

use super::queries::{self, CustomerOrganization};

use crate::{Database, Side};

const FIND_CUSTOMER_ORGANIZATION: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_CustomerOrganizationTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_CustomerOrganizationTable` joined to the budget each organization names.
pub struct CustomerOrganizationRepository {
    database: Database,
}

impl CustomerOrganizationRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The customer organization with id `customer_org_id` and its budget, or `None` when there is none.
    pub async fn find_with_budget(
        &self,
        customer_org_id: &str,
    ) -> Result<Option<CustomerOrganization>, litellm_db::Error> {
        self.database
            .run(FIND_CUSTOMER_ORGANIZATION, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_customer_organization(session.executor(), customer_org_id).await
            })
            .await
    }
}
