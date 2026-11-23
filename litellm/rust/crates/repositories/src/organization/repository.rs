//! The organization repository: one organization by id, with its budget when asked, or the organizations sharing an
//! alias, read when the auth cache misses.

use litellm_db::Call;

use super::queries::{self, Organization};
use crate::{Database, Side};

const FIND_ORGANIZATION: Call = Call {
    call_type: "get_org_object",
    table: None,
    operation: None,
};

const FIND_ORGANIZATIONS_BY_ALIAS: Call = Call {
    call_type: "get_org_object_by_alias",
    table: None,
    operation: None,
};

/// Reads `LiteLLM_OrganizationTable`, joined to its `LiteLLM_BudgetTable` row on request.
pub struct OrganizationRepository {
    database: Database,
}

impl OrganizationRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The organization `organization_id`, carrying its budget when `include_budget_table` is set, or `None` when no
    /// row has that id.
    pub async fn find_with_optional_budget(
        &self,
        organization_id: &str,
        include_budget_table: bool,
    ) -> Result<Option<Organization>, litellm_db::Error> {
        self.database
            .run(FIND_ORGANIZATION, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_organization(
                    session.executor(),
                    organization_id,
                    include_budget_table,
                )
                .await
            })
            .await
    }

    /// Up to two organizations named `alias`, without budgets, so the caller can refuse an alias
    /// two organizations share.
    pub async fn find_up_to_two_by_alias(
        &self,
        alias: &str,
    ) -> Result<Vec<Organization>, litellm_db::Error> {
        self.database
            .run(FIND_ORGANIZATIONS_BY_ALIAS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_organizations_by_alias(session.executor(), alias).await
            })
            .await
    }
}
