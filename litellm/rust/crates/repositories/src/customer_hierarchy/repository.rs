use litellm_db::{Call, Error};

use super::queries::{self, CustomerMemberRow, CustomerOrgRow};
use crate::{Database, Side};

const MEMBER: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_CustomerMemberTable"),
    operation: Some("find_unique"),
};
const ORGANIZATIONS: Call = Call {
    call_type: "find_many",
    table: Some("LiteLLM_CustomerOrganizationTable"),
    operation: Some("find_many"),
};

/// Reads customer hierarchy rows and their budget relations.
pub struct CustomerHierarchyRepository {
    database: Database,
}

impl CustomerHierarchyRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    pub async fn member(&self, id: &str) -> Result<Option<CustomerMemberRow>, Error> {
        self.database
            .run(MEMBER, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::member(session.executor(), id).await
            })
            .await
    }

    pub async fn organizations(&self, alias: &str) -> Result<Vec<CustomerOrgRow>, Error> {
        self.database
            .run(ORGANIZATIONS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::organizations(session.executor(), alias).await
            })
            .await
    }
}
