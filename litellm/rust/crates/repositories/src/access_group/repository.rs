//! The access group repository: one `LiteLLM_AccessGroupTable` row, the access object that grants models, MCP
//! servers, agents, skills and vector stores to the teams and keys it names.

use litellm_db::Call;

use super::queries::{self, AccessGroup};

use crate::{Database, Side};

const FIND_ACCESS_GROUP: Call = Call {
    call_type: "get_access_object",
    table: Some("LiteLLM_AccessGroupTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_AccessGroupTable`.
pub struct AccessGroupRepository {
    database: Database,
}

impl AccessGroupRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The access group with id `access_group_id` read from `side`, or `None` when there is none.
    pub async fn find_by_access_group_id(
        &self,
        access_group_id: &str,
        side: Side,
    ) -> Result<Option<AccessGroup>, litellm_db::Error> {
        self.database
            .run(FIND_ACCESS_GROUP, || async {
                let mut session = self.database.session(side).await?;
                queries::find_access_group(session.executor(), access_group_id).await
            })
            .await
    }
}
