//! The object permission repository: one `LiteLLM_ObjectPermissionTable` row, the MCP, agent, vector store,
//! model, tool and skill grants a key, team, user, project or end user points at.

use litellm_db::Call;

use super::queries::{self, ObjectPermission};

use crate::{Database, Side};

const FIND_OBJECT_PERMISSION: Call = Call {
    call_type: "get_object_permission",
    table: Some("LiteLLM_ObjectPermissionTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_ObjectPermissionTable`.
pub struct ObjectPermissionRepository {
    database: Database,
}

impl ObjectPermissionRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The object permission with id `object_permission_id` read from `side`, or `None` when there is none.
    pub async fn find_by_object_permission_id(
        &self,
        object_permission_id: &str,
        side: Side,
    ) -> Result<Option<ObjectPermission>, litellm_db::Error> {
        self.database
            .run(FIND_OBJECT_PERMISSION, || async {
                let mut session = self.database.session(side).await?;
                queries::find_object_permission(session.executor(), object_permission_id).await
            })
            .await
    }
}
