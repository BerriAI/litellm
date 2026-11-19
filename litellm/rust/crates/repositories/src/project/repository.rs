//! The project repository: one `LiteLLM_ProjectTable` row with its budget, for project auth checks.

use litellm_db::Call;

use super::queries::{self, Project};

use crate::{Database, Side};

const FIND_PROJECT: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_ProjectTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_ProjectTable` joined to the budget each project names.
pub struct ProjectRepository {
    database: Database,
}

impl ProjectRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The project with id `project_id`, or `None` when there is none.
    pub async fn find_with_budget(
        &self,
        project_id: &str,
    ) -> Result<Option<Project>, litellm_db::Error> {
        self.database
            .run(FIND_PROJECT, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_project(session.executor(), project_id).await
            })
            .await
    }
}
