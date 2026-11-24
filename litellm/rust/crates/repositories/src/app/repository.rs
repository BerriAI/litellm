//! The app repository: one `LiteLLM_AppTable` row, read when an app spend counter reseeds.

use litellm_db::Call;

use super::queries::{self, App};

use crate::{Database, Side};

const FIND_APP: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_AppTable"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_AppTable`.
pub struct AppRepository {
    database: Database,
}

impl AppRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The app with id `app_id`, or `None` when there is none.
    pub async fn find(&self, app_id: &str) -> Result<Option<App>, litellm_db::Error> {
        self.database
            .run(FIND_APP, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_app(session.executor(), app_id).await
            })
            .await
    }
}
