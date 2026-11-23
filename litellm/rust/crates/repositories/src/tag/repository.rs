//! The tag repository: which tags have a row, and those rows with their budgets.

use litellm_db::Call;

use super::queries::{self, Tag};

use crate::{Database, Side};

const TAG_NAMES: Call = Call {
    call_type: "find_many",
    table: Some("LiteLLM_TagTable"),
    operation: Some("find_many"),
};

const FIND_TAGS: Call = Call {
    call_type: "get_tag_objects_batch",
    table: Some("LiteLLM_TagTable"),
    operation: Some("find_many"),
};

/// Reads `LiteLLM_TagTable` joined to the budget each tag names.
pub struct TagRepository {
    database: Database,
}

impl TagRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Up to `limit` tag names.
    pub async fn names(&self, limit: i64) -> Result<Vec<String>, litellm_db::Error> {
        self.database
            .run(TAG_NAMES, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_tag_names(session.executor(), limit).await
            })
            .await
    }

    /// The rows of whichever of `names` exist, in no particular order.
    pub async fn find_many_with_budgets(
        &self,
        names: &[String],
    ) -> Result<Vec<Tag>, litellm_db::Error> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        self.database
            .run(FIND_TAGS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_tags(session.executor(), names).await
            })
            .await
    }
}
