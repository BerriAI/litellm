//! The config repository: the `LiteLLM_Config` rows for the settings sections a caller asks for.

use litellm_db::Call;

use super::queries::{self, ConfigParam};
use crate::{Database, Side};

const FIND_CONFIG_PARAMS: Call = Call {
    call_type: "find_many",
    table: Some("LiteLLM_Config"),
    operation: Some("find_many"),
};

/// Reads `LiteLLM_Config`.
pub struct ConfigRepository {
    database: Database,
}

impl ConfigRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The rows for whichever of `param_names` are stored, in no particular order, read from the replica when one
    /// is configured as Python's `prefetch_config_params` does.
    pub async fn find_many(
        &self,
        param_names: &[String],
    ) -> Result<Vec<ConfigParam>, litellm_db::Error> {
        if param_names.is_empty() {
            return Ok(Vec::new());
        }
        self.database
            .run(FIND_CONFIG_PARAMS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_config_params(session.executor(), param_names).await
            })
            .await
    }
}
