//! The proxy model repository: every model deployment stored in `LiteLLM_ProxyModelTable`.

use litellm_db::Call;

use super::queries::{self, ProxyModel};
use crate::Database;

const FIND_PROXY_MODELS: Call = Call {
    call_type: "find_many",
    table: Some("LiteLLM_ProxyModelTable"),
    operation: Some("find_many"),
};

/// Reads `LiteLLM_ProxyModelTable`.
pub struct ProxyModelRepository {
    database: Database,
}

impl ProxyModelRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Every stored deployment, in no particular order. Read from the writer, so a model another pod just
    /// added is never missing and one it just deleted never lingers, unless the writer is down, when the replica
    /// keeps a boot during a writer outage loading models, like Python's `get_models_from_db`.
    pub async fn all(&self) -> Result<Vec<ProxyModel>, litellm_db::Error> {
        self.database
            .run(FIND_PROXY_MODELS, || async {
                let mut session = self.database.writer_preferred_session().await?;
                queries::find_proxy_models(session.executor()).await
            })
            .await
    }
}
