//! The agent identity repository: records that a managed agent authenticated.

use litellm_db::Call;

use super::queries;
use crate::{Database, Side};

const UPDATE_MANY_AGENT_IDENTITIES: Call = Call {
    call_type: "update_many",
    table: Some("LiteLLM_AgentIdentity"),
    operation: Some("update"),
};

/// Writes `LiteLLM_AgentIdentity`.
pub struct AgentIdentityRepository {
    database: Database,
}

impl AgentIdentityRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Stamps the identity of `agent_id` as authenticated now. False when the identity is no longer the active
    /// one at `revision`, or its agent is disabled or no longer identity managed, so nothing was recorded.
    pub async fn record_authentication(
        &self,
        agent_id: &str,
        revision: &str,
    ) -> Result<bool, litellm_db::Error> {
        self.database
            .run(UPDATE_MANY_AGENT_IDENTITIES, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::record_authentication(session.executor(), agent_id, revision).await
            })
            .await
    }
}
