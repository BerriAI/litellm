//! The credential repository: every named credential stored in `LiteLLM_CredentialsTable`.

use litellm_db::Call;

use super::queries::{self, Credential};
use crate::Database;

const FIND_CREDENTIALS: Call = Call {
    call_type: "find_many",
    table: Some("LiteLLM_CredentialsTable"),
    operation: Some("find_many"),
};

/// Reads `LiteLLM_CredentialsTable`.
pub struct CredentialRepository {
    database: Database,
}

impl CredentialRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Every stored credential, in no particular order, read from the writer unless it is down, like Python's
    /// `get_credentials`.
    pub async fn all(&self) -> Result<Vec<Credential>, litellm_db::Error> {
        self.database
            .run(FIND_CREDENTIALS, || async {
                let mut session = self.database.writer_preferred_session().await?;
                queries::find_credentials(session.executor()).await
            })
            .await
    }
}
