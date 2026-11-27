//! The MCP server repository: what the proxy's MCP registry reads of the servers stored in its
//! database that changes what it logs of a request.

use litellm_db::Call;

use super::queries;
use crate::Database;

const ANY_SERVER_FORWARDS_HEADERS: Call = Call {
    call_type: "any_server_forwards_headers",
    table: Some("LiteLLM_MCPServerTable"),
    operation: None,
};

/// Reads the MCP servers stored in the proxy's database.
pub struct McpServerRepository {
    database: Database,
}

impl McpServerRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Whether a stored server forwards a caller's header upstream through `extra_headers`, other
    /// than `authorization`, which `upstream_credential_headers` then drops from every logged copy
    /// of a request's headers.
    pub async fn any_server_forwards_headers(&self) -> Result<bool, litellm_db::Error> {
        self.database
            .run(ANY_SERVER_FORWARDS_HEADERS, || async {
                let mut session = self.database.writer_preferred_session().await?;
                queries::any_server_forwards_headers(session.executor()).await
            })
            .await
    }
}
