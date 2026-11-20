//! The audit log repository: inserts the rows the management endpoints record for each change they make.

use jiff::Timestamp;
use litellm_db::Call;
use serde_json::value::RawValue;

use super::queries;
use crate::{Database, Side};

const CREATE_AUDIT_LOG: Call = Call {
    call_type: "create",
    table: Some("LiteLLM_AuditLog"),
    operation: Some("insert"),
};

const CREATE_MANY_AUDIT_LOGS: Call = Call {
    call_type: "create_many",
    table: Some("LiteLLM_AuditLog"),
    operation: Some("insert"),
};

/// One audit log row. `changed_by` and `changed_by_api_key` are stored as empty when `None`, the column
/// default. The two values are JSON documents, which may hold credentials, so the type has no `Debug`.
#[derive(Clone)]
pub struct NewAuditLog {
    pub id: String,
    pub updated_at: Timestamp,
    pub changed_by: Option<String>,
    pub changed_by_api_key: Option<String>,
    pub action: String,
    pub table_name: String,
    pub object_id: String,
    pub before_value: Option<Box<RawValue>>,
    pub updated_values: Option<Box<RawValue>>,
}

/// Writes `LiteLLM_AuditLog`.
pub struct AuditLogRepository {
    database: Database,
}

impl AuditLogRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    pub async fn insert(&self, row: &NewAuditLog) -> Result<(), litellm_db::Error> {
        self.insert_rows(CREATE_AUDIT_LOG, std::slice::from_ref(row))
            .await
    }

    /// Inserts every row in one statement, so either all of them are stored or none is.
    pub async fn insert_many(&self, rows: &[NewAuditLog]) -> Result<(), litellm_db::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        self.insert_rows(CREATE_MANY_AUDIT_LOGS, rows).await
    }

    async fn insert_rows(&self, call: Call, rows: &[NewAuditLog]) -> Result<(), litellm_db::Error> {
        self.database
            .run(call, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::insert_audit_logs(session.executor(), rows).await
            })
            .await
    }
}
