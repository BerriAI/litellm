//! The stored config repository: what the proxy loads from its database on top of its config file
//! when `store_model_in_db` is on, read in one pass so a caller can tell which requests it reaches.

use litellm_db::Call;

use super::queries;
use crate::{Database, Side};

/// The `LiteLLM_Config` rows the proxy layers on its config file's sections.
const SECTIONS: [&str; 4] = [
    "general_settings",
    "router_settings",
    "litellm_settings",
    "environment_variables",
];

const CONFIG_SECTIONS: Call = Call {
    call_type: "get_config_param",
    table: Some("LiteLLM_Config"),
    operation: Some("find_many"),
};

const MODEL_NAMES: Call = Call {
    call_type: "get_models_from_db",
    table: Some("LiteLLM_ProxyModelTable"),
    operation: Some("find_many"),
};

const GUARDRAILS: Call = Call {
    call_type: "get_all_guardrails_from_db",
    table: Some("LiteLLM_GuardrailsTable"),
    operation: Some("find_many"),
};

const POLICY_ATTACHMENTS: Call = Call {
    call_type: "sync_attachments_from_db",
    table: Some("LiteLLM_PolicyAttachmentTable"),
    operation: Some("find_many"),
};

/// A settings section the database holds: its name, the keys it sets, and whether it turns
/// `store_model_in_db` on, which only `general_settings` can.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredSection {
    pub section: String,
    pub keys: Vec<String>,
    pub store_model_in_db: bool,
}

/// What the database holds that the proxy serves from when `store_model_in_db` is on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoredConfig {
    pub sections: Vec<StoredSection>,
    /// The model group of each database model, each once.
    pub models: Vec<String>,
    /// The active guardrails that run on every request.
    pub default_on_guardrails: Vec<String>,
    /// The policies an attachment attaches.
    pub attached_policies: Vec<String>,
}

/// Reads `LiteLLM_Config`, `LiteLLM_ProxyModelTable`, `LiteLLM_GuardrailsTable` and
/// `LiteLLM_PolicyAttachmentTable`.
pub struct StoredConfigRepository {
    database: Database,
}

impl StoredConfigRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Everything the proxy loads from the database on top of its config file. Each read sees the
    /// latest commit, as the proxy's reconcile reads models through the writer.
    pub async fn read(&self) -> Result<StoredConfig, litellm_db::Error> {
        let sections = self.sections().await?;
        let models = self
            .database
            .run(MODEL_NAMES, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::model_names(session.executor()).await
            })
            .await?;
        let default_on_guardrails = self
            .database
            .run(GUARDRAILS, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::default_on_guardrails(session.executor()).await
            })
            .await?;
        let attached_policies = self
            .database
            .run(POLICY_ATTACHMENTS, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::attached_policies(session.executor()).await
            })
            .await?;
        Ok(StoredConfig {
            sections,
            models,
            default_on_guardrails,
            attached_policies,
        })
    }

    /// The settings sections alone, which say whether the proxy loads the rest.
    pub async fn sections(&self) -> Result<Vec<StoredSection>, litellm_db::Error> {
        let names = SECTIONS.map(str::to_owned);
        let rows = self
            .database
            .run(CONFIG_SECTIONS, || async {
                let mut session = self.database.session(Side::Writer).await?;
                queries::config_sections(session.executor(), &names).await
            })
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| StoredSection {
                section: row.param_name,
                keys: row.keys,
                store_model_in_db: row.store_model_in_db,
            })
            .collect())
    }
}
