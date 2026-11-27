//! The UI settings repository: the one `LiteLLM_UISettings` row the admin UI saves its settings to.

use litellm_db::Call;

use super::queries;
use crate::{Database, RawJson, Side};

const FIND_UI_SETTINGS: Call = Call {
    call_type: "find_unique",
    table: Some("LiteLLM_UISettings"),
    operation: Some("find_unique"),
};

/// Reads `LiteLLM_UISettings`.
pub struct UiSettingsRepository {
    database: Database,
}

impl UiSettingsRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The `ui_settings` column of the `ui_settings` row exactly as stored, or `None` when the UI never saved one.
    /// Read from the replica when one is configured, as Python's `sync_ui_settings_to_general_settings` does at
    /// startup and on each config reload.
    pub async fn find(&self) -> Result<Option<RawJson>, litellm_db::Error> {
        self.database
            .run(FIND_UI_SETTINGS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_ui_settings(session.executor()).await
            })
            .await
    }
}
