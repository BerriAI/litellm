use std::time::SystemTime;

/// Where the installed catalog came from and when, written once per install and never edited
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub origin: Origin,
    /// `LITELLM_LOCAL_MODEL_COST_MAP`, offline mode or a CLI process kept the bundled map
    pub is_env_forced: bool,
    /// Why the remote map was not installed, for a bundled map that stands in for it
    pub fallback_reason: Option<String>,
    pub loaded_at: SystemTime,
    /// `git_blob_id` of the installed bytes
    pub revision: Option<String>,
    pub etag: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The map shipped with the package, standing in for `attempted` when that is set
    Bundled {
        attempted: Option<String>,
    },
    Remote {
        url: String,
    },
}
