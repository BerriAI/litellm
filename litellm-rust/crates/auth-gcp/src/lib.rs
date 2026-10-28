mod auth;
mod config;
pub mod constants;
#[cfg(feature = "google-sdk")]
mod sdk;

pub use auth::{
    VertexAuth, VertexAuthFuture, VertexEnvironment, VertexProviderLoader, VertexTokenSource,
};
pub use config::{
    CredentialSource, VertexConfig, get_vertex_ai_location, get_vertex_ai_project,
    get_vertex_ai_project_from_credentials,
};
pub use constants::secret_names;
#[cfg(feature = "google-sdk")]
pub use sdk::GoogleCredentials;
