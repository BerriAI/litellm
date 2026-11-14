mod auth;
mod config;
pub mod constants;
mod identity;
#[cfg(feature = "google-sdk")]
mod sdk;

pub use auth::{VertexAuth, VertexEnvironment};
pub use config::{
    CredentialSource, VertexConfig, get_vertex_ai_location, get_vertex_ai_project,
    get_vertex_ai_project_from_credentials,
};
pub use constants::secret_names;
pub use identity::{GcpAuth, GcpAuthFuture, GcpProviderLoader, GcpTokenSource};
#[cfg(feature = "google-sdk")]
pub use sdk::GoogleCredentials;
