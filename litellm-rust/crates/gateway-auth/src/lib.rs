mod authentication;
mod caller;
mod custom;
mod error;
pub mod extract;
mod http;
pub mod keys;
mod ui;

use sha2::{Digest, Sha256};

pub use authentication::Auth;
pub use caller::{AccessRequest, Caller, McpAction};
pub use custom::{CustomAuth, CustomOutcome};
pub use error::{Error, KeyError, UiAuthError};
pub use extract::Bearer;
pub use http::authenticate;
pub use ui::{UI_CSRF_KEY, UiAuthSession, UiBackend, UiCredentials, UiSession, UiUser};

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}
