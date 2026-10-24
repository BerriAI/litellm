use serde::Deserialize;

use crate::SecretValue;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct UiCredentials {
    pub username: String,
    pub password: SecretValue,
}
