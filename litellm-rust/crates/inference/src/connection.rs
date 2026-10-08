use std::time::Duration;

use litellm_auth::SecretValue;
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default)]
pub struct Connection {
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

impl Connection {
    pub fn exposed_api_key(&self) -> Option<&str> {
        self.api_key.as_ref().map(SecretValue::expose)
    }

    pub fn api_base_value(&self) -> Option<&str> {
        self.api_base.as_deref()
    }

    pub fn extra_headers_value(&self) -> Option<Map<String, Value>> {
        self.extra_headers.clone()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn debug_hides_the_api_key() {
        let connection = Connection {
            api_key: Some(SecretValue::new("caller-api-key")),
            ..Connection::default()
        };

        assert!(!format!("{connection:?}").contains("caller-api-key"));
    }
}
