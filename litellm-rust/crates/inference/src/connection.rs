use std::time::Duration;

use litellm_auth::{SecretValue, Sourced};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default)]
pub struct Connection {
    pub api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
    pub extra_headers: Option<Sourced<Map<String, Value>>>,
    pub timeout: Option<Duration>,
}

impl Connection {
    pub fn exposed_api_key(&self) -> Option<&str> {
        self.api_key.as_ref().map(|key| key.value().expose())
    }

    pub fn api_base_value(&self) -> Option<&str> {
        self.api_base.as_ref().map(|base| base.value().as_str())
    }

    pub fn extra_headers_value(&self) -> Option<Map<String, Value>> {
        self.extra_headers
            .as_ref()
            .map(|headers| headers.value().clone())
    }
}

#[cfg(test)]
mod tests {
    use litellm_auth::InputSource;
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn debug_hides_the_api_key() {
        let connection = Connection {
            api_key: Some(Sourced::new(
                SecretValue::new("caller-api-key"),
                InputSource::Request,
            )),
            ..Connection::default()
        };

        assert!(!format!("{connection:?}").contains("caller-api-key"));
    }
}
