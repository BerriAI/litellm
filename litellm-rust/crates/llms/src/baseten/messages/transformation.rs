use crate::{
    Error,
    base_llm::messages::transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
    baseten::common_utils::{SECRET_NAMES, complete_url, validate_environment},
};

pub struct BasetenMessagesConfig;

pub const BASETEN_MESSAGES_CONFIG: BasetenMessagesConfig = BasetenMessagesConfig;

impl BaseMessagesConfig for BasetenMessagesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        validate_environment(headers, api_key, env_lookup)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_url(api_base, "messages", env_lookup))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }
}
