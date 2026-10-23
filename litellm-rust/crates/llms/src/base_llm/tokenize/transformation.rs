use litellm_llms_types::formats::tokenize::TokenizeResponse;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{Error, ErrorDetail};

use crate::base_llm::auth::{Headers, ValidatedEnvironment};

pub trait BaseTokenizeConfig: Sync {
    fn get_supported_tokenize_params(&self) -> &'static [&'static str];

    fn map_tokenize_params(
        &self,
        _model: &str,
        params: &Map<String, Value>,
    ) -> Result<Map<String, Value>, Error> {
        if let Some(parameter) = params.keys().find(|name| {
            !self
                .get_supported_tokenize_params()
                .contains(&name.as_str())
        }) {
            return Err(Error::InvalidRequest(ErrorDetail::InvalidChoice {
                field: "parameter",
                actual: parameter.clone(),
                choices: self.get_supported_tokenize_params().to_vec(),
            }));
        }
        Ok(params.clone())
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_tokenize_request(
        &self,
        model: &str,
        prompt: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Value, Error>;

    fn transform_tokenize_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<TokenizeResponse, Error>;

    fn secret_names(&self) -> Vec<&'static str>;

    fn secret_param_names(&self) -> &'static [&'static str] {
        &[]
    }
}

pub fn decode_params<T: DeserializeOwned>(params: &Map<String, Value>) -> Result<T, Error> {
    serde_json::from_value(Value::Object(params.clone())).map_err(|source| {
        Error::InvalidRequest(ErrorDetail::invalid("tokenization parameters", source))
    })
}

pub fn validate_token_count(count: u64, length: usize) -> Result<(), Error> {
    if count != length as u64 {
        return Err(Error::InvalidResponse(ErrorDetail::InvalidValue {
            field: "count",
            expected: "the number of returned tokens",
            actual: Value::from(count),
        }));
    }
    Ok(())
}
