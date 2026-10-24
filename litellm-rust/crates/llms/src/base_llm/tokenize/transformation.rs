use std::collections::BTreeMap;

use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_llms_types::formats::tokenize::{TokenizeOutput, TokenizeResponse};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{Error, ErrorDetail};

use crate::base_llm::auth::{Headers, ValidatedEnvironment};

pub type TokenizeMap = BTreeMap<String, Sourced<Value>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenizeCapabilities {
    pub token_ids: bool,
    pub tokens: bool,
}

impl TokenizeCapabilities {
    pub fn supports(self, output: TokenizeOutput) -> bool {
        match output {
            TokenizeOutput::Count | TokenizeOutput::Available => true,
            TokenizeOutput::Ids => self.token_ids,
            TokenizeOutput::Tokens => self.tokens,
            TokenizeOutput::Both => self.token_ids && self.tokens,
        }
    }
}

pub struct TokenizeContext<'a> {
    pub api_key: Option<&'a Sourced<SecretValue>>,
    pub api_base: Option<&'a Sourced<String>>,
    pub extra_headers: &'a TokenizeMap,
    pub deployment_params: &'a TokenizeMap,
    pub optional_params: &'a TokenizeMap,
}

impl<'a> TokenizeContext<'a> {
    pub fn new(
        api_key: Option<&'a Sourced<SecretValue>>,
        api_base: Option<&'a Sourced<String>>,
        extra_headers: &'a TokenizeMap,
        deployment_params: &'a TokenizeMap,
        optional_params: &'a TokenizeMap,
    ) -> Self {
        Self {
            api_key,
            api_base,
            extra_headers,
            deployment_params,
            optional_params,
        }
    }

    pub fn string_headers(&self) -> Result<Headers, Error> {
        self.extra_headers
            .iter()
            .map(|(name, value)| {
                let value = value.value().as_str().ok_or_else(|| {
                    Error::InvalidRequest(ErrorDetail::InvalidValue {
                        field: "extra_headers",
                        expected: "string header values",
                        actual: value.value().clone(),
                    })
                })?;
                Ok((name.clone(), value.to_owned()))
            })
            .collect()
    }
}

pub trait BaseTokenizeConfig: Sync {
    fn get_supported_tokenize_params(&self) -> &'static [&'static str];

    fn tokenize_capabilities(&self) -> TokenizeCapabilities {
        TokenizeCapabilities::default()
    }

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

    fn validate_tokenize_input(
        &self,
        input: &TokenizeContext<'_>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Error> {
        validate_common_tokenize_input(input, self.get_supported_tokenize_params())
    }

    fn validate_environment(
        &self,
        input: &TokenizeContext<'_>,
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

pub fn validate_common_tokenize_input(
    input: &TokenizeContext<'_>,
    supported_params: &[&'static str],
) -> Result<(), Error> {
    if input
        .deployment_params
        .values()
        .any(|value| value.source() == InputSource::Request)
    {
        return Err(Error::InvalidRequest(
            "request parameters cannot be deployment parameters".into(),
        ));
    }
    if let Some(parameter) = input
        .optional_params
        .keys()
        .find(|name| !supported_params.contains(&name.as_str()))
    {
        return Err(Error::InvalidRequest(ErrorDetail::InvalidChoice {
            field: "parameter",
            actual: parameter.clone(),
            choices: supported_params.to_vec(),
        }));
    }
    Ok(())
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
