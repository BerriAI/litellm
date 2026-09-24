use litellm_http::request::{has_bearer_auth, has_header};
use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
};

use crate::{
    anthropic::experimental_pass_through::messages::thinking::ThinkingContext,
    base_llm::chat::transformation::Error,
};

pub type Headers = Vec<(String, String)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagesAuthStrategy {
    Bearer,
    Header(&'static str),
}

impl MessagesAuthStrategy {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(header_name) => header_name,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MessagesTransformContext {
    pub thinking: ThinkingContext,
    pub drop_params: bool,
}

pub trait BaseAnthropicMessagesConfig: Sync {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_anthropic_messages_request(
        &self,
        request: AnthropicMessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<AnthropicMessagesRequest, Error> {
        Ok(request)
    }

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: AnthropicMessagesResponse,
    ) -> Result<AnthropicMessagesResponse, Error> {
        Ok(response)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Header("x-api-key")
    }

    fn accepts_bearer_auth(&self) -> bool {
        false
    }

    fn authenticate(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, Error> {
        let strategy = self.auth_strategy();
        if has_header(&headers, strategy.header_name())
            || (self.accepts_bearer_auth() && has_bearer_auth(&headers))
        {
            return Ok(headers);
        }
        let api_key = self.resolve_api_key(api_key, env_lookup)?;
        let auth_header = match strategy {
            MessagesAuthStrategy::Bearer => {
                ("authorization".to_string(), format!("Bearer {api_key}"))
            }
            MessagesAuthStrategy::Header(name) => (name.to_string(), api_key),
        };
        Ok(headers.into_iter().chain([auth_header]).collect())
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }

    fn request_headers(&self, headers: Headers, _request: &AnthropicMessagesRequest) -> Headers {
        headers
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const X_API_KEY: MessagesAuthStrategy = MessagesAuthStrategy::Header("x-api-key");

    struct StubConfig {
        strategy: MessagesAuthStrategy,
        accepts_bearer: bool,
    }

    impl BaseAnthropicMessagesConfig for StubConfig {
        fn get_complete_url(
            &self,
            _api_base: Option<&str>,
            _model: &str,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<String, Error> {
            Ok(String::new())
        }

        fn resolve_api_key(
            &self,
            api_key: Option<&str>,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<String, Error> {
            api_key
                .map(str::to_string)
                .ok_or(Error::MissingField("api_key"))
        }

        fn auth_strategy(&self) -> MessagesAuthStrategy {
            self.strategy
        }

        fn accepts_bearer_auth(&self) -> bool {
            self.accepts_bearer
        }
    }

    struct DefaultsConfig;

    impl BaseAnthropicMessagesConfig for DefaultsConfig {
        fn get_complete_url(
            &self,
            _api_base: Option<&str>,
            _model: &str,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<String, Error> {
            Ok(String::new())
        }

        fn resolve_api_key(
            &self,
            api_key: Option<&str>,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<String, Error> {
            api_key
                .map(str::to_string)
                .ok_or(Error::MissingField("api_key"))
        }
    }

    #[test]
    fn default_config_adds_its_key_next_to_a_forwarded_bearer() {
        assert_eq!(
            DefaultsConfig.authenticate(
                headers(&[("authorization", "Bearer forwarded")]),
                Some("sk"),
                &|_| None
            ),
            Ok(headers(&[
                ("authorization", "Bearer forwarded"),
                ("x-api-key", "sk")
            ]))
        );
    }

    #[test]
    fn default_request_headers_are_the_given_headers() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude",
            "max_tokens": 16,
            "speed": "fast",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert_eq!(
            DefaultsConfig.request_headers(headers(&[("x-api-key", "sk")]), &request),
            headers(&[("x-api-key", "sk")])
        );
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[rstest]
    #[case::own_header_is_kept(
        X_API_KEY,
        false,
        headers(&[("x-api-key", "forwarded")]),
        None,
        Ok(headers(&[("x-api-key", "forwarded")]))
    )]
    #[case::own_header_in_any_casing_is_kept(
        X_API_KEY,
        false,
        headers(&[("X-Api-Key", "forwarded")]),
        None,
        Ok(headers(&[("X-Api-Key", "forwarded")]))
    )]
    #[case::accepted_bearer_is_kept(
        X_API_KEY,
        true,
        headers(&[("authorization", "Bearer forwarded")]),
        None,
        Ok(headers(&[("authorization", "Bearer forwarded")]))
    )]
    #[case::bearer_the_provider_does_not_accept_gets_the_key_too(
        X_API_KEY,
        false,
        headers(&[("authorization", "Bearer forwarded")]),
        Some("sk"),
        Ok(headers(&[("authorization", "Bearer forwarded"), ("x-api-key", "sk")]))
    )]
    #[case::blank_bearer_gets_the_key(
        X_API_KEY,
        true,
        headers(&[("authorization", "Bearer  ")]),
        Some("sk"),
        Ok(headers(&[("authorization", "Bearer  "), ("x-api-key", "sk")]))
    )]
    #[case::key_goes_in_the_provider_header(
        X_API_KEY,
        false,
        headers(&[("content-type", "application/json")]),
        Some("sk"),
        Ok(headers(&[("content-type", "application/json"), ("x-api-key", "sk")]))
    )]
    #[case::key_goes_in_a_bearer(
        MessagesAuthStrategy::Bearer,
        false,
        headers(&[]),
        Some("sk"),
        Ok(headers(&[("authorization", "Bearer sk")]))
    )]
    #[case::bearer_strategy_keeps_a_forwarded_authorization(
        MessagesAuthStrategy::Bearer,
        false,
        headers(&[("authorization", "Bearer forwarded")]),
        None,
        Ok(headers(&[("authorization", "Bearer forwarded")]))
    )]
    #[case::missing_key_is_an_error(
        X_API_KEY,
        false,
        headers(&[]),
        None,
        Err(Error::MissingField("api_key"))
    )]
    fn default_authenticate_applies_the_key_unless_a_credential_is_forwarded(
        #[case] strategy: MessagesAuthStrategy,
        #[case] accepts_bearer: bool,
        #[case] forwarded: Headers,
        #[case] api_key: Option<&str>,
        #[case] expected: Result<Headers, Error>,
    ) {
        let config = StubConfig {
            strategy,
            accepts_bearer,
        };
        assert_eq!(config.authenticate(forwarded, api_key, &|_| None), expected);
    }
}
