use crate::anthropic::messages::transformation::retry_anthropic_messages_after_http_error;
use litellm_auth::{CredentialPlacement, InputSource, Sourced};
use litellm_auth_azure::{AzureAuthInputs, ConfigValue, DEFAULT_AZURE_SCOPE};
use litellm_host::interceptors::RetriedRequest;
use litellm_http::request::{has_bearer_auth, has_header};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest},
    providers::anthropic::{BetaProvider, DEFAULT_HEADERS, MESSAGES_PATH},
};
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;
use serde_json::Value;

use crate::{
    Error,
    anthropic::{
        common_utils::{API_KEY_PLACEMENT, strip_trailing_slashes},
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{transform_messages_request, update_headers_with_safeguards_beta},
        },
    },
    azure_ai::common_utils::{
        AZURE_AD_TOKEN_ENV, AZURE_API_BASE_ENV, AZURE_API_KEY_ENV, AZURE_CLIENT_ID_ENV,
        AZURE_CLIENT_SECRET_ENV, AZURE_OPENAI_API_KEY_ENV, AZURE_PASSWORD_ENV, AZURE_SCOPE_ENV,
        AZURE_TENANT_ID_ENV, AZURE_USERNAME_ENV, CALLABLE_TOKEN_PROVIDER,
        CALLER_TOKEN_PROVIDER_PARAM, TOKEN_REFRESH_PARAM, resolve_azure_api_base,
        resolve_azure_api_key,
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::{MessagesShapeOptions, MessagesTransformContext},
            normalization::{normalize_system_role_messages, strip_cache_control_scope},
            transformation::BaseMessagesConfig,
        },
    },
};

const ANTHROPIC_PATH_SEGMENT: &str = "/anthropic";
const SCHEME_SEPARATOR: &str = "://";
const AZURE_API_KEY_HEADER: &str = "api-key";
const OIDC_PREFIX: &str = "oidc/";
const AUTHORIZATION_HEADER: &str = "authorization";
const BEARER_SCHEME: &str = "bearer";
const ENVIRONMENT_REFERENCE_PREFIX: &str = "os.environ/";

pub struct AzureAnthropicMessagesConfig;

pub const AZURE_ANTHROPIC_MESSAGES_CONFIG: AzureAnthropicMessagesConfig =
    AzureAnthropicMessagesConfig;

impl BaseMessagesConfig for AzureAnthropicMessagesConfig {
    fn tool_custom_field(&self, model: &str) -> crate::base_llm::messages::tools::ToolCustomField {
        crate::base_llm::messages::tools::ToolCustomField::Rewrite {
            hoist_defer_loading: crate::anthropic::messages::transformation::is_claude_model(model),
        }
    }

    fn retry_after_http_error(
        &self,
        status: u16,
        error_text: &str,
        request_body: &Value,
        context: &MessagesTransformContext,
    ) -> Option<RetriedRequest> {
        retry_anthropic_messages_after_http_error(status, error_text, request_body, context)
    }

    fn shape_request(
        &self,
        request: MessagesRequest,
        options: MessagesShapeOptions,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, options)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _litellm_params: &LitellmParams,
        _stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        complete_azure_anthropic_url(api_base, env_lookup)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request(
            strip_cache_control_scope(normalize_system_role_messages(
                request,
                context
                    .thinking
                    .capabilities
                    .supports_mid_conversation_system,
            )),
            &context.keeping_display_updates(),
        )
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[AZURE_OPENAI_API_KEY_ENV, AZURE_API_KEY_ENV]
    }

    /// `AZURE_AD_TOKEN`, which Python reads through `get_secret_str` only when there is no key,
    /// no caller credential and no `azure_ad_token` param, and `AZURE_API_BASE`, which it reads
    /// only when the call's base is empty (`api_base or get_secret_str("AZURE_API_BASE")`).
    fn deferred_secret_names(
        &self,
        headers: &Headers,
        api_key: Option<&str>,
        api_base: Option<&str>,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> &'static [&'static str] {
        let token_param = litellm_params
            .extra
            .get("azure_ad_token")
            .and_then(Value::as_str)
            .is_some_and(|token| !token.is_empty());
        let reads_token = !(has_caller_credential(headers)
            || resolve_azure_api_key(api_key, env_lookup).is_some()
            || token_param);
        let reads_base = api_base.is_none_or(str::is_empty);
        match (reads_token, reads_base) {
            (false, false) => &[],
            (true, false) => &[AZURE_AD_TOKEN_ENV],
            (false, true) => &[AZURE_API_BASE_ENV],
            (true, true) => &[AZURE_AD_TOKEN_ENV, AZURE_API_BASE_ENV],
        }
    }

    /// A forwarded `x-api-key`, `api-key` (sent as `x-api-key`) or `Authorization` other than a
    /// blank bearer is the only credential sent: no deployment key and no Entra ID token. Otherwise the Azure key goes in `x-api-key`, and
    /// without one an Entra ID token is minted as Python's `get_azure_ad_token` does.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_header(&headers, API_KEY_PLACEMENT.header_name())
            || has_caller_authorization(&headers)
        {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        if has_header(&headers, AZURE_API_KEY_HEADER) {
            return Ok(ValidatedEnvironment {
                headers: with_azure_key_as_x_api_key(headers),
                auth: AuthScheme::Forwarded,
            });
        }
        let auth = match resolve_azure_api_key(api_key, env_lookup) {
            Some(key) => AuthScheme::Credential {
                placement: API_KEY_PLACEMENT,
                secret: SecretValue::new(key),
            },
            None => entra_id_auth(litellm_params, env_lookup, &|name| std::env::var(name).ok()),
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_safeguards_beta(headers, request, BetaProvider::AzureAi)
    }
}

fn has_caller_credential(headers: &Headers) -> bool {
    has_header(headers, API_KEY_PLACEMENT.header_name())
        || has_header(headers, AZURE_API_KEY_HEADER)
        || has_caller_authorization(headers)
}

/// Any `Authorization` the caller sends is their credential, except a bearer with no token.
fn has_caller_authorization(headers: &Headers) -> bool {
    has_bearer_auth(headers)
        || headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case(AUTHORIZATION_HEADER)
                && !value.trim().is_empty()
                && !value
                    .trim()
                    .get(..BEARER_SCHEME.len())
                    .is_some_and(|scheme| scheme.eq_ignore_ascii_case(BEARER_SCHEME))
        })
}

fn with_azure_key_as_x_api_key(headers: Headers) -> Headers {
    headers
        .into_iter()
        .map(
            |(name, value)| match name.eq_ignore_ascii_case(AZURE_API_KEY_HEADER) {
                true => (API_KEY_PLACEMENT.header_name().to_string(), value),
                false => (name, value),
            },
        )
        .collect()
}

/// Python's `get_azure_ad_token` order, with each setting read from the call and then where
/// Python reads it: `AZURE_AD_TOKEN` through the secret lookup (`get_secret_str`), the rest from
/// the process environment (`os.getenv`). Flows only Python implements (username and password, OIDC references,
/// `os.environ/` references, token refresh) are left to the host's `get_azure_ad_token`, so
/// Python both orders and mints them.
fn entra_id_auth(
    litellm_params: &LitellmParams,
    secret_lookup: &dyn Fn(&str) -> Option<String>,
    process_env: &dyn Fn(&str) -> Option<String>,
) -> AuthScheme {
    let param = |name: &str| {
        litellm_params
            .extra
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let from = |lookup: &dyn Fn(&str) -> Option<String>, name: &str, environment: &str| {
        param(name).or_else(|| lookup(environment).filter(|value| !value.is_empty()))
    };
    let setting = |name: &str, environment: &str| from(process_env, name, environment);
    let flag = |name: &str| litellm_params.extra.get(name) == Some(&Value::Bool(true));
    let token = from(secret_lookup, "azure_ad_token", AZURE_AD_TOKEN_ENV);
    let tenant_id = setting("tenant_id", AZURE_TENANT_ID_ENV);
    let client_id = setting("client_id", AZURE_CLIENT_ID_ENV);
    let client_secret = setting("client_secret", AZURE_CLIENT_SECRET_ENV);
    let username = setting("azure_username", AZURE_USERNAME_ENV);
    let password = setting("azure_password", AZURE_PASSWORD_ENV);
    let named_provider = litellm_params.extra.get(CALLER_TOKEN_PROVIDER_PARAM);
    let caller_provider = named_provider.is_some();

    let client_secret_flow = match (&tenant_id, &client_id, &client_secret) {
        (Some(tenant_id), Some(client_id), Some(client_secret)) if !caller_provider => {
            Some((tenant_id, client_id, client_secret))
        }
        _ => None,
    };
    let username_flow = !caller_provider
        && client_secret_flow.is_none()
        && username.is_some()
        && password.is_some()
        && client_id.is_some();
    let oidc = client_id.is_some()
        && tenant_id.is_some()
        && token
            .as_deref()
            .is_some_and(|token| token.starts_with(OIDC_PREFIX));
    let refresh = !oidc
        && !caller_provider
        && client_secret_flow.is_none()
        && !username_flow
        && flag(TOKEN_REFRESH_PARAM);
    let referenced = client_secret_flow.is_some_and(|(tenant_id, client_id, client_secret)| {
        [tenant_id, client_id, client_secret]
            .iter()
            .any(|value| value.starts_with(ENVIRONMENT_REFERENCE_PREFIX))
    });
    let uncallable = named_provider.is_some_and(|provider| provider != CALLABLE_TOKEN_PROVIDER);
    if username_flow || oidc || refresh || referenced || uncallable {
        return AuthScheme::HostAzureAdToken;
    }
    if caller_provider {
        return AuthScheme::AzureAdToken {
            inputs: Box::default(),
        };
    }
    if let Some((tenant_id, client_id, client_secret)) = client_secret_flow {
        let configured = |value: &str| {
            ConfigValue::Value(Sourced::new(value.to_string(), InputSource::Deployment))
        };
        let scope = param("azure_scope")
            .or_else(|| process_env(AZURE_SCOPE_ENV))
            .unwrap_or_else(|| DEFAULT_AZURE_SCOPE.to_string());
        return AuthScheme::AzureAdToken {
            inputs: Box::new(AzureAuthInputs {
                tenant_id: configured(tenant_id),
                client_id: configured(client_id),
                client_secret: ConfigValue::Value(Sourced::new(
                    SecretValue::new(client_secret.as_str()),
                    InputSource::Deployment,
                )),
                azure_scope: configured(&scope),
                ..AzureAuthInputs::default()
            }),
        };
    }
    match token {
        Some(token) => AuthScheme::Credential {
            placement: CredentialPlacement::Bearer,
            secret: SecretValue::new(token),
        },
        None => AuthScheme::Forwarded,
    }
}

/// Python's `azure_anthropic_messages_url`: the base is cut at the first `/anthropic` of its path,
/// never of its host, and kept untrimmed.
pub fn complete_azure_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let api_base = resolve_azure_api_base(api_base, env_lookup)?;
    let base = api_base.trim_end_matches('/');
    if base.ends_with(MESSAGES_PATH) {
        return Ok(base.to_string());
    }
    let path_start = match base.find(SCHEME_SEPARATOR) {
        Some(scheme) => {
            let authority = scheme + SCHEME_SEPARATOR.len();
            base[authority..].find('/').map(|slash| authority + slash)
        }
        None => Some(0),
    };
    let root = path_start
        .and_then(|start| {
            base[start..]
                .find(ANTHROPIC_PATH_SEGMENT)
                .map(|anthropic| &base[..start + anthropic])
        })
        .unwrap_or(base);
    Ok(format!(
        "{}{ANTHROPIC_PATH_SEGMENT}{MESSAGES_PATH}",
        strip_trailing_slashes(root)
    ))
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::json;

    use litellm_auth::CredentialPlacement;

    use super::*;
    use crate::ErrorDetail;
    use crate::base_llm::messages::{
        context::MessagesModelCapabilities, mid_conversation_system::CONVERTED_SYSTEM_NOTE,
    };

    fn request_from(value: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn to_value(request: MessagesRequest) -> serde_json::Value {
        serde_json::to_value(request).expect("serializable request")
    }

    #[rstest]
    #[case::bare_host(
        "https://resource.services.ai.azure.com",
        "https://resource.services.ai.azure.com/anthropic/v1/messages"
    )]
    #[case::anthropic_segment(
        "https://resource.services.ai.azure.com/anthropic",
        "https://resource.services.ai.azure.com/anthropic/v1/messages"
    )]
    #[case::complete_endpoint(
        "https://resource.services.ai.azure.com/anthropic/v1/messages",
        "https://resource.services.ai.azure.com/anthropic/v1/messages"
    )]
    #[case::messages_path_without_anthropic(
        "https://resource.services.ai.azure.com/v1/messages",
        "https://resource.services.ai.azure.com/v1/messages"
    )]
    #[case::path_after_anthropic_is_dropped(
        "https://resource.services.ai.azure.com/anthropic/extra/",
        "https://resource.services.ai.azure.com/anthropic/v1/messages"
    )]
    #[case::host_starting_with_anthropic(
        "http://anthropic-x.services.ai.azure.com/d/dep/anthropic",
        "http://anthropic-x.services.ai.azure.com/d/dep/anthropic/v1/messages"
    )]
    #[case::bare_host_starting_with_anthropic(
        "http://anthropic-x.services.ai.azure.com",
        "http://anthropic-x.services.ai.azure.com/anthropic/v1/messages"
    )]
    #[case::userinfo_starting_with_anthropic(
        "http://anthropic@resource.services.ai.azure.com/d/dep",
        "http://anthropic@resource.services.ai.azure.com/d/dep/anthropic/v1/messages"
    )]
    #[case::empty_host_is_kept_for_the_client_to_refuse(
        "http:///d/dep/anthropic",
        "http:///d/dep/anthropic/v1/messages"
    )]
    #[case::without_scheme_cuts_anywhere(
        "resource.services.ai.azure.com/d/dep/anthropic",
        "resource.services.ai.azure.com/d/dep/anthropic/v1/messages"
    )]
    #[case::slash_before_query_moves_behind_the_path(
        "https://h/d/dep/?q=1",
        "https://h/d/dep?q=1/anthropic/v1/messages"
    )]
    #[case::trailing_spaces_without_anthropic_are_kept(
        "https://h/d/dep  ",
        "https://h/d/dep  /anthropic/v1/messages"
    )]
    #[case::trailing_spaces_after_anthropic_are_cut(
        "https://h/d/dep/anthropic  ",
        "https://h/d/dep/anthropic/v1/messages"
    )]
    #[case::trailing_crlf_after_anthropic_is_cut(
        "https://h/d/dep/anthropic\r\n",
        "https://h/d/dep/anthropic/v1/messages"
    )]
    #[case::leading_space_is_kept(
        " https://h/d/dep/anthropic",
        " https://h/d/dep/anthropic/v1/messages"
    )]
    #[case::blank_base_is_kept("   ", "   /anthropic/v1/messages")]
    fn url_is_built_as_python_builds_it(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            complete_azure_anthropic_url(Some(base), &|_| None).expect("url builds"),
            expected
        );
    }

    #[rstest]
    #[case::call_base_wins(
        Some("http://call/d"),
        Some("http://env/d"),
        "http://call/d/anthropic/v1/messages"
    )]
    #[case::empty_call_base_falls_to_environment(
        Some(""),
        Some("http://env/d"),
        "http://env/d/anthropic/v1/messages"
    )]
    #[case::blank_call_base_is_kept(Some("   "), Some("http://env/d"), "   /anthropic/v1/messages")]
    #[case::environment_base_is_kept_untrimmed(
        None,
        Some(" http://env/d "),
        " http://env/d /anthropic/v1/messages"
    )]
    fn url_resolves_the_base_as_python_does(
        #[case] base: Option<&str>,
        #[case] environment: Option<&str>,
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            (name == AZURE_API_BASE_ENV)
                .then(|| environment.map(str::to_string))
                .flatten()
        };
        assert_eq!(
            complete_azure_anthropic_url(base, &lookup).expect("url builds"),
            expected
        );
    }

    #[rstest]
    #[case::no_base(None)]
    #[case::empty_base(Some(""))]
    fn a_missing_base_is_python_s_value_error(#[case] base: Option<&str>) {
        assert_eq!(
            complete_azure_anthropic_url(base, &|_| None),
            Err(Error::InvalidRequest(ErrorDetail::MissingAzureApiBase))
        );
    }

    #[rstest]
    #[case::param_wins(Some("sk-param"), &[("AZURE_OPENAI_API_KEY", "sk-openai")], Some("sk-param"))]
    #[case::empty_param_falls_through(Some(""), &[("AZURE_API_KEY", "sk-env")], Some("sk-env"))]
    #[case::blank_param_is_kept_as_sent(Some("  "), &[("AZURE_API_KEY", "sk-env")], Some("  "))]
    #[case::azure_openai_key_before_azure_key(
        None,
        &[("AZURE_OPENAI_API_KEY", "sk-openai"), ("AZURE_API_KEY", "sk-env")],
        Some("sk-openai")
    )]
    #[case::empty_azure_openai_key_falls_through(
        None,
        &[("AZURE_OPENAI_API_KEY", ""), ("AZURE_API_KEY", "sk-env")],
        Some("sk-env")
    )]
    #[case::nothing(None, &[], None)]
    fn api_key_resolves_like_python(
        #[case] api_key: Option<&str>,
        #[case] env: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        assert_eq!(resolve_azure_api_key(api_key, &lookup).as_deref(), expected);
    }

    fn validated(forwarded: &[(&str, &str)], api_key: Option<&str>) -> ValidatedEnvironment {
        validated_with(forwarded, api_key, json!({}))
    }

    fn validated_with(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        params: serde_json::Value,
    ) -> ValidatedEnvironment {
        AZURE_ANTHROPIC_MESSAGES_CONFIG
            .validate_environment(
                forwarded
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                api_key,
                "claude",
                &MessagesOptionalParams::default(),
                &litellm_params(params),
                &|_| None,
            )
            .unwrap()
    }

    fn litellm_params(params: serde_json::Value) -> LitellmParams {
        let serde_json::Value::Object(params) = params else {
            panic!("case params are an object")
        };
        serde_json::from_value(serde_json::Value::Object(
            params
                .into_iter()
                .chain([("model".to_string(), json!("claude"))])
                .collect(),
        ))
        .expect("valid params")
    }

    #[rstest]
    #[case::lowercase(&[("api-key", "caller"), ("x-trace", "1")])]
    #[case::any_casing(&[("Api-Key", "caller"), ("x-trace", "1")])]
    fn a_caller_api_key_replaces_the_azure_key_as_x_api_key(#[case] forwarded: &[(&str, &str)]) {
        let environment = validated(forwarded, Some("sk-azure"));
        assert!(matches!(environment.auth, AuthScheme::Forwarded));
        assert_eq!(
            environment.headers,
            vec![
                ("x-api-key".to_string(), "caller".to_string()),
                ("x-trace".to_string(), "1".to_string()),
            ]
        );
    }

    fn lookup_in<'a>(pairs: &'a [(&str, &str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn entra_flow(params: serde_json::Value, env: &[(&str, &str)]) -> String {
        entra_flow_from(params, &[], env)
    }

    fn entra_flow_from(
        params: serde_json::Value,
        secrets: &[(&str, &str)],
        process_env: &[(&str, &str)],
    ) -> String {
        match entra_id_auth(
            &litellm_params(params),
            &lookup_in(secrets),
            &lookup_in(process_env),
        ) {
            AuthScheme::HostAzureAdToken => "host".to_string(),
            AuthScheme::AzureAdToken { inputs } => match inputs.tenant_id.as_value() {
                Some(tenant_id) => format!(
                    "client_secret:{}:{}",
                    tenant_id.value(),
                    inputs
                        .azure_scope
                        .as_value()
                        .map(|scope| scope.value().as_str())
                        .unwrap_or_default()
                ),
                None => "caller".to_string(),
            },
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret,
            } => format!("bearer:{}", secret.expose()),
            AuthScheme::Forwarded => "unauthenticated".to_string(),
            other => panic!("unexpected scheme {other:?}"),
        }
    }

    const SECRET: &str = "client_secret:t:https://cognitiveservices.azure.com/.default";

    #[rstest]
    #[case::nothing_is_unauthenticated(json!({}), &[], "unauthenticated")]
    #[case::a_provider_python_would_not_call_is_left_to_python(
        json!({"azure_ad_token": "tok", "azure_ad_token_provider": "uncallable"}),
        &[],
        "host"
    )]
    #[case::a_static_token(json!({"azure_ad_token": "tok"}), &[], "bearer:tok")]
    #[case::the_caller_provider_outranks_a_token(
        json!({"azure_ad_token": "tok", "azure_ad_token_provider": "callable"}),
        &[],
        "caller"
    )]
    #[case::client_secret(json!({"tenant_id": "t", "client_id": "c", "client_secret": "s"}), &[], SECRET)]
    #[case::client_secret_from_the_process_environment(
        json!({}),
        &[("AZURE_TENANT_ID", "t"), ("AZURE_CLIENT_ID", "c"), ("AZURE_CLIENT_SECRET", "s")],
        SECRET
    )]
    #[case::client_secret_with_the_scope_python_reads(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s"}),
        &[("AZURE_SCOPE", "api://scope/.default")],
        "client_secret:t:api://scope/.default"
    )]
    #[case::client_secret_outranks_a_token(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s", "azure_ad_token": "tok"}),
        &[],
        SECRET
    )]
    #[case::the_caller_provider_skips_client_secret(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s", "azure_ad_token_provider": "callable"}),
        &[],
        "caller"
    )]
    #[case::an_environment_reference_is_left_to_python(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "os.environ/SECRET"}),
        &[],
        "host"
    )]
    #[case::username_and_password_are_left_to_python(
        json!({"azure_username": "u", "azure_password": "p", "client_id": "c"}),
        &[],
        "host"
    )]
    #[case::username_without_a_client_id_is_not_a_flow(
        json!({"azure_username": "u", "azure_password": "p", "azure_ad_token": "tok"}),
        &[],
        "bearer:tok"
    )]
    #[case::client_secret_outranks_username_and_password(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s", "azure_username": "u", "azure_password": "p"}),
        &[],
        SECRET
    )]
    #[case::the_caller_provider_skips_username_and_password(
        json!({"azure_username": "u", "azure_password": "p", "client_id": "c", "azure_ad_token_provider": "callable"}),
        &[],
        "caller"
    )]
    #[case::an_oidc_reference_is_left_to_python(
        json!({"azure_ad_token": "oidc/env/ASSERTION", "tenant_id": "t", "client_id": "c"}),
        &[],
        "host"
    )]
    #[case::an_oidc_reference_with_a_caller_provider_is_left_to_python(
        json!({"azure_ad_token": "oidc/env/ASSERTION", "tenant_id": "t", "client_id": "c", "azure_ad_token_provider": "callable"}),
        &[],
        "host"
    )]
    #[case::an_oidc_reference_without_a_tenant_is_sent_as_a_token(
        json!({"azure_ad_token": "oidc/env/ASSERTION", "client_id": "c"}),
        &[],
        "bearer:oidc/env/ASSERTION"
    )]
    #[case::token_refresh_with_a_static_token_is_left_to_python(
        json!({"azure_ad_token": "tok", "enable_azure_ad_token_refresh": true}),
        &[],
        "host"
    )]
    #[case::token_refresh_alone_is_left_to_python(json!({"enable_azure_ad_token_refresh": true}), &[], "host")]
    #[case::token_refresh_after_the_caller_provider(
        json!({"enable_azure_ad_token_refresh": true, "azure_ad_token_provider": "callable"}),
        &[],
        "caller"
    )]
    #[case::token_refresh_after_client_secret(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s", "enable_azure_ad_token_refresh": true}),
        &[],
        SECRET
    )]
    fn entra_id_flows_follow_pythons_get_azure_ad_token_order(
        #[case] params: serde_json::Value,
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        assert_eq!(entra_flow(params, env), expected);
    }

    #[rstest]
    #[case::a_token_from_the_secret_lookup(json!({}), &[("AZURE_AD_TOKEN", "secret-tok")], &[], "bearer:secret-tok")]
    #[case::an_empty_token_param_falls_to_the_secret_lookup(
        json!({"azure_ad_token": ""}),
        &[("AZURE_AD_TOKEN", "secret-tok")],
        &[],
        "bearer:secret-tok"
    )]
    #[case::a_token_only_in_the_process_environment_is_not_read_there(
        json!({}),
        &[],
        &[("AZURE_AD_TOKEN", "env-tok")],
        "unauthenticated"
    )]
    #[case::client_secret_only_in_the_secret_lookup_is_not_used(
        json!({}),
        &[("AZURE_TENANT_ID", "t"), ("AZURE_CLIENT_ID", "c"), ("AZURE_CLIENT_SECRET", "s")],
        &[],
        "unauthenticated"
    )]
    #[case::scope_only_in_the_secret_lookup_is_not_used(
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s"}),
        &[("AZURE_SCOPE", "api://secret/.default")],
        &[],
        SECRET
    )]
    #[case::username_only_in_the_secret_lookup_is_not_used(
        json!({"client_id": "c"}),
        &[("AZURE_USERNAME", "u"), ("AZURE_PASSWORD", "p")],
        &[],
        "unauthenticated"
    )]
    #[case::username_from_the_process_environment(
        json!({"client_id": "c"}),
        &[],
        &[("AZURE_USERNAME", "u"), ("AZURE_PASSWORD", "p")],
        "host"
    )]
    fn entra_settings_are_read_where_python_reads_them(
        #[case] params: serde_json::Value,
        #[case] secrets: &[(&str, &str)],
        #[case] process_env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        assert_eq!(entra_flow_from(params, secrets, process_env), expected);
    }

    #[test]
    fn the_azure_key_goes_in_x_api_key() {
        assert!(matches!(
            validated(&[], Some("sk-azure")).auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == "sk-azure"
        ));
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::entra_id_bearer(&[("Authorization", "Bearer eyJ-token")])]
    fn a_forwarded_key_or_bearer_is_the_credential(#[case] forwarded: &[(&str, &str)]) {
        assert!(matches!(
            validated(forwarded, Some("sk-azure")).auth,
            AuthScheme::Forwarded
        ));
    }

    #[rstest]
    #[case::bearer_on_an_entra_only_deployment(
        &[("Authorization", "Bearer caller")],
        json!({"azure_ad_token": "entra"})
    )]
    #[case::lowercase_authorization_with_entra_settings(
        &[("authorization", "Bearer caller")],
        json!({"tenant_id": "t", "client_id": "c", "client_secret": "s"})
    )]
    #[case::bearer_with_python_only_entra_settings(
        &[("Authorization", "Bearer caller")],
        json!({"azure_username": "u", "azure_password": "p", "client_id": "c"})
    )]
    #[case::x_api_key_with_entra_settings(&[("x-api-key", "caller")], json!({"azure_ad_token": "entra"}))]
    #[case::mixed_case_api_key_with_entra_settings(&[("API-Key", "caller")], json!({"azure_ad_token": "entra"}))]
    #[case::non_bearer_authorization(&[("Authorization", "Basic caller")], json!({"azure_ad_token": "entra"}))]
    fn a_caller_credential_skips_the_deployment_key_and_entra(
        #[case] forwarded: &[(&str, &str)],
        #[case] params: serde_json::Value,
    ) {
        assert!(matches!(
            validated_with(forwarded, Some("sk-azure"), params.clone()).auth,
            AuthScheme::Forwarded
        ));
        assert!(matches!(
            validated_with(forwarded, None, params).auth,
            AuthScheme::Forwarded
        ));
    }

    #[test]
    fn a_blank_bearer_does_not_count_as_a_credential() {
        assert!(matches!(
            validated(&[("Authorization", "Bearer  ")], Some("sk-azure")).auth,
            AuthScheme::Credential { .. }
        ));
    }

    #[test]
    fn default_headers_match_python() {
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[test]
    fn transform_request_strips_scope_from_system_and_messages() {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "sys",
                    "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral", "scope": "global"}
                        },
                        {"type": "text", "text": "no cache control"}
                    ]
                }
            ]
        }));

        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
                .expect("request transforms"),
        );

        assert_eq!(
            transformed["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(
            transformed["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            transformed["messages"][0]["content"][1],
            json!({"type": "text", "text": "no cache control"})
        );
    }

    #[test]
    fn transform_request_is_idempotent_and_preserves_string_system() {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "system": "plain string system",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let once = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .expect("request transforms");
        let twice = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                once.clone(),
                &MessagesTransformContext::default(),
            )
            .expect("request transforms");
        assert_eq!(once, twice);
        assert_eq!(to_value(once)["system"], json!("plain string system"));
    }

    #[test]
    fn transform_request_preserves_all_supported_params() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be terse",
            "metadata": {"user_id": "u1"},
            "stop_sequences": ["STOP"],
            "stream": false,
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "service_tier": "auto",
            "container": {"id": "c1"},
            "mcp_servers": [{"type": "url", "url": "https://mcp.example", "name": "x"}],
            "context_management": {"edits": []},
            "output_format": {"type": "json_schema"},
            "output_config": {"effort": "high"},
            "speed": "fast",
            "inference_geo": "us",
            "litellm_metadata": {"trace": "abc"}
        });
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                supports_legacy_thinking: true,
                supports_output_config: true,
                supports_speed: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request_from(body.clone()), &context)
                .expect("request transforms"),
        );
        assert_eq!(transformed, body);
    }

    fn context_with(supports_mid_conversation_system: bool) -> MessagesTransformContext {
        MessagesTransformContext::new(
            MessagesModelCapabilities {
                supports_mid_conversation_system,
                ..MessagesModelCapabilities::default()
            },
            false,
        )
    }

    #[rstest]
    #[case::leading_turn_joins_the_top_level_system(
        json!({
            "system": [{"type": "text", "text": "base system"}],
            "messages": [
                {"role": "system", "content": "Available agent types: claude"},
                {"role": "user", "content": "fix the bug"}
            ]
        }),
        false,
        json!([{"type": "text", "text": "base system"}, {"type": "text", "text": "Available agent types: claude"}]),
        json!([{"role": "user", "content": "fix the bug"}]),
    )]
    #[case::leading_turn_becomes_the_system_when_there_is_none(
        json!({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys block"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]}),
        false,
        json!([{"type": "text", "text": "sys block"}]),
        json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
    )]
    #[case::billing_block_in_a_leading_turn_is_dropped(
        json!({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}, {"type": "text", "text": "keep"}]},
            {"role": "user", "content": "hi"}
        ]}),
        false,
        json!([{"type": "text", "text": "keep"}]),
        json!([{"role": "user", "content": "hi"}]),
    )]
    #[case::later_turn_is_converted_in_place_for_a_model_without_the_capability(
        json!({
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "system", "content": "reminder"}
            ]
        }),
        false,
        json!("be terse"),
        json!([
            {"role": "user", "content": "fix the bug"},
            {"role": "user", "content": [
                {"type": "text", "text": CONVERTED_SYSTEM_NOTE},
                {"type": "text", "text": "reminder"}
            ]}
        ]),
    )]
    #[case::later_turn_stays_for_a_model_with_the_capability(
        json!({
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "system", "content": "reminder"}
            ]
        }),
        true,
        json!("be terse"),
        json!([
            {"role": "user", "content": "fix the bug"},
            {"role": "system", "content": "reminder"}
        ]),
    )]
    fn transform_request_normalizes_system_role_messages_like_python(
        #[case] body: serde_json::Value,
        #[case] supports_mid_conversation_system: bool,
        #[case] system: serde_json::Value,
        #[case] messages: serde_json::Value,
    ) {
        let mut body = body;
        body["model"] = json!("claude-sonnet-4-5");
        body["max_tokens"] = json!(256);

        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(
                    request_from(body),
                    &context_with(supports_mid_conversation_system),
                )
                .expect("request transforms"),
        );

        assert_eq!(transformed["system"], system);
        assert_eq!(transformed["messages"], messages);
    }

    #[test]
    fn transform_request_leaves_requests_without_system_role_untouched() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ]
        });
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                supports_legacy_thinking: true,
                supports_output_config: true,
                supports_speed: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request_from(body.clone()), &context)
                .expect("request transforms"),
        );
        assert_eq!(transformed, body);
    }

    #[test]
    fn transform_request_rejects_non_object_body() {
        let err = serde_json::from_value::<MessagesRequest>(json!("bad"))
            .expect_err("non-object body should error");
        assert!(err.is_data());
    }

    #[rstest]
    #[case::compact_context_management_edit(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &[],
        &[("x-api-key", "k"), ("anthropic-beta", "compact-2026-01-12")]
    )]
    #[case::forwarded_beta_merged_with_structured_output(
        json!({"output_config": {"format": {"type": "json_schema"}}}),
        &[("anthropic-beta", "web-search-2025-03-05")],
        &[("x-api-key", "k"), ("anthropic-beta", "structured-outputs-2025-11-13,web-search-2025-03-05")]
    )]
    #[case::compaction_needs_the_compact_2026_09_04_beta(
        json!({"compaction": {"type": "summarize"}}),
        &[],
        &[("x-api-key", "k"), ("anthropic-beta", "compact-2026-09-04")]
    )]
    #[case::no_feature_needs_a_beta(json!({}), &[], &[("x-api-key", "k")])]
    #[case::safeguards_need_the_dangerous_tool_use_beta(
        json!({"safeguards": [{"type": "dangerous_tool_use"}]}),
        &[("anthropic-beta", "claude-code-20250219")],
        &[("x-api-key", "k"), ("anthropic-beta", "claude-code-20250219,dangerous-tool-use-2026-09-03")]
    )]
    #[case::null_safeguards_need_no_beta(json!({"safeguards": null}), &[], &[("x-api-key", "k")])]
    fn request_headers_carry_the_anthropic_feature_betas(
        #[case] fields: serde_json::Value,
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let pairs = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect()
        };
        let serde_json::Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let request = request_from(serde_json::Value::Object(
            [
                ("model".to_string(), json!("claude-sonnet")),
                ("max_tokens".to_string(), json!(16)),
                (
                    "messages".to_string(),
                    json!([{"role": "user", "content": "hi"}]),
                ),
            ]
            .into_iter()
            .chain(fields)
            .collect(),
        ));
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.request_headers(
                pairs(&[("x-api-key", "k")])
                    .into_iter()
                    .chain(pairs(forwarded))
                    .collect(),
                &request
            ),
            pairs(expected)
        );
    }

    #[test]
    fn transform_response_passes_through() {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .expect("valid response");
        let transformed = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_response("claude-sonnet-4-5", response)
            .expect("response transforms");
        let value = serde_json::to_value(transformed).expect("serializable");
        assert_eq!(value["stop_reason"], json!("end_turn"));
        assert_eq!(value["stop_sequence"], json!(null));
        assert_eq!(value["content"][0]["text"], json!("hello"));
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = std::cell::RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = AZURE_ANTHROPIC_MESSAGES_CONFIG.validate_environment(
            Vec::new(),
            None,
            "claude",
            &MessagesOptionalParams::default(),
            &LitellmParams::default(),
            &record,
        );
        let _ = AZURE_ANTHROPIC_MESSAGES_CONFIG.get_complete_url(
            None,
            "claude",
            &LitellmParams::default(),
            false,
            &record,
        );
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let deferred = AZURE_ANTHROPIC_MESSAGES_CONFIG.deferred_secret_names(
            &Vec::new(),
            None,
            None,
            &LitellmParams::default(),
            &|_| None,
        );
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !AZURE_ANTHROPIC_MESSAGES_CONFIG
                    .secret_names()
                    .contains(&name.as_str())
                    && !deferred.contains(&name.as_str())
            })
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }

    #[rstest]
    #[case::no_key_and_no_caller_credential(&[], None, json!({}), &["AZURE_AD_TOKEN"])]
    #[case::a_key(&[], Some("sk-azure"), json!({}), &[])]
    #[case::an_azure_ad_token_param(&[], None, json!({"azure_ad_token": "tok"}), &[])]
    #[case::an_empty_azure_ad_token_param(&[], None, json!({"azure_ad_token": ""}), &["AZURE_AD_TOKEN"])]
    #[case::a_caller_x_api_key(&[("X-Api-Key", "caller")], None, json!({}), &[])]
    #[case::a_caller_api_key(&[("API-Key", "caller")], None, json!({}), &[])]
    #[case::a_caller_bearer(&[("authorization", "Bearer caller")], None, json!({}), &[])]
    #[case::a_caller_basic_authorization(&[("Authorization", "Basic caller")], None, json!({}), &[])]
    #[case::a_blank_caller_bearer(&[("Authorization", "Bearer ")], None, json!({}), &["AZURE_AD_TOKEN"])]
    fn the_azure_ad_token_secret_is_read_only_where_python_reads_it(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] params: serde_json::Value,
        #[case] expected: &[&str],
    ) {
        let headers: Headers = forwarded
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.deferred_secret_names(
                &headers,
                api_key,
                Some("https://call.example"),
                &litellm_params(params),
                &|_| None
            ),
            expected
        );
    }

    #[rstest]
    #[case::a_call_base(Some("https://call.example"), &[])]
    #[case::a_blank_call_base(Some("   "), &[])]
    #[case::an_empty_call_base(Some(""), &["AZURE_API_BASE"])]
    #[case::no_call_base(None, &["AZURE_API_BASE"])]
    fn the_azure_api_base_secret_is_read_only_where_python_reads_it(
        #[case] api_base: Option<&str>,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.deferred_secret_names(
                &Vec::new(),
                Some("sk-azure"),
                api_base,
                &LitellmParams::default(),
                &|_| None
            ),
            expected
        );
    }

    #[test]
    fn without_a_key_or_a_call_base_both_deferred_secrets_are_read() {
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.deferred_secret_names(
                &Vec::new(),
                None,
                None,
                &LitellmParams::default(),
                &|_| None
            ),
            ["AZURE_AD_TOKEN", "AZURE_API_BASE"]
        );
    }
}
