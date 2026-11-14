use crate::anthropic::messages::transformation::retry_anthropic_messages_after_http_error;
use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::{
    AzureAuthInputs, AzureTokenRetry, ConfigValue, DEFAULT_AZURE_SCOPE, TokenTransport,
};
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
        beta_gated_fields::strip_beta_gated_system_messages,
        beta_headers::BetaPolicy,
        common_utils::{API_KEY_PLACEMENT, existing_betas, strip_trailing_slashes},
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
            context::{MessagesShapeOptions, MessagesTransformContext, UnparsedMessages},
            normalization::{normalize_system_role_messages, strip_cache_control_scope},
            transformation::BaseMessagesConfig,
        },
    },
};

const ANTHROPIC_PATH_SEGMENT: &str = "/anthropic";
const SCHEME_SEPARATOR: &str = "://";
const AZURE_API_KEY_HEADER: &str = "api-key";
const OIDC_PREFIX: &str = "oidc/";
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

    fn logs_dropped_params(&self) -> bool {
        true
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
        shape_anthropic_messages_request(request, options.reading_unparsed_messages())
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
            request,
            &context
                .keeping_display_updates()
                .reading_unparsed_messages(),
        )
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[AZURE_OPENAI_API_KEY_ENV, AZURE_API_KEY_ENV]
    }

    /// `AZURE_AD_TOKEN`, which Python reads through `get_secret_str` only when there is no key,
    /// no exact caller `api-key` and no `azure_ad_token` param, and `AZURE_API_BASE`, which it
    /// reads only when the call's base is empty
    /// (`api_base or get_secret_str("AZURE_API_BASE")`). Without a Python host the call never
    /// reaches Entra ID auth, so the token is not read, as on main.
    fn deferred_secret_names(
        &self,
        python_host: bool,
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
        let reads_token = python_host
            && !(has_exact_header(headers, AZURE_API_KEY_HEADER)
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

    /// Python's `_base_validate_azure_environment` over a dict keyed by exact name: a caller
    /// `api-key` stops it, otherwise the Azure key is added as `api-key` or, without one, an
    /// Entra ID token as `Authorization`, beside or over the caller's headers as that dict would
    /// hold them. `api-key` is then renamed to `x-api-key` unless `x-api-key` is already there, and
    /// the client joins names that differ only in case. Without a key or a Python host, main's
    /// key-only check stands: a caller `x-api-key` or bearer is forwarded, else the key is missing.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let forwarded = |headers: Headers| {
            Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            })
        };
        if has_exact_header(&headers, AZURE_API_KEY_HEADER) {
            return forwarded(with_api_key_as_x_api_key(headers));
        }
        let auth = match resolve_azure_api_key(api_key, env_lookup) {
            Some(key)
                if has_header(&headers, AZURE_API_KEY_HEADER)
                    || has_header(&headers, API_KEY_PLACEMENT.header_name()) =>
            {
                return forwarded(with_api_key_as_x_api_key(
                    headers
                        .into_iter()
                        .chain([(AZURE_API_KEY_HEADER.to_string(), key)])
                        .collect(),
                ));
            }
            Some(key) => AuthScheme::Credential {
                placement: API_KEY_PLACEMENT,
                secret: SecretValue::new(key),
            },
            None => AuthScheme::PythonHostedAzureAdToken {
                entra: Box::new(entra_id_auth(litellm_params, env_lookup, &|name| {
                    std::env::var(name).ok()
                })),
                unhosted: match has_header(&headers, API_KEY_PLACEMENT.header_name())
                    || has_bearer_auth(&headers)
                {
                    true => Ok(()),
                    false => Err(litellm_auth::Error::MissingApiKey {
                        provider: "Azure",
                        environment_variable: AZURE_API_KEY_ENV,
                    }),
                },
            },
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    /// Python's `_update_headers_with_anthropic_beta`, then the handler's
    /// `update_headers_with_filtered_beta` over the azure_ai row: unknown and unsupported betas
    /// are dropped, the caller's with the ones the request needs.
    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        BetaPolicy::Filter(BetaProvider::AzureAi).apply(update_headers_with_safeguards_beta(
            headers,
            request,
            BetaProvider::AzureAi,
            UnparsedMessages::Read,
        ))
    }

    /// Python's transform strips the system fields the final header does not unlock, then
    /// normalizes system roles and drops `cache_control.scope`, all after the betas were chosen
    /// from the request as sent.
    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: &Headers,
        context: &MessagesTransformContext,
    ) -> MessagesRequest {
        let stripped = MessagesRequest {
            messages: strip_beta_gated_system_messages(
                request.messages,
                &existing_betas(headers),
                context.drops_unsupported_params,
                &request.model,
                BetaProvider::AzureAi,
            ),
            ..request
        };
        strip_cache_control_scope(normalize_system_role_messages(
            stripped,
            context
                .thinking
                .capabilities
                .supports_mid_conversation_system,
            context.drops_unsupported_params,
            BetaProvider::AzureAi,
        ))
    }
}

fn has_exact_header(headers: &Headers, name: &str) -> bool {
    headers.iter().any(|(key, _)| key == name)
}

/// Python's `headers["x-api-key"] = headers.pop("api-key")`, run only when `x-api-key` is not
/// already a key: the exact `api-key` moves to the end as `x-api-key`.
fn with_api_key_as_x_api_key(headers: Headers) -> Headers {
    let x_api_key = API_KEY_PLACEMENT.header_name();
    match has_exact_header(&headers, x_api_key) {
        true => headers,
        false => {
            let (api_key, rest): (Headers, Headers) = headers
                .into_iter()
                .partition(|(name, _)| name == AZURE_API_KEY_HEADER);
            rest.into_iter()
                .chain(
                    api_key
                        .into_iter()
                        .map(|(_, value)| (x_api_key.to_string(), value)),
                )
                .collect()
        }
    }
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
                token_retry: AzureTokenRetry::Python,
                token_transport: TokenTransport::PythonRequests,
                ..AzureAuthInputs::default()
            }),
        };
    }
    match token {
        Some(token) => AuthScheme::AzureAdTokenValue {
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

    fn finalized(
        request: MessagesRequest,
        headers: &[(&str, &str)],
        context: &MessagesTransformContext,
    ) -> serde_json::Value {
        let headers: Headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        to_value(AZURE_ANTHROPIC_MESSAGES_CONFIG.finalize_request(request, &headers, context))
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

    fn pairs_of(headers: &[(&str, &str)]) -> Headers {
        headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[rstest]
    #[case::a06_x_api_key_beside_the_operator_api_key(
        &[("x-api-key", "caller"), ("x-trace", "1")],
        &[("x-api-key", "caller"), ("x-trace", "1"), ("api-key", "sk-azure")]
    )]
    #[case::a07_other_casing_x_api_key_beside_the_operator_x_api_key(
        &[("X-Api-Key", "caller")],
        &[("X-Api-Key", "caller"), ("x-api-key", "sk-azure")]
    )]
    #[case::a08_bearer_beside_the_operator_x_api_key(
        &[("Authorization", "Bearer caller")],
        &[("Authorization", "Bearer caller"), ("x-api-key", "sk-azure")]
    )]
    #[case::a09_exact_api_key_replaces_the_operator_key(
        &[("api-key", "caller"), ("x-trace", "1")],
        &[("x-trace", "1"), ("x-api-key", "caller")]
    )]
    #[case::a09b_other_casing_api_key_beside_the_operator_x_api_key(
        &[("Api-Key", "caller"), ("x-trace", "1")],
        &[("Api-Key", "caller"), ("x-trace", "1"), ("x-api-key", "sk-azure")]
    )]
    #[case::non_bearer_authorization_beside_the_operator_x_api_key(
        &[("Authorization", "Basic caller")],
        &[("Authorization", "Basic caller"), ("x-api-key", "sk-azure")]
    )]
    #[case::exact_api_key_and_x_api_key_are_both_kept(
        &[("api-key", "caller"), ("x-api-key", "other")],
        &[("api-key", "caller"), ("x-api-key", "other")]
    )]
    fn caller_and_operator_keys_are_sent_as_python_sends_them(
        #[case] forwarded: &[(&str, &str)],
        #[case] sent: &[(&str, &str)],
    ) {
        let environment = validated(forwarded, Some("sk-azure"));
        let headers = match environment.auth {
            AuthScheme::Forwarded => environment.headers,
            AuthScheme::Credential { placement, secret } => [
                environment.headers,
                vec![(
                    placement.header_name().to_string(),
                    secret.expose().to_string(),
                )],
            ]
            .concat(),
            other => panic!("unexpected auth {other:?}"),
        };
        assert_eq!(headers, pairs_of(sent));
    }

    #[rstest]
    #[case::lowercase_moves_to_the_end_as_x_api_key(
        &[("api-key", "caller"), ("x-trace", "1")],
        &[("x-trace", "1"), ("x-api-key", "caller")]
    )]
    #[case::mixed_case(&[("API-Key", "caller"), ("x-trace", "1")], &[("API-Key", "caller"), ("x-trace", "1")])]
    #[case::title_case(&[("Api-Key", "caller"), ("x-trace", "1")], &[("Api-Key", "caller"), ("x-trace", "1")])]
    fn a_caller_api_key_without_an_azure_key_is_sent_as_python_sends_it(
        #[case] forwarded: &[(&str, &str)],
        #[case] sent: &[(&str, &str)],
    ) {
        let environment = validated(forwarded, None);
        let credential = match environment.auth {
            AuthScheme::PythonHostedAzureAdToken { entra, .. } => *entra,
            auth => auth,
        };
        assert!(matches!(credential, AuthScheme::Forwarded));
        assert_eq!(environment.headers, pairs_of(sent));
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
                    "client_secret:{}:{}:{:?}:{:?}",
                    tenant_id.value(),
                    inputs
                        .azure_scope
                        .as_value()
                        .map(|scope| scope.value().as_str())
                        .unwrap_or_default(),
                    inputs.token_retry,
                    inputs.token_transport
                ),
                None => "caller".to_string(),
            },
            AuthScheme::AzureAdTokenValue { secret } => format!("bearer:{}", secret.expose()),
            AuthScheme::Forwarded => "unauthenticated".to_string(),
            other => panic!("unexpected scheme {other:?}"),
        }
    }

    const SECRET: &str =
        "client_secret:t:https://cognitiveservices.azure.com/.default:Python:PythonRequests";

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
        "client_secret:t:api://scope/.default:Python:PythonRequests"
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
    #[case::a10b_other_casing_bearer_is_joined_with_the_entra_bearer(
        &[("authorization", "Bearer caller")],
        &[("authorization", "Bearer caller"), ("Authorization", "Bearer entra")]
    )]
    #[case::a10c_x_api_key_beside_the_entra_bearer(
        &[("x-api-key", "caller")],
        &[("x-api-key", "caller"), ("Authorization", "Bearer entra")]
    )]
    #[case::other_casing_api_key_beside_the_entra_bearer(
        &[("API-Key", "caller")],
        &[("API-Key", "caller"), ("Authorization", "Bearer entra")]
    )]
    #[case::a10_exact_bearer_is_overwritten_by_the_entra_bearer(
        &[("Authorization", "Bearer caller"), ("x-trace", "1")],
        &[("x-trace", "1"), ("Authorization", "Bearer entra")]
    )]
    #[case::other_casing_basic_is_joined_with_the_entra_bearer(
        &[("authorization", "Basic caller")],
        &[("authorization", "Basic caller"), ("Authorization", "Bearer entra")]
    )]
    #[tokio::test]
    async fn caller_credentials_are_sent_beside_the_entra_token_as_python_sends_them(
        #[case] forwarded: &[(&str, &str)],
        #[case] sent: &[(&str, &str)],
    ) {
        let ValidatedEnvironment { headers, auth } =
            validated_with(forwarded, None, json!({"azure_ad_token": "entra"}));
        let AuthScheme::PythonHostedAzureAdToken { entra, .. } = auth else {
            panic!("a call without a key runs Entra ID with a Python host, got {auth:?}")
        };
        let authenticated = crate::base_llm::auth::resolve_auth(
            &litellm_auth::AuthServices::default(),
            ValidatedEnvironment {
                headers,
                auth: *entra,
            },
            &|_| None,
        )
        .await
        .expect("a static token resolves");
        assert_eq!(authenticated.headers, pairs_of(sent));
    }

    #[test]
    fn a_blank_bearer_does_not_count_as_a_credential() {
        assert!(matches!(
            validated(&[("Authorization", "Bearer  ")], Some("sk-azure")).auth,
            AuthScheme::Credential { .. }
        ));
    }

    #[test]
    fn finalize_request_strips_scope_from_system_and_messages() {
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

        let transformed = finalized(request, &[], &MessagesTransformContext::default());

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

    #[rstest]
    #[case::string(json!("plain string system"), Some(json!([{"type": "text", "text": "plain string system"}])))]
    #[case::empty_string(json!(""), None)]
    #[case::billing_string(json!("x-anthropic-billing-header: cc_version=1"), None)]
    fn finalize_request_sends_the_system_as_blocks_like_python(
        #[case] system: serde_json::Value,
        #[case] expected: Option<serde_json::Value>,
    ) {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "system": system,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let once = finalized(request, &[], &MessagesTransformContext::default());
        assert_eq!(once.get("system").cloned(), expected);
        assert_eq!(
            finalized(
                request_from(once.clone()),
                &[],
                &MessagesTransformContext::default()
            ),
            once
        );
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

    fn dropping_context_with(supports_mid_conversation_system: bool) -> MessagesTransformContext {
        MessagesTransformContext {
            drops_unsupported_params: true,
            ..context_with(supports_mid_conversation_system)
        }
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
        json!([{"type": "text", "text": "be terse"}]),
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
        json!([{"type": "text", "text": "be terse"}]),
        json!([
            {"role": "user", "content": "fix the bug"},
            {"role": "system", "content": "reminder"}
        ]),
    )]
    fn finalize_request_normalizes_system_role_messages_like_python(
        #[case] body: serde_json::Value,
        #[case] supports_mid_conversation_system: bool,
        #[case] system: serde_json::Value,
        #[case] messages: serde_json::Value,
    ) {
        let mut body = body;
        body["model"] = json!("claude-sonnet-4-5");
        body["max_tokens"] = json!(256);

        let transformed = finalized(
            request_from(body),
            &[],
            &context_with(supports_mid_conversation_system),
        );

        assert_eq!(transformed["system"], system);
        assert_eq!(transformed["messages"], messages);
    }

    fn reminder(system: serde_json::Value) -> serde_json::Value {
        json!({
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                system,
                {"role": "user", "content": "go on"}
            ]
        })
    }

    #[rstest]
    #[case::clear_at_without_its_beta(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        &[],
        json!({"role": "system", "content": "note"})
    )]
    #[case::clear_at_with_its_beta(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        &[("anthropic-beta", "mid-conversation-system-clear-at-2026-08-21")],
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}})
    )]
    #[case::tool_addition_without_a_beta(
        json!({"role": "system", "content": [{"type": "text", "text": "note"}, {"type": "tool_addition", "tool": {"name": "t"}}]}),
        &[("anthropic-beta", "context-1m-2025-08-07")],
        json!({"role": "system", "content": [{"type": "text", "text": "note"}]})
    )]
    #[case::tool_reference_with_the_tool_changes_beta(
        json!({"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "tool_name": "t"}}]}),
        &[("anthropic-beta", "mid-conversation-tool-changes-2026-07-01")],
        json!({"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "tool_name": "t"}}]})
    )]
    fn finalize_request_strips_what_the_final_betas_do_not_unlock_under_drop_params(
        #[case] system: serde_json::Value,
        #[case] headers: &[(&str, &str)],
        #[case] expected: serde_json::Value,
    ) {
        assert_eq!(
            finalized(
                request_from(reminder(system)),
                headers,
                &dropping_context_with(true)
            )["messages"][2],
            expected
        );
    }

    #[rstest]
    fn finalize_request_keeps_gated_fields_without_drop_params() {
        let system =
            json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}});
        assert_eq!(
            finalized(
                request_from(reminder(system.clone())),
                &[],
                &context_with(true)
            )["messages"][2],
            system
        );
    }

    #[rstest]
    fn finalize_request_strips_a_leading_turn_before_hoisting_it() {
        let request = request_from(json!({
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "be terse"}, {"type": "tool_addition", "tool": {"name": "t"}}]},
                {"role": "user", "content": "hi"}
            ]
        }));
        assert_eq!(
            finalized(request, &[], &dropping_context_with(true))["system"],
            json!([{"type": "text", "text": "be terse"}])
        );
    }

    #[rstest]
    fn finalize_request_strips_a_converted_turn_whatever_the_betas() {
        let tool_reference =
            json!({"type": "tool_addition", "tool": {"type": "tool_reference", "tool_name": "t"}});
        let request = request_from(reminder(
            json!({"role": "system", "content": [{"type": "text", "text": "note"}, tool_reference]}),
        ));
        assert_eq!(
            finalized(
                request,
                &[("anthropic-beta", "mid-conversation-tool-changes-2026-07-01")],
                &dropping_context_with(false)
            )["messages"][2]["content"],
            json!([
                {"type": "text", "text": CONVERTED_SYSTEM_NOTE},
                {"type": "text", "text": "note"}
            ])
        );
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
    #[case::unknown_and_unsupported_caller_betas_are_dropped(
        json!({}),
        &[("anthropic-beta", " web-search-2025-03-05 ,bash_20250124,unlisted-2030-01-01,,context-1m-2025-08-07")],
        &[("x-api-key", "k"), ("anthropic-beta", "context-1m-2025-08-07,web-search-2025-03-05")]
    )]
    #[case::a_header_left_empty_is_removed(
        json!({}),
        &[("ANTHROPIC-BETA", "mcp-servers-2025-12-04,unlisted-2030-01-01")],
        &[("x-api-key", "k")]
    )]
    #[case::caller_casings_are_joined_into_one_header(
        json!({}),
        &[("Anthropic-Beta", "effort-2025-11-24,text_editor_20250124"), ("anthropic-beta", "context-1m-2025-08-07")],
        &[("x-api-key", "k"), ("anthropic-beta", "context-1m-2025-08-07,effort-2025-11-24")]
    )]
    #[case::the_advisor_beta_is_not_sent(
        json!({"tools": [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"}]}),
        &[],
        &[("x-api-key", "k")]
    )]
    #[case::feature_betas_survive_beside_an_unknown_caller_beta(
        json!({"output_format": {"type": "json_schema"}, "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "updates"}}),
        &[("anthropic-beta", "unlisted-2030-02-02")],
        &[("x-api-key", "k"), ("anthropic-beta", "structured-outputs-2025-11-13,thinking-display-updates-2026-08-18")]
    )]
    fn request_headers_carry_the_feature_betas_filtered_through_the_azure_ai_row(
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
            true,
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
    #[case::a_caller_x_api_key(&[("X-Api-Key", "caller")], None, json!({}), &["AZURE_AD_TOKEN"])]
    #[case::a_caller_other_casing_api_key(&[("API-Key", "caller")], None, json!({}), &["AZURE_AD_TOKEN"])]
    #[case::a_caller_exact_api_key(&[("api-key", "caller")], None, json!({}), &[])]
    #[case::a_caller_other_casing_bearer(&[("authorization", "Bearer caller")], None, json!({}), &["AZURE_AD_TOKEN"])]
    #[case::a_caller_exact_bearer(&[("Authorization", "Bearer caller")], None, json!({}), &["AZURE_AD_TOKEN"])]
    #[case::a_caller_basic_authorization(&[("Authorization", "Basic caller")], None, json!({}), &["AZURE_AD_TOKEN"])]
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
                true,
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
                true,
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
                true,
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
