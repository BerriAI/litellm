//! How a provider call authenticates, decided by the provider config when the request is
//! prepared and applied once here when it is sent.
//!
//! Python folds this into `validate_environment` plus `sign_request`. The Rust configs keep
//! that split: `validate_environment` shapes the forwarded headers and names the credential
//! as an [`AuthScheme`], and [`resolve_auth`] turns the scheme into headers and a signer.

use litellm_auth::{AuthServices, CredentialPlacement, TokenProviderHandle};
use litellm_auth_azure::AzureAuthInputs;
use litellm_secret_value::SecretValue;

use litellm_auth_aws::{AwsCredentialSource, SigV4Signer};
use litellm_auth_gcp::VertexConfig;
use litellm_http::request::with_header;

pub type Headers = Vec<(String, String)>;

const ENTRA_AUTHORIZATION_HEADER: &str = "Authorization";

#[derive(Clone, Debug)]
pub enum AuthScheme {
    /// The caller's own credential is already in the headers and is sent as is.
    Forwarded,
    /// A credential in hand, placed in its header. A forwarded header of the same name is
    /// replaced: the deployment's identity outranks the caller's.
    Credential {
        placement: CredentialPlacement,
        secret: SecretValue,
    },
    /// A bearer acquired when the request is sent, from a token source such as a cloud SDK
    /// or a caller-supplied callable.
    Token { provider: TokenProviderHandle },
    /// An Entra ID token minted from the config's Azure inputs when the request is sent, as
    /// Python's `get_azure_ad_token` does. Inputs that yield no token leave the call
    /// unauthenticated.
    AzureAdToken { inputs: Box<AzureAuthInputs> },
    /// A configured Entra ID token (`azure_ad_token`), set under Python's `Authorization` key
    /// as an Entra ID token minted at send time is.
    AzureAdTokenValue { secret: SecretValue },
    /// An Entra ID token only Python's `get_azure_ad_token` can mint (username and password,
    /// OIDC references, token refresh), asked of the host when the request is sent. A host
    /// answer without a token leaves the call unauthenticated.
    HostAzureAdToken,
    /// Entra ID auth for a call without a key, which only a call with a Python host runs
    /// ([`AzureAdTokenHosts::host`]). Without one the call is what it was when the config took
    /// only a key: `Ok` forwards the caller's own credential, `Err` is the missing key.
    PythonHostedAzureAdToken {
        entra: Box<AuthScheme>,
        unhosted: Result<(), litellm_auth::Error>,
    },
    /// A Google access token for the Vertex AI project the config names, acquired when the
    /// request is sent through the shared GCP token cache.
    GcpAccessToken { config: Box<VertexConfig> },
    /// AWS SigV4 over the bytes that go on the wire, so the handler signs after the body is
    /// serialized.
    AwsSigV4 {
        region: String,
        service: &'static str,
        credentials: Box<AwsCredentialSource>,
    },
}

/// The outcome of a config's `validate_environment`: the headers it shaped and how the
/// call authenticates.
#[derive(Clone, Debug)]
pub struct ValidatedEnvironment {
    pub headers: Headers,
    pub auth: AuthScheme,
}

/// The Entra ID token sources a call's host supplies: the caller's `azure_ad_token_provider`
/// and the host's own `get_azure_ad_token`.
#[derive(Clone, Debug, Default)]
pub struct AzureAdTokenHosts {
    pub caller: Option<TokenProviderHandle>,
    pub host: Option<TokenProviderHandle>,
}

impl ValidatedEnvironment {
    /// Hands an Entra ID scheme the token source it names. Other schemes never read them.
    pub fn with_azure_ad_token_hosts(self, hosts: AzureAdTokenHosts) -> Result<Self, crate::Error> {
        let auth = match (self.auth, &hosts.host) {
            (AuthScheme::PythonHostedAzureAdToken { unhosted, .. }, None) => {
                unhosted.map(|()| AuthScheme::Forwarded)?
            }
            (AuthScheme::PythonHostedAzureAdToken { entra, .. }, Some(_)) => *entra,
            (auth, _) => auth,
        };
        let provider = match &auth {
            AuthScheme::AzureAdToken { .. } => hosts.caller,
            AuthScheme::HostAzureAdToken => hosts.host,
            _ => None,
        };
        Ok(match (auth, provider) {
            (AuthScheme::AzureAdToken { inputs }, Some(provider)) => Self {
                auth: AuthScheme::AzureAdToken {
                    inputs: Box::new(AzureAuthInputs {
                        azure_ad_token_provider: Some(provider),
                        ..*inputs
                    }),
                },
                headers: self.headers,
            },
            (AuthScheme::HostAzureAdToken, Some(provider)) => Self {
                auth: AuthScheme::AzureAdToken {
                    inputs: Box::new(AzureAuthInputs {
                        azure_ad_token_provider: Some(provider),
                        ..AzureAuthInputs::default()
                    }),
                },
                headers: self.headers,
            },
            (auth, _) => Self {
                auth,
                headers: self.headers,
            },
        })
    }
}

#[derive(Debug)]
pub struct Authenticated {
    pub headers: Headers,
    pub signer: Option<SigV4Signer>,
}

pub async fn resolve_auth(
    services: &AuthServices,
    validated: ValidatedEnvironment,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Authenticated, litellm_auth::Error> {
    let ValidatedEnvironment { headers, auth } = validated;
    match auth {
        AuthScheme::Forwarded => Ok(Authenticated {
            headers,
            signer: None,
        }),
        AuthScheme::Credential { placement, secret } => Ok(Authenticated {
            headers: with_credential(headers, placement, secret.expose()),
            signer: None,
        }),
        AuthScheme::Token { provider } => {
            let token = provider.acquire().await?;
            Ok(Authenticated {
                headers: with_credential(
                    headers,
                    CredentialPlacement::Bearer,
                    token.secret().expose(),
                ),
                signer: None,
            })
        }
        AuthScheme::AzureAdToken { inputs } => {
            let token = services
                .azure
                .get_azure_ad_token(&inputs, env_lookup)
                .await
                .or_else(|error| match error {
                    litellm_auth::Error::EmptyCallerCredential { .. } => Ok(None),
                    other => Err(other),
                })?;
            Ok(Authenticated {
                headers: match token {
                    Some(token) => with_entra_bearer(headers, token.value().secret().expose()),
                    None => headers,
                },
                signer: None,
            })
        }
        AuthScheme::AzureAdTokenValue { secret } => Ok(Authenticated {
            headers: with_entra_bearer(headers, secret.expose()),
            signer: None,
        }),
        AuthScheme::PythonHostedAzureAdToken { unhosted, .. } => unhosted.map(|()| Authenticated {
            headers,
            signer: None,
        }),
        AuthScheme::HostAzureAdToken => Err(litellm_auth::Error::CredentialAcquisition(
            litellm_auth::ErrorDetail::Unresolved {
                subject: "Azure AD token without a Python host",
            },
        )),
        AuthScheme::GcpAccessToken { config } => {
            let token = services.gcp.access_token(&config, env_lookup).await?;
            Ok(Authenticated {
                headers: with_credential(headers, CredentialPlacement::Bearer, &token),
                signer: None,
            })
        }
        AuthScheme::AwsSigV4 {
            region,
            service,
            credentials,
        } => Ok(Authenticated {
            headers,
            signer: Some(
                SigV4Signer::resolve(&services.aws, region, service, *credentials, env_lookup)
                    .await?,
            ),
        }),
    }
}

/// Python's `_base_validate_azure_environment` sets `headers["Authorization"]`: only that exact
/// key is replaced, and a caller `authorization` in another casing is joined with it on the wire.
fn with_entra_bearer(headers: Headers, token: &str) -> Headers {
    headers
        .into_iter()
        .filter(|(name, _)| name != ENTRA_AUTHORIZATION_HEADER)
        .chain([(
            ENTRA_AUTHORIZATION_HEADER.to_string(),
            format!("Bearer {token}"),
        )])
        .collect()
}

fn with_credential(headers: Headers, placement: CredentialPlacement, credential: &str) -> Headers {
    let name = placement.header_name();
    let value = match placement {
        CredentialPlacement::Bearer => format!("Bearer {credential}"),
        CredentialPlacement::Header(_) => credential.to_string(),
    };
    with_header(headers, name, value)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_auth::{AuthServices, ResolvedCredential, TokenFuture, TokenProvider};
    use litellm_auth_aws::Credentials;
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    async fn resolve(headers: Headers, auth: AuthScheme) -> Authenticated {
        resolve_auth(
            &AuthServices::default(),
            ValidatedEnvironment { headers, auth },
            &no_env,
        )
        .await
        .unwrap()
    }

    #[rstest]
    #[case::header_is_appended(
        &[("content-type", "application/json")],
        CredentialPlacement::Header("x-api-key"),
        &[("content-type", "application/json"), ("x-api-key", "sk")],
    )]
    #[case::forwarded_header_of_the_same_name_is_replaced_in_any_casing(
        &[("X-Api-Key", "caller"), ("x-trace", "1")],
        CredentialPlacement::Header("x-api-key"),
        &[("x-trace", "1"), ("x-api-key", "sk")],
    )]
    #[case::bearer_replaces_a_forwarded_authorization(
        &[("Authorization", "Bearer caller")],
        CredentialPlacement::Bearer,
        &[("authorization", "Bearer sk")],
    )]
    #[tokio::test]
    async fn a_credential_lands_in_its_header_and_outranks_the_forwarded_one(
        #[case] forwarded: &[(&str, &str)],
        #[case] placement: CredentialPlacement,
        #[case] expected: &[(&str, &str)],
    ) {
        let authenticated = resolve(
            headers(forwarded),
            AuthScheme::Credential {
                placement,
                secret: SecretValue::new("sk"),
            },
        )
        .await;
        assert_eq!(authenticated.headers, headers(expected));
        assert!(authenticated.signer.is_none());
    }

    #[tokio::test]
    async fn forwarded_auth_sends_the_headers_untouched() {
        let forwarded = headers(&[("x-api-key", "caller"), ("authorization", "Bearer caller")]);
        let authenticated = resolve(forwarded.clone(), AuthScheme::Forwarded).await;
        assert_eq!(authenticated.headers, forwarded);
        assert!(authenticated.signer.is_none());
    }

    #[derive(Debug)]
    struct StaticToken(&'static str);

    impl TokenProvider for StaticToken {
        fn acquire(&self) -> TokenFuture<'_> {
            Box::pin(async move {
                Ok(ResolvedCredential::AccessToken {
                    token: SecretValue::new(self.0),
                    expires_on: None,
                })
            })
        }
    }

    #[tokio::test]
    async fn a_token_is_acquired_at_send_time_and_sent_as_a_bearer() {
        let authenticated = resolve(
            headers(&[("authorization", "Bearer stale")]),
            AuthScheme::Token {
                provider: TokenProviderHandle::new(Arc::new(StaticToken("fresh"))),
            },
        )
        .await;
        assert_eq!(
            authenticated.headers,
            headers(&[("authorization", "Bearer fresh")])
        );
    }

    fn entra_inputs(token: Option<&str>) -> AzureAuthInputs {
        AzureAuthInputs {
            azure_ad_token: token.map_or(litellm_auth_azure::ConfigValue::Absent, |token| {
                litellm_auth_azure::ConfigValue::Value(litellm_auth::Sourced::new(
                    SecretValue::new(token),
                    litellm_auth::InputSource::Deployment,
                ))
            }),
            ..AzureAuthInputs::default()
        }
    }

    #[rstest]
    #[case::a_configured_token_is_sent_as_a_bearer(Some("entra"), None, &[("Authorization", "Bearer entra")])]
    #[case::the_caller_provider_outranks_a_configured_token(
        Some("entra"),
        Some("from-provider"),
        &[("Authorization", "Bearer from-provider")]
    )]
    #[case::an_empty_provider_token_sends_no_credential(None, Some(""), &[])]
    #[case::no_entra_input_sends_no_credential(None, None, &[])]
    #[tokio::test]
    async fn an_entra_scheme_sends_the_token_python_would_mint(
        #[case] token: Option<&'static str>,
        #[case] provider: Option<&'static str>,
        #[case] expected: &[(&str, &str)],
    ) {
        let validated = ValidatedEnvironment {
            headers: Vec::new(),
            auth: AuthScheme::AzureAdToken {
                inputs: Box::new(entra_inputs(token)),
            },
        }
        .with_azure_ad_token_hosts(AzureAdTokenHosts {
            caller: provider.map(|token| TokenProviderHandle::new(Arc::new(StaticToken(token)))),
            host: None,
        })
        .unwrap();
        let authenticated = resolve(validated.headers, validated.auth).await;
        assert_eq!(authenticated.headers, headers(expected));
    }

    #[rstest]
    #[case::pythons_exact_key_is_replaced(
        &[("Authorization", "Bearer caller"), ("x-trace", "1")],
        &[("x-trace", "1"), ("Authorization", "Bearer entra")]
    )]
    #[case::another_casing_stays_beside_it(
        &[("authorization", "Bearer caller"), ("x-api-key", "caller")],
        &[("authorization", "Bearer caller"), ("x-api-key", "caller"), ("Authorization", "Bearer entra")]
    )]
    #[tokio::test]
    async fn an_entra_token_is_set_under_pythons_authorization_key(
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let authenticated = resolve(
            headers(forwarded),
            AuthScheme::AzureAdToken {
                inputs: Box::new(entra_inputs(Some("entra"))),
            },
        )
        .await;
        assert_eq!(authenticated.headers, headers(expected));
    }

    #[rstest]
    #[case::the_host_token_is_sent_as_a_bearer(Some("from-python"), &[("Authorization", "Bearer from-python")])]
    #[case::no_host_token_sends_no_credential(Some(""), &[])]
    #[tokio::test]
    async fn a_host_entra_scheme_sends_the_token_python_minted(
        #[case] host: Option<&'static str>,
        #[case] expected: &[(&str, &str)],
    ) {
        let validated = ValidatedEnvironment {
            headers: Vec::new(),
            auth: AuthScheme::HostAzureAdToken,
        }
        .with_azure_ad_token_hosts(AzureAdTokenHosts {
            caller: Some(TokenProviderHandle::new(Arc::new(StaticToken("caller")))),
            host: host.map(|token| TokenProviderHandle::new(Arc::new(StaticToken(token)))),
        })
        .unwrap();
        let authenticated = resolve(validated.headers, validated.auth).await;
        assert_eq!(authenticated.headers, headers(expected));
    }

    #[tokio::test]
    async fn a_host_entra_scheme_without_a_host_fails() {
        let error = resolve_auth(
            &AuthServices::default(),
            ValidatedEnvironment {
                headers: Vec::new(),
                auth: AuthScheme::HostAzureAdToken,
            },
            &no_env,
        )
        .await
        .expect_err("no host to ask");
        assert!(matches!(
            error,
            litellm_auth::Error::CredentialAcquisition(_)
        ));
    }

    #[tokio::test]
    async fn sigv4_leaves_the_headers_to_the_signer() {
        let forwarded = headers(&[("x-request-id", "abc")]);
        let authenticated = resolve(
            forwarded.clone(),
            AuthScheme::AwsSigV4 {
                region: "us-east-1".into(),
                service: "bedrock",
                credentials: Box::new(AwsCredentialSource::HostSupplied(Credentials::new(
                    "AKIDEXAMPLE",
                    "secret",
                    None,
                    None,
                    "test",
                ))),
            },
        )
        .await;
        assert_eq!(authenticated.headers, forwarded);
        assert!(authenticated.signer.is_some());
    }
}
