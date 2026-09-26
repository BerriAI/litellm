//! How a provider call authenticates, decided by the provider config when the request is
//! prepared and applied once here when it is sent.
//!
//! Python folds this into `validate_environment` plus `sign_request`. The Rust configs keep
//! that split: `validate_environment` shapes the forwarded headers and names the credential
//! as an [`AuthScheme`], and [`resolve_auth`] turns the scheme into headers and a signer.

use litellm_auth::{AuthServices, CredentialPlacement, SecretValue, TokenProviderHandle};
use litellm_auth_aws::{AwsCredentialSource, SigV4Signer};

pub type Headers = Vec<(String, String)>;

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

/// Fills in the defaults the caller did not forward, matching Python's
/// `if name not in headers` checks.
pub fn with_default_headers(headers: Headers, defaults: &[(&str, &str)]) -> Headers {
    let missing: Vec<(String, String)> = defaults
        .iter()
        .filter(|(name, _)| {
            !headers
                .iter()
                .any(|(header, _)| header.eq_ignore_ascii_case(name))
        })
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    headers.into_iter().chain(missing).collect()
}

fn with_credential(headers: Headers, placement: CredentialPlacement, credential: &str) -> Headers {
    let name = placement.header_name();
    let value = match placement {
        CredentialPlacement::Bearer => format!("Bearer {credential}"),
        CredentialPlacement::Header(_) => credential.to_string(),
    };
    headers
        .into_iter()
        .filter(|(header, _)| !header.eq_ignore_ascii_case(name))
        .chain([(name.to_ascii_lowercase(), value)])
        .collect()
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

    #[rstest]
    #[case::nothing_forwarded(
        &[],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("x-version", "1"), ("content-type", "application/json")],
    )]
    #[case::forwarded_header_wins_in_any_case(
        &[("X-Version", "custom"), ("x-api-key", "k")],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("X-Version", "custom"), ("x-api-key", "k"), ("content-type", "application/json")],
    )]
    #[case::no_defaults(&[("x-api-key", "k")], &[], &[("x-api-key", "k")])]
    fn default_headers_fill_only_missing_names(
        #[case] forwarded: &[(&str, &str)],
        #[case] defaults: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            with_default_headers(headers(forwarded), defaults),
            headers(expected)
        );
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
