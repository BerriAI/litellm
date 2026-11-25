use std::sync::Arc;

use litellm_auth_types::{CredentialPlacement, Error, http::apply_credential};

use crate::{
    config::{VertexConfig, credential_source, get_vertex_ai_project, non_empty_env},
    constants::{VERTEX_AI_API_KEY_ENV, VERTEXAI_API_KEY_ENV},
    identity::{GcpAuth, GcpProviderLoader, GcpTokenSource},
};

pub struct VertexEnvironment {
    pub headers: Vec<(String, String)>,
    pub project_id: String,
}

#[derive(Clone, Default)]
pub struct VertexAuth {
    identity: GcpAuth,
}

impl VertexAuth {
    pub fn new(loader: Arc<dyn GcpProviderLoader>) -> Self {
        Self {
            identity: GcpAuth::new(loader),
        }
    }

    pub async fn access_token(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<String, Error> {
        self.identity
            .access_token(credential_source(config, env_lookup))
            .await
            .map(|token| token.expose().to_owned())
    }

    pub async fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<VertexEnvironment, Error> {
        let has_authorization = headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("Authorization"));
        let static_token = api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| non_empty_env(env_lookup, VERTEX_AI_API_KEY_ENV))
            .or_else(|| non_empty_env(env_lookup, VERTEXAI_API_KEY_ENV));
        let project_id = get_vertex_ai_project(config, env_lookup);

        let static_token = match (has_authorization, static_token) {
            (true, _) => None,
            (false, Some(token)) => Some(token),
            (false, None) => {
                let provider = self.load_provider(config, env_lookup).await?;
                let (token, project_id) = match project_id {
                    Some(project_id) => (provider.token().await?, project_id),
                    None => tokio::try_join!(provider.token(), provider.project_id())?,
                };
                return Ok(VertexEnvironment {
                    headers: apply_credential(
                        headers,
                        token.expose(),
                        CredentialPlacement::Bearer,
                    )?,
                    project_id,
                });
            }
        };

        let project_id = match project_id {
            Some(project_id) => project_id,
            None => {
                self.load_provider(config, env_lookup)
                    .await?
                    .project_id()
                    .await?
            }
        };
        let headers = match static_token {
            Some(token) => apply_credential(headers, &token, CredentialPlacement::Bearer)?,
            None => headers,
        };
        Ok(VertexEnvironment {
            headers,
            project_id,
        })
    }

    async fn load_provider(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Arc<dyn GcpTokenSource>, Error> {
        self.identity
            .load_provider(credential_source(config, env_lookup))
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use crate::{
        CredentialSource, GcpAuthFuture, GcpProviderLoader, GcpTokenSource, VertexAuth,
        VertexConfig, get_vertex_ai_location, secret_names,
    };
    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use serde_json::{Value, json};

    struct FakeProvider {
        project: Option<&'static str>,
        calls: Arc<AtomicUsize>,
    }

    impl GcpTokenSource for FakeProvider {
        fn project_id(&self) -> GcpAuthFuture<'_, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                self.project.map(str::to_owned).ok_or(
                    litellm_auth_types::Error::CredentialAcquisition(
                        litellm_auth_types::ErrorDetail::Unresolved {
                            subject: "credential project",
                        },
                    ),
                )
            })
        }

        fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(SecretValue::new("adc-token")) })
        }
    }

    struct FakeLoader {
        loads: Arc<AtomicUsize>,
        provider: Arc<dyn GcpTokenSource>,
    }

    impl GcpProviderLoader for FakeLoader {
        fn load(&self, _source: CredentialSource) -> GcpAuthFuture<'_, Arc<dyn GcpTokenSource>> {
            let loads = self.loads.clone();
            let provider = self.provider.clone();
            Box::pin(async move {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(provider)
            })
        }
    }

    fn config(value: Value) -> VertexConfig {
        VertexConfig::from_sourced_optional_params(value.as_object().unwrap(), &BTreeMap::new())
            .unwrap()
    }

    fn auth(calls: Arc<AtomicUsize>, loads: Arc<AtomicUsize>) -> VertexAuth {
        let provider: Arc<dyn GcpTokenSource> = Arc::new(FakeProvider {
            calls,
            project: Some("adc-project"),
        });
        VertexAuth::new(Arc::new(FakeLoader { loads, provider }))
    }

    #[rstest]
    #[tokio::test]
    async fn secret_names_cover_environment_reads() {
        let seen = Arc::new(std::sync::Mutex::new(BTreeSet::<String>::new()));
        let recorded = seen.clone();
        let env = |name: &str| {
            recorded.lock().unwrap().insert(name.to_string());
            None
        };
        let auth = auth(Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        auth.validate_environment(Vec::new(), None, &VertexConfig::default(), &env)
            .await
            .unwrap();
        get_vertex_ai_location(&VertexConfig::default(), &env);
        let allowed = secret_names()
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<BTreeSet<_>>();
        assert!(seen.lock().unwrap().is_subset(&allowed));
    }

    #[rstest]
    #[tokio::test]
    async fn explicit_token_and_header_do_not_acquire_adc() {
        let loads = Arc::new(AtomicUsize::new(0));
        let auth = auth(Arc::new(AtomicUsize::new(0)), loads.clone());
        let configured = config(json!({"vertex_project":"project-1"}));
        let explicit = auth
            .validate_environment(Vec::new(), Some("access-token"), &configured, &|_| None)
            .await
            .unwrap();
        assert_eq!(explicit.headers[0].1, "Bearer access-token");
        let existing = auth
            .validate_environment(
                vec![("authorization".into(), "Bearer existing".into())],
                None,
                &configured,
                &|_| None,
            )
            .await
            .unwrap();
        assert_eq!(existing.headers[0].1, "Bearer existing");
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn provider_is_reused_across_authentication_calls() {
        let calls = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let auth = auth(calls.clone(), loads.clone());
        let config = VertexConfig::default();
        let environment = |_: &str| None;
        let (first, second) = tokio::try_join!(
            auth.validate_environment(Vec::new(), None, &config, &environment),
            auth.validate_environment(Vec::new(), None, &config, &environment),
        )
        .unwrap();
        assert_eq!(first.project_id, "adc-project");
        assert_eq!(first.headers[0].1, "Bearer adc-token");
        assert_eq!(first.project_id, second.project_id);
        assert_eq!(first.headers, second.headers);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    #[rstest]
    #[tokio::test]
    async fn an_explicit_project_does_not_require_project_metadata_in_the_credentials() {
        let auth = VertexAuth::new(Arc::new(FakeLoader {
            loads: Arc::new(AtomicUsize::new(0)),
            provider: Arc::new(FakeProvider {
                calls: Arc::new(AtomicUsize::new(0)),
                project: None,
            }),
        }));
        let configured = VertexConfig::new(None, Some("configured-project".into()), None);
        let environment = auth
            .validate_environment(Vec::new(), None, &configured, &|_| None)
            .await
            .unwrap();
        assert_eq!(environment.project_id, "configured-project");
        assert_eq!(environment.headers[0].1, "Bearer adc-token");
    }
}
