use std::time::SystemTime;

use crate::{
    facets::{
        AnonymousCore, AuthCore, AuthPlanner, AuthorizeCore, CustomAuthCore, JwtCore, KeyCore,
        ModelBudgetCore, OAuth2Core, TargetAuthCore,
    },
    host::{
        AuthorizeHost, CustomAuthHost, JwtHost, KeyHost, ModelBudgetHost, OAuth2Host,
        TeamMemberModelBudgetHost,
    },
    inputs::{
        AuthorizeSettings, JwtSettings, KeySettings, OAuth2Settings, PlanSettings, RequestFacts,
    },
    outcome::{
        AnonymousReason, CustomIdentity, Denial, Identity, ModelBudgetResult, OAuth2Denial,
        OAuth2Identity, Outcome, Validator,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCore {
    unported: Unported,
    oauth2: NativeOAuth2Core,
    custom: NativeCustomAuthCore,
}

impl AuthCore for NativeCore {
    type Keys = Unported;
    type ModelBudgets = Unported;

    fn keys(&self) -> &Unported {
        &self.unported
    }

    fn model_budgets(&self) -> &Unported {
        &self.unported
    }
}

impl TargetAuthCore for NativeCore {
    type Planner = Unported;
    type Anonymous = Unported;
    type Jwt = Unported;
    type OAuth2 = NativeOAuth2Core;
    type Custom = NativeCustomAuthCore;
    type Authorize = Unported;

    fn plan(&self) -> &Unported {
        &self.unported
    }

    fn anonymous(&self) -> &Unported {
        &self.unported
    }

    fn jwt(&self) -> &Unported {
        &self.unported
    }

    fn oauth2(&self) -> &NativeOAuth2Core {
        &self.oauth2
    }

    fn custom(&self) -> &NativeCustomAuthCore {
        &self.custom
    }

    fn authorize(&self) -> &Unported {
        &self.unported
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeOAuth2Core;

impl OAuth2Core for NativeOAuth2Core {
    async fn authenticate<H: OAuth2Host>(
        &self,
        host: &H,
        facts: &RequestFacts,
        _settings: &OAuth2Settings,
    ) -> Result<Result<OAuth2Identity, OAuth2Denial>, H::Error> {
        let Some(token) = &facts.token else {
            return Ok(Err(OAuth2Denial::MissingKey));
        };
        let introspection = host.introspect(token.expose()).await?;
        if !introspection.active {
            return Ok(Err(OAuth2Denial::InactiveToken));
        }
        Ok(Ok(OAuth2Identity {
            user_id: introspection.subject,
            team_id: None,
            scopes: introspection.scopes,
        }))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCustomAuthCore;

impl CustomAuthCore for NativeCustomAuthCore {
    async fn authenticate<H: CustomAuthHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
    ) -> Result<CustomIdentity, H::Error> {
        host.custom_auth(facts).await
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Unported;

impl KeyCore for Unported {
    async fn authenticate<H: KeyHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
        _settings: &KeySettings,
        _now: SystemTime,
    ) -> Result<Outcome, H::Error> {
        unimplemented!("the key facet runs in PythonCore")
    }

    async fn check<H: KeyHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
        _settings: &KeySettings,
        _now: SystemTime,
    ) -> Result<Outcome, H::Error> {
        unimplemented!("the key facet runs in PythonCore")
    }
}

impl ModelBudgetCore for Unported {
    async fn check_key_and_user<H: ModelBudgetHost>(
        &self,
        _host: &H,
    ) -> Result<ModelBudgetResult, H::Error> {
        unimplemented!("the model budget facet runs in PythonCore")
    }

    async fn check_user<H: ModelBudgetHost>(
        &self,
        _host: &H,
    ) -> Result<ModelBudgetResult, H::Error> {
        unimplemented!("the model budget facet runs in PythonCore")
    }

    async fn check_team_member<H: TeamMemberModelBudgetHost>(
        &self,
        _host: &H,
    ) -> Result<ModelBudgetResult, H::Error> {
        unimplemented!("the model budget facet runs in PythonCore")
    }
}

impl AuthPlanner for Unported {
    fn plan(&self, _facts: &RequestFacts, _settings: &PlanSettings) -> Validator {
        unimplemented!("the proxy picks the credential path")
    }
}

impl AnonymousCore for Unported {
    fn authenticate(&self, _facts: &RequestFacts, _settings: &PlanSettings) -> AnonymousReason {
        unimplemented!("the proxy decides anonymous requests")
    }
}

impl JwtCore for Unported {
    async fn authenticate<H: JwtHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
        _settings: &JwtSettings,
        _now: SystemTime,
    ) -> Result<Result<Identity, Denial>, H::Error> {
        unimplemented!("the proxy validates JWTs")
    }
}

impl AuthorizeCore for Unported {
    async fn authorize<H: AuthorizeHost>(
        &self,
        _host: &H,
        _identity: &Identity,
        _facts: &RequestFacts,
        _settings: &AuthorizeSettings,
    ) -> Result<Outcome, H::Error> {
        unimplemented!("the proxy runs route auth and the common checks")
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, convert::Infallible, sync::Mutex};

    use litellm_secret_value::SecretValue;
    use rstest::{fixture, rstest};

    use super::*;
    use crate::answers::Introspection;

    struct ScriptedOAuth2Host {
        introspection: Introspection,
        headers: BTreeMap<String, String>,
        introspected: Mutex<Vec<String>>,
    }

    impl OAuth2Host for ScriptedOAuth2Host {
        type Error = Infallible;

        async fn introspect(&self, token: &str) -> Result<Introspection, Infallible> {
            self.introspected
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(token.to_owned());
            Ok(self.introspection.clone())
        }

        fn proxy_headers(&self) -> &BTreeMap<String, String> {
            &self.headers
        }
    }

    #[fixture]
    fn settings() -> OAuth2Settings {
        OAuth2Settings {
            user_id_header: None,
            team_id_header: None,
        }
    }

    fn host(active: bool) -> ScriptedOAuth2Host {
        ScriptedOAuth2Host {
            introspection: Introspection {
                active,
                subject: Some("user-1".to_owned()),
                client_id: None,
                scopes: vec!["read".to_owned()],
            },
            headers: BTreeMap::new(),
            introspected: Mutex::new(Vec::new()),
        }
    }

    fn facts(token: Option<&str>) -> RequestFacts {
        RequestFacts {
            route: "/chat/completions".to_owned(),
            token: token.map(SecretValue::from),
            is_llm_api_route: true,
        }
    }

    #[rstest]
    #[case::active(
        true,
        Ok(OAuth2Identity {
            user_id: Some("user-1".to_owned()),
            team_id: None,
            scopes: vec!["read".to_owned()],
        })
    )]
    #[case::inactive(false, Err(OAuth2Denial::InactiveToken))]
    #[tokio::test]
    async fn oauth2_introspects_the_token_and_trusts_only_an_active_answer(
        settings: OAuth2Settings,
        #[case] active: bool,
        #[case] expected: Result<OAuth2Identity, OAuth2Denial>,
    ) {
        let host = host(active);
        let outcome = NativeCore::default()
            .oauth2()
            .authenticate(&host, &facts(Some("tok-1")), &settings)
            .await;
        assert_eq!(outcome, Ok(expected));
        assert_eq!(
            *host
                .introspected
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec!["tok-1".to_owned()]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn oauth2_without_a_token_is_denied_before_any_introspection(settings: OAuth2Settings) {
        let host = host(true);
        let outcome = NativeCore::default()
            .oauth2()
            .authenticate(&host, &facts(None), &settings)
            .await;
        assert_eq!(outcome, Ok(Err(OAuth2Denial::MissingKey)));
        assert!(
            host.introspected
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
        );
    }

    struct ScriptedCustomAuthHost(CustomIdentity);

    impl CustomAuthHost for ScriptedCustomAuthHost {
        type Error = Infallible;

        async fn custom_auth(&self, _facts: &RequestFacts) -> Result<CustomIdentity, Infallible> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn custom_auth_returns_the_identity_the_host_resolved() {
        let identity = CustomIdentity {
            user_id: Some("user-1".to_owned()),
            team_id: Some("team-1".to_owned()),
            hashed_token: None,
        };
        let outcome = NativeCore::default()
            .custom()
            .authenticate(
                &ScriptedCustomAuthHost(identity.clone()),
                &facts(Some("tok-1")),
            )
            .await;
        assert_eq!(outcome, Ok(identity));
    }
}
