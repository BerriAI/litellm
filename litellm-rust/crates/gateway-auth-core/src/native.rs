use std::{net::IpAddr, time::SystemTime};

use ipnet::IpNet;

use crate::{
    facets::{
        AnonymousCore, AuthCore, AuthPlanner, AuthorizeCore, CustomAuthCore, JwtCore, KeyCore,
        ModelBudgetCore, OAuth2Core,
    },
    host::{
        AuthorizeHost, CustomAuthHost, JwtHost, KeyHost, ModelBudgetHost, OAuth2IntrospectionHost,
        OAuth2ProxyHost, TeamMemberModelBudgetHost,
    },
    inputs::{
        AuthorizeSettings, JwtSettings, KeySettings, OAuth2IntrospectionSettings,
        OAuth2ProxySettings, PlanFacts, PlanSettings, RequestFacts,
    },
    outcome::{
        Anonymous, AnonymousReason, CredentialPlan, CustomAuthResult, Denial, Identity, KeyPlan,
        ModelBudgetResult, OAuth2Denial, OAuth2Identity, OAuth2ProxyDenial, OAuth2ProxyIdentity,
        Outcome, Plan,
    },
};

const ALLOWED_OAUTH2_PROXY_FIELDS: [&str; 6] = [
    "user_id",
    "user_email",
    "team_id",
    "team_alias",
    "org_id",
    "models",
];

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCore {
    unported: Unported,
    oauth2: NativeOAuth2Core,
    custom: NativeCustomAuthCore,
}

impl AuthCore for NativeCore {
    type Keys = Unported;
    type ModelBudgets = Unported;
    type Planner = Unported;
    type Anonymous = Unported;
    type Jwt = Unported;
    type OAuth2 = NativeOAuth2Core;
    type Custom = NativeCustomAuthCore;
    type Authorize = Unported;

    fn keys(&self) -> &Unported {
        &self.unported
    }

    fn model_budgets(&self) -> &Unported {
        &self.unported
    }

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

fn ip_in_networks(client_ip: Option<&str>, networks: &[IpNet]) -> bool {
    let Some(Ok(address)) = client_ip.map(|ip| ip.trim().parse::<IpAddr>()) else {
        return false;
    };
    let unmapped = match address {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(address, IpAddr::V4),
        IpAddr::V4(_) => address,
    };
    networks
        .iter()
        .any(|network| network.contains(&address) || network.contains(&unmapped))
}

fn proxy_identity<H: OAuth2ProxyHost>(
    host: &H,
    settings: &OAuth2ProxySettings,
) -> OAuth2ProxyIdentity {
    let value = |field: &str| {
        settings
            .header_mappings
            .get(field)
            .and_then(|header| host.header(header))
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    OAuth2ProxyIdentity {
        user_id: value("user_id"),
        user_email: value("user_email"),
        team_id: value("team_id"),
        team_alias: value("team_alias"),
        org_id: value("org_id"),
        models: value("models").map(|models| {
            models
                .split(',')
                .map(|model| model.trim().to_owned())
                .collect()
        }),
    }
}

impl OAuth2Core for NativeOAuth2Core {
    async fn introspect<H: OAuth2IntrospectionHost>(
        &self,
        host: &H,
        token: &str,
        settings: &OAuth2IntrospectionSettings,
    ) -> Result<Result<OAuth2Identity, OAuth2Denial>, H::Error> {
        if !settings.endpoint_configured {
            return Ok(Err(OAuth2Denial::IntrospectionNotConfigured));
        }
        let answer = host.introspect(token).await?;
        if settings.requires_active && !answer.active {
            return Ok(Err(OAuth2Denial::InactiveToken));
        }
        Ok(Ok(OAuth2Identity {
            token: token.to_owned(),
            user_id: answer.user_id,
            role: answer.role,
            team_id: answer.team_id,
        }))
    }

    fn proxy<H: OAuth2ProxyHost>(
        &self,
        host: &H,
        settings: &OAuth2ProxySettings,
    ) -> Result<OAuth2ProxyIdentity, OAuth2ProxyDenial> {
        if settings.trusted_networks.is_empty() {
            return Err(OAuth2ProxyDenial::TrustedProxyRangesMissing);
        }
        let client_ip = host.client_ip();
        if !ip_in_networks(client_ip, &settings.trusted_networks) {
            return Err(OAuth2ProxyDenial::UntrustedProxy {
                client_ip: client_ip.map(str::to_owned),
            });
        }
        if settings.header_mappings.is_empty() {
            return Err(OAuth2ProxyDenial::ProxyMappingsMissing);
        }
        let disallowed: Vec<String> = settings
            .header_mappings
            .keys()
            .filter(|field| !ALLOWED_OAUTH2_PROXY_FIELDS.contains(&field.as_str()))
            .cloned()
            .collect();
        if !disallowed.is_empty() {
            return Err(OAuth2ProxyDenial::DisallowedProxyMappings { fields: disallowed });
        }
        Ok(proxy_identity(host, settings))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCustomAuthCore;

impl CustomAuthCore for NativeCustomAuthCore {
    async fn authenticate<H: CustomAuthHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
    ) -> Result<CustomAuthResult, H::Error> {
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
    fn plan(&self, _facts: &PlanFacts, _settings: &PlanSettings) -> Plan {
        unimplemented!("the planner runs in PythonCore")
    }

    fn after_custom_auth(
        &self,
        _facts: &PlanFacts,
        _settings: &PlanSettings,
        _issued_key: bool,
    ) -> CredentialPlan {
        unimplemented!("the planner runs in PythonCore")
    }

    fn after_jwt(
        &self,
        _facts: &PlanFacts,
        _settings: &PlanSettings,
        _issued_key: bool,
    ) -> KeyPlan {
        unimplemented!("the planner runs in PythonCore")
    }

    fn oauth2_introspection_applies(&self, _facts: &PlanFacts, _settings: &PlanSettings) -> bool {
        unimplemented!("the planner runs in PythonCore")
    }
}

impl AnonymousCore for Unported {
    fn authenticate(&self, _reason: AnonymousReason) -> Anonymous {
        unimplemented!("the anonymous facet runs in PythonCore")
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
    use std::{
        collections::BTreeMap,
        convert::Infallible,
        sync::{Mutex, PoisonError},
    };

    use litellm_secret_value::SecretValue;
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::answers::Introspection;

    struct ScriptedIntrospection {
        answer: Introspection,
        introspected: Mutex<Vec<String>>,
    }

    impl OAuth2IntrospectionHost for ScriptedIntrospection {
        type Error = Infallible;

        async fn introspect(&self, token: &str) -> Result<Introspection, Infallible> {
            self.introspected
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(token.to_owned());
            Ok(self.answer.clone())
        }
    }

    fn introspection(active: bool) -> ScriptedIntrospection {
        ScriptedIntrospection {
            answer: Introspection {
                active,
                user_id: json!("user-1"),
                role: json!("internal_user"),
                team_id: json!(7),
            },
            introspected: Mutex::new(Vec::new()),
        }
    }

    fn identity() -> OAuth2Identity {
        OAuth2Identity {
            token: "tok".to_owned(),
            user_id: json!("user-1"),
            role: json!("internal_user"),
            team_id: json!(7),
        }
    }

    #[rstest]
    #[case::active(true, true, Ok(identity()))]
    #[case::inactive(false, true, Err(OAuth2Denial::InactiveToken))]
    #[case::token_info_endpoint(false, false, Ok(identity()))]
    #[tokio::test]
    async fn introspection_trusts_an_inactive_answer_only_from_an_introspection_endpoint(
        #[case] active: bool,
        #[case] requires_active: bool,
        #[case] expected: Result<OAuth2Identity, OAuth2Denial>,
    ) {
        let host = introspection(active);
        let settings = OAuth2IntrospectionSettings {
            endpoint_configured: true,
            requires_active,
        };
        let outcome = NativeCore::default()
            .oauth2()
            .introspect(&host, "tok", &settings)
            .await;
        assert_eq!(outcome, Ok(expected));
        assert_eq!(
            *host
                .introspected
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
            vec!["tok".to_owned()]
        );
    }

    #[tokio::test]
    async fn introspection_without_an_endpoint_never_calls_out() {
        let host = introspection(true);
        let settings = OAuth2IntrospectionSettings {
            endpoint_configured: false,
            requires_active: true,
        };
        let outcome = NativeCore::default()
            .oauth2()
            .introspect(&host, "tok", &settings)
            .await;
        assert_eq!(outcome, Ok(Err(OAuth2Denial::IntrospectionNotConfigured)));
        assert!(
            host.introspected
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );
    }

    struct Headers {
        client_ip: Option<&'static str>,
        headers: BTreeMap<&'static str, &'static str>,
        read: Mutex<Vec<String>>,
    }

    impl OAuth2ProxyHost for Headers {
        fn client_ip(&self) -> Option<&str> {
            self.client_ip
        }

        fn header(&self, name: &str) -> Option<&str> {
            self.read
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(name.to_owned());
            self.headers.get(name).copied()
        }
    }

    fn headers(
        client_ip: Option<&'static str>,
        headers: &[(&'static str, &'static str)],
    ) -> Headers {
        Headers {
            client_ip,
            headers: headers.iter().copied().collect(),
            read: Mutex::new(Vec::new()),
        }
    }

    #[fixture]
    fn trusted() -> Vec<IpNet> {
        vec!["10.0.0.0/8".parse().unwrap()]
    }

    fn settings(networks: Vec<IpNet>, mappings: &[(&str, &str)]) -> OAuth2ProxySettings {
        OAuth2ProxySettings {
            trusted_networks: networks,
            header_mappings: mappings
                .iter()
                .map(|(field, header)| ((*field).to_owned(), (*header).to_owned()))
                .collect(),
        }
    }

    #[rstest]
    fn proxy_headers_assert_only_the_mapped_identity_fields_that_are_present(trusted: Vec<IpNet>) {
        let host = headers(
            Some("10.1.2.3"),
            &[
                ("x-user", "user-1"),
                ("x-team", ""),
                ("x-models", "gpt-a, gpt-b"),
            ],
        );
        let settings = settings(
            trusted,
            &[
                ("user_id", "x-user"),
                ("team_id", "x-team"),
                ("models", "x-models"),
            ],
        );
        assert_eq!(
            NativeCore::default().oauth2().proxy(&host, &settings),
            Ok(OAuth2ProxyIdentity {
                user_id: Some("user-1".to_owned()),
                models: Some(vec!["gpt-a".to_owned(), "gpt-b".to_owned()]),
                ..OAuth2ProxyIdentity::default()
            })
        );
    }

    #[rstest]
    #[case::ipv4("10.1.2.3", true)]
    #[case::ipv4_mapped("::ffff:10.1.2.3", true)]
    #[case::outside("192.168.1.1", false)]
    #[case::not_an_ip("not-an-ip", false)]
    fn proxy_headers_trust_only_a_peer_inside_the_ranges(
        trusted: Vec<IpNet>,
        #[case] client_ip: &'static str,
        #[case] is_trusted: bool,
    ) {
        let host = headers(Some(client_ip), &[("x-user", "user-1")]);
        let outcome = NativeCore::default()
            .oauth2()
            .proxy(&host, &settings(trusted, &[("user_id", "x-user")]));
        assert_eq!(outcome.is_ok(), is_trusted);
        assert_eq!(
            host.read
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty(),
            !is_trusted
        );
    }

    #[rstest]
    fn proxy_headers_refuse_every_non_identity_mapping_in_sorted_order(trusted: Vec<IpNet>) {
        let host = headers(Some("10.1.2.3"), &[]);
        let settings = settings(
            trusted,
            &[
                ("user_role", "x-role"),
                ("user_id", "x-user"),
                ("max_budget", "x-budget"),
            ],
        );
        assert_eq!(
            NativeCore::default().oauth2().proxy(&host, &settings),
            Err(OAuth2ProxyDenial::DisallowedProxyMappings {
                fields: vec!["max_budget".to_owned(), "user_role".to_owned()],
            })
        );
    }

    #[rstest]
    #[case::no_ranges(Vec::new(), &[("user_id", "x-user")], OAuth2ProxyDenial::TrustedProxyRangesMissing)]
    #[case::no_mappings(trusted(), &[], OAuth2ProxyDenial::ProxyMappingsMissing)]
    fn proxy_headers_need_ranges_and_mappings(
        #[case] networks: Vec<IpNet>,
        #[case] mappings: &[(&str, &str)],
        #[case] expected: OAuth2ProxyDenial,
    ) {
        let host = headers(Some("10.1.2.3"), &[]);
        assert_eq!(
            NativeCore::default()
                .oauth2()
                .proxy(&host, &settings(networks, mappings)),
            Err(expected)
        );
    }

    struct ScriptedCustomAuth(CustomAuthResult);

    impl CustomAuthHost for ScriptedCustomAuth {
        type Error = Infallible;

        async fn custom_auth(&self, _facts: &RequestFacts) -> Result<CustomAuthResult, Infallible> {
            Ok(self.0.clone())
        }
    }

    #[rstest]
    #[case::identity(CustomAuthResult::Identity(crate::outcome::CustomIdentity {
        user_id: Some("user-1".to_owned()),
        team_id: Some("team-1".to_owned()),
        hashed_token: None,
    }))]
    #[case::replaced_key(CustomAuthResult::ReplacedKey { token: "sk-new".to_owned() })]
    #[case::no_identity(CustomAuthResult::NoIdentity)]
    #[tokio::test]
    async fn custom_auth_returns_what_the_custom_function_resolved(
        #[case] answer: CustomAuthResult,
    ) {
        let facts = RequestFacts {
            route: "/chat/completions".to_owned(),
            token: Some(SecretValue::from("tok-1")),
            is_llm_api_route: true,
        };
        let outcome = NativeCore::default()
            .custom()
            .authenticate(&ScriptedCustomAuth(answer.clone()), &facts)
            .await;
        assert_eq!(outcome, Ok(answer));
    }
}
