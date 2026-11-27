use std::{future::Future, pin::Pin, sync::Arc};

use axum::http::request::Parts;
use jiff::Timestamp;
use litellm_gateway_auth_core::{
    self as auth_core, AuthHost, Caller, Model, Outcome, Rows, Settings, Token, authenticate_jwt,
    authenticate_key, plan::Plan,
};
use litellm_secret_value::SecretValue;

use crate::{Authenticated, Bearer, CredentialExtractor, Error};

const TEAM_HEADER: &str = "x-litellm-team-id";

pub type AuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait Clock: Send + Sync {
    fn now(&self) -> Timestamp;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        Timestamp::now()
    }
}

pub enum Credential {
    Token(SecretValue),
    Transport,
}

/// Who a selected credential stands for. The only answer is `gateway-auth-core`'s [`Caller`], so
/// nothing this crate hands a route was decided here.
pub trait Authenticator<R: Rows>: Send + Sync {
    /// Whether authenticating `credential` decides on the model the request's body names, so a
    /// server that can read the body first should. Decided with no IO.
    fn reads_model(&self, credential: &Credential, request: &Parts) -> bool;

    /// `model` is what the server read of the body, [`Model::Unread`] when it read none.
    fn authenticate<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
        model: Model<'a>,
    ) -> AuthFuture<'a, Caller<R>>;
}

pub struct Auth<R: Rows> {
    extractor: Arc<dyn CredentialExtractor>,
    authenticator: Arc<dyn Authenticator<R>>,
}

impl<R: Rows> Clone for Auth<R> {
    fn clone(&self) -> Self {
        Self {
            extractor: self.extractor.clone(),
            authenticator: self.authenticator.clone(),
        }
    }
}

impl<R: Rows> Auth<R> {
    pub fn new(authenticator: Arc<dyn Authenticator<R>>) -> Self {
        Self {
            extractor: Arc::new(Bearer),
            authenticator,
        }
    }

    pub fn with_extractor(self, extractor: Arc<dyn CredentialExtractor>) -> Self {
        Self { extractor, ..self }
    }

    /// Whether authenticating `request` decides on the model its body names. A request with no
    /// usable credential reads none.
    pub fn reads_model(&self, request: &Parts) -> bool {
        self.extractor
            .extract(request)
            .is_ok_and(|credential| self.authenticator.reads_model(&credential, request))
    }

    pub async fn authenticate_parts(
        &self,
        request: &Parts,
        model: Model<'_>,
    ) -> Result<Authenticated<R>, Error> {
        let credential = self.extractor.extract(request)?;
        self.authenticate_credential(&credential, request, model)
            .await
            .map_err(|error| self.extractor.reject(request, error))
    }

    pub(crate) async fn authenticate_credential(
        &self,
        credential: &Credential,
        request: &Parts,
        model: Model<'_>,
    ) -> Result<Authenticated<R>, Error> {
        let caller = self
            .authenticator
            .authenticate(credential, request, model)
            .await?;
        Ok(Authenticated::from(caller))
    }
}

/// Hands a credential to `gateway-auth-core` over the host `H`, letting `auth_core::plan` pick
/// the key path or the JWT path and declining plans this server does not serve natively. Every
/// route this crate guards is an inference route, which is what the request it hands the core
/// says.
pub struct CoreAuthenticator<H> {
    host: H,
    settings: Settings,
    clock: Arc<dyn Clock>,
}

impl<H> CoreAuthenticator<H> {
    pub fn new(host: H, settings: Settings, clock: Arc<dyn Clock>) -> Self {
        Self {
            host,
            settings,
            clock,
        }
    }
}

fn team_header(parts: &Parts) -> Option<&str> {
    parts
        .headers
        .get(TEAM_HEADER)
        .and_then(|value| value.to_str().ok())
}

fn token_of(credential: &Credential) -> Option<Token> {
    match credential {
        Credential::Token(presented) => Some(Token::new(presented.clone())),
        Credential::Transport => None,
    }
}

impl<H> CoreAuthenticator<H> {
    fn core_request<'a>(
        &self,
        token: Option<&'a Token>,
        request: &'a Parts,
        model: Model<'a>,
    ) -> auth_core::Request<'a> {
        auth_core::Request {
            route: request.uri.path(),
            method: Some(&request.method),
            token,
            model,
            team_header: team_header(request),
            is_llm_api_route: true,
            runs_inference_checks: true,
            now: self.clock.now(),
            ..auth_core::Request::default()
        }
    }
}

impl<H: AuthHost> Authenticator<H::Rows> for CoreAuthenticator<H> {
    /// Only the JWT path reads the model: to pick a team, and for the role and scope rules.
    fn reads_model(&self, credential: &Credential, request: &Parts) -> bool {
        let token = token_of(credential);
        let core_request = self.core_request(token.as_ref(), request, Model::Unread);
        match auth_core::plan(&core_request, &self.settings) {
            Ok(Plan::Jwt) => true,
            Ok(
                Plan::Key
                | Plan::Public
                | Plan::OAuth2Introspection
                | Plan::OAuth2Proxy
                | Plan::JwtWithOperatorCode
                | Plan::PassThroughOpen,
            )
            | Err(_) => false,
        }
    }

    fn authenticate<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
        model: Model<'a>,
    ) -> AuthFuture<'a, Caller<H::Rows>> {
        Box::pin(async move {
            let token = token_of(credential);
            let core_request = self.core_request(token.as_ref(), request, model);
            let plan = auth_core::plan(&core_request, &self.settings)
                .map_err(|premium| Error::from(auth_core::Error::from(premium)))?;
            let outcome = match plan {
                Plan::Key => {
                    authenticate_key(&self.host, &core_request, token.as_ref(), &self.settings)
                        .await
                }
                Plan::Jwt => authenticate_jwt(&self.host, &core_request, &self.settings).await,
                Plan::Public
                | Plan::OAuth2Introspection
                | Plan::OAuth2Proxy
                | Plan::JwtWithOperatorCode
                | Plan::PassThroughOpen => return Err(Error::NotNative),
            };
            match outcome {
                Ok(Outcome::Authenticated(identity)) => Ok(Caller::try_from(identity)?),
                Ok(Outcome::Denied(error, _)) | Err(error) => Err(error.into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, sync::OnceLock, time::Duration};

    use axum::http::Request as HttpRequest;
    use litellm_config::{GeneralSettings, JwtAuthSettings, LitellmRole, RouteList, ScopeMapping};
    use litellm_gateway_auth_core::{
        AgentIdentityFailure, HostFacts, KeyHash, MasterKey, OperatorHooks, RouteGroups,
        authn::{
            jwt::{KeyMappingRow, KeyMappingWrite, MappedClaim, MappedKey},
            key::{KeyRecord, Keyed},
        },
        body::Params,
        identity::{
            jwt::{
                AliasRead, DelegatedTeam, IdentityRead, ManagedAgent, ModelAccess, Org,
                RegisteredAgent, TeamRead, UserLookup, UserRead,
            },
            key::Owner,
        },
    };
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct StubRows;

    impl Rows for StubRows {
        type Key = KeyRecord;
        type User = ();
        type Team = ();
        type EndUser = ();
        type ObjectPermission = Infallible;
        type Project = ();
        type Membership = ();
        type Org = ();
        type Agent = Infallible;
    }

    /// Declines every step a host needs a store for, and misses every cache.
    #[derive(Default)]
    struct DecliningHost {
        /// The JWKS the identity provider serves, which this host fetches when it holds one.
        jwks: Option<Value>,
        /// The team id the core first asked `team` for.
        team_asked: OnceLock<String>,
    }

    struct NoHooks;

    impl OperatorHooks for NoHooks {
        async fn custom_validate(&self, _: &Value) -> Result<bool, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
    }

    impl AuthHost for DecliningHost {
        type Rows = StubRows;

        async fn cached_key(
            &self,
            _: &KeyHash,
        ) -> Result<Option<Keyed<KeyRecord>>, auth_core::Error> {
            Ok(None)
        }
        async fn db_key(
            &self,
            _: &KeyHash,
            _: &(),
        ) -> Result<Option<Keyed<KeyRecord>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn key_by_hash(
            &self,
            _: &KeyHash,
        ) -> Result<Option<MappedKey<KeyRecord>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn key_mapping_hash(
            &self,
            _: &str,
            _: &str,
            _: Option<&str>,
        ) -> Result<Option<KeyHash>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn reusable_key_hash(
            &self,
            _: &str,
            _: Option<&str>,
        ) -> Result<Option<KeyHash>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn ui_key(&self, _: &Token) -> Result<Option<Keyed<KeyRecord>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn evict_key(&self, _: &KeyHash) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn fetch_json(&self, _: &str) -> Result<Option<Value>, auth_core::Error> {
            self.jwks
                .clone()
                .map(Some)
                .ok_or(auth_core::Error::HostDeclined)
        }
        async fn fetch_userinfo(
            &self,
            _: &str,
            _: &Token,
        ) -> Result<Option<Value>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn cached_document(&self, _: &str) -> Result<Option<Value>, auth_core::Error> {
            Ok(None)
        }
        async fn cache_document(
            &self,
            _: &str,
            _: &Value,
            _: Duration,
        ) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn cached_key_mapping(
            &self,
            _: MappedClaim<'_>,
        ) -> Result<Option<Value>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn cache_key_mapping(
            &self,
            _: MappedClaim<'_>,
            _: &Value,
            _: Duration,
        ) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn forget_key_mapping(&self, _: MappedClaim<'_>) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        fn operator_hooks(&self) -> Option<&impl OperatorHooks> {
            None::<&NoHooks>
        }
        async fn managed_agent(
            &self,
            _: &Value,
        ) -> Result<Option<ManagedAgent<Infallible>>, auth_core::Error> {
            Ok(None)
        }
        async fn registered_agent(
            &self,
            _: &str,
        ) -> Result<Option<RegisteredAgent>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn delegated_agent_team(
            &self,
            _: &DelegatedTeam<'_>,
        ) -> Result<Option<String>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn key_owner(&self, _: &str) -> Result<Option<Owner<()>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn user(&self, _: &UserLookup<'_>) -> Result<UserRead<()>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn resolve_user_identity(
            &self,
            _: &UserLookup<'_>,
            _: Option<&str>,
        ) -> Result<IdentityRead<StubRows>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn cached_team(
            &self,
            _: &str,
        ) -> Result<Option<auth_core::Refreshed<()>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn key_team(
            &self,
            _: &str,
        ) -> Result<Option<auth_core::identity::key::KeyTeam<()>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn team(&self, team_id: &str, _: bool) -> Result<TeamRead<()>, auth_core::Error> {
            let _ = self.team_asked.set(team_id.to_owned());
            Err(auth_core::Error::HostDeclined)
        }
        async fn granting_team(&self, _: &str) -> Result<TeamRead<()>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn team_by_alias(&self, _: &str) -> Result<AliasRead<()>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn load_team_membership(
            &self,
            _: &str,
            _: &str,
            _: auth_core::AffiliationRead,
        ) -> Result<Option<()>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn org(
            &self,
            _: &str,
            _: auth_core::AffiliationRead,
        ) -> Result<Option<Org<()>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn org_by_alias(&self, _: &str) -> Result<Option<Org<()>>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn jwt_end_user(&self, _: &str) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn end_user(&self, _: Option<&str>) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn validated_end_user_id(&self, _: &str) -> Result<Option<String>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn object_permission(&self, _: &str) -> Result<Option<Infallible>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn project(&self, _: &str) -> Result<Option<()>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn prefetch_identity(
            &self,
            _: Option<&Token>,
            _: Option<&str>,
        ) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn prefetch_auth_objects(
            &self,
            _: &KeyRecord,
            _: &(),
        ) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn set_user_role(&self, _: &(), _: LitellmRole) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn patch_team_membership(
            &self,
            _: &(),
            _: &[String],
            _: &[String],
        ) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn normalize_model(&self, _: &KeyRecord) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn with_temp_budget(
            &self,
            key: Keyed<KeyRecord>,
        ) -> Result<Keyed<KeyRecord>, auth_core::Error> {
            Ok(key)
        }
        async fn key_skips_budget_checks(&self, _: &KeyRecord) -> Result<bool, auth_core::Error> {
            Ok(false)
        }
        async fn team_model_access(
            &self,
            _: &(),
            _: &str,
        ) -> Result<ModelAccess, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn team_member_budget(
            &self,
            _: &KeyRecord,
            _: Option<&()>,
        ) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn key_budgets(
            &self,
            _: &KeyRecord,
            _: Option<&()>,
            _: bool,
        ) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn jwt_user_model_budget(
            &self,
            _: &(),
            _: &str,
            _: Option<&str>,
        ) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn proxy_budget_alert(&self, _: &KeyRecord) -> Result<(), auth_core::Error> {
            Ok(())
        }
        async fn record_agent_authentication(
            &self,
            agent: &Infallible,
        ) -> Result<Option<AgentIdentityFailure>, auth_core::Error> {
            match *agent {}
        }
        async fn add_team_member(&self, _: &str, _: &str) -> Result<(), auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn create_key_mapping(
            &self,
            _: &KeyMappingRow<'_>,
        ) -> Result<KeyMappingWrite, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn with_team(&self, _: &KeyRecord, _: &()) -> Result<KeyRecord, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
        async fn credential(&self, _: &str) -> Result<Option<Params>, auth_core::Error> {
            Err(auth_core::Error::HostDeclined)
        }
    }

    const JWKS_URL: &str = "https://idp.example/jwks.json";
    const PRIVATE_PEM: &str = "-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIEjkuYWkj6THWjxVRFL36QT1XS2a1KjdV89JMvLev7r2
-----END PRIVATE KEY-----
";
    const PUBLIC_X: &str = "iO5qGwVLMzVNKsLll1PGbMPQzI_aTO9tZuoe8AEWCo4";

    /// The JWKS that publishes the key [`admin_jwt`] is signed with.
    fn jwks() -> Value {
        json!({"keys": [{"kty": "OKP", "crv": "Ed25519", "x": PUBLIC_X}]})
    }

    /// A proxy admin's JWT, valid for the next hour.
    fn admin_jwt() -> String {
        let now = Timestamp::now().as_second();
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::EdDSA),
            &json!({"scope": "litellm_proxy_admin", "iat": now, "exp": now + 3600}),
            &jsonwebtoken::EncodingKey::from_ed_pem(PRIVATE_PEM.as_bytes()).unwrap(),
        )
        .unwrap()
    }

    /// The core's settings on a host with no database, whose environment names [`JWKS_URL`].
    fn settings(general: GeneralSettings, premium: bool) -> Settings {
        Settings::new(
            &general,
            HostFacts {
                master_key: Some(MasterKey::new("sk-master")),
                premium,
                has_db: false,
                db_configured: false,
                ui_login: false,
                proxy_budget: false,
                metrics_require_auth: true,
                route_groups: RouteGroups::default(),
            },
            &|name| (name == "JWT_PUBLIC_KEY_URL").then(|| JWKS_URL.to_owned()),
        )
    }

    /// JWT auth `enabled` or not, where a proxy admin's JWT may call `/v1/messages`.
    fn jwt_auth(enabled: bool) -> GeneralSettings {
        GeneralSettings {
            enable_jwt_auth: enabled,
            litellm_jwtauth: Some(JwtAuthSettings {
                admin_allowed_routes: ["/v1/messages".to_owned()].into(),
                ..JwtAuthSettings::default()
            }),
            ..GeneralSettings::default()
        }
    }

    fn public_messages_route() -> GeneralSettings {
        GeneralSettings {
            public_routes: Some(RouteList::Valid(["/v1/messages".to_owned()].into())),
            ..GeneralSettings::default()
        }
    }

    fn authenticator(
        host: DecliningHost,
        general: GeneralSettings,
        premium: bool,
    ) -> CoreAuthenticator<DecliningHost> {
        CoreAuthenticator::new(host, settings(general, premium), Arc::new(SystemClock))
    }

    #[rstest]
    #[case::a_jwt_without_an_enterprise_license_is_refused(jwt_auth(true), false, "forbidden")]
    #[case::a_jwt_takes_the_jwt_path_to_the_jwks_this_host_declines(
        jwt_auth(true),
        true,
        "not_native"
    )]
    #[case::a_jwt_with_jwt_auth_off_takes_the_key_path(jwt_auth(false), true, "invalid_token")]
    #[case::a_public_route_is_not_served_natively(public_messages_route(), true, "not_native")]
    #[tokio::test]
    async fn a_request_is_answered_by_the_path_the_plan_picks(
        #[case] general: GeneralSettings,
        #[case] premium: bool,
        #[case] answered: &str,
    ) {
        let auth = authenticator(DecliningHost::default(), general, premium);
        let token = Credential::Token(SecretValue::from(admin_jwt()));
        let request = HttpRequest::post("/v1/messages")
            .body(())
            .unwrap()
            .into_parts()
            .0;

        let error = auth
            .authenticate(&token, &request, Model::Unread)
            .await
            .unwrap_err();

        assert_eq!(<&'static str>::from(error), answered);
    }

    /// [`jwt_auth`] with the admin scope mapped to the model `claude` and scope mappings enforced.
    fn scoped_to_claude() -> GeneralSettings {
        let general = jwt_auth(true);
        GeneralSettings {
            litellm_jwtauth: general.litellm_jwtauth.map(|jwt| JwtAuthSettings {
                enforce_scope_based_access: true,
                scope_mappings: Some(
                    [ScopeMapping {
                        scope: "litellm_proxy_admin".to_owned(),
                        models: Some(["claude".to_owned()].into()),
                        routes: None,
                    }]
                    .into(),
                ),
                ..jwt
            }),
            ..general
        }
    }

    fn messages_request() -> Parts {
        HttpRequest::post("/v1/messages")
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    #[rstest]
    #[case::a_jwt(jwt_auth(true), true, Some(admin_jwt()), true)]
    #[case::a_jwt_with_jwt_auth_off(jwt_auth(false), true, Some(admin_jwt()), false)]
    #[case::a_jwt_without_an_enterprise_license(jwt_auth(true), false, Some(admin_jwt()), false)]
    #[case::a_jwt_on_a_public_route(public_messages_route(), true, Some(admin_jwt()), false)]
    #[case::a_key(jwt_auth(true), true, Some("sk-virtual".to_owned()), false)]
    #[case::no_token(jwt_auth(true), true, None, false)]
    fn only_a_request_planned_as_a_jwt_reads_the_model(
        #[case] general: GeneralSettings,
        #[case] premium: bool,
        #[case] token: Option<String>,
        #[case] reads: bool,
    ) {
        let auth = authenticator(DecliningHost::default(), general, premium);
        let credential = token.map_or(Credential::Transport, |token| {
            Credential::Token(SecretValue::from(token))
        });

        assert_eq!(auth.reads_model(&credential, &messages_request()), reads);
    }

    #[rstest]
    #[case::the_model_its_scope_reaches(Model::Named(Some("claude")), None)]
    #[case::a_model_its_scope_does_not_reach(Model::Named(Some("gpt")), Some("forbidden"))]
    #[case::a_body_the_server_did_not_read(Model::Unread, Some("not_native"))]
    #[tokio::test]
    async fn a_jwt_is_judged_on_the_model_the_server_read(
        #[case] model: Model<'static>,
        #[case] refused: Option<&str>,
    ) {
        let host = DecliningHost {
            jwks: Some(jwks()),
            ..DecliningHost::default()
        };
        let auth = authenticator(host, scoped_to_claude(), true);
        let token = Credential::Token(SecretValue::from(admin_jwt()));

        let answered = auth.authenticate(&token, &messages_request(), model).await;

        assert_eq!(answered.err().map(<&'static str>::from), refused);
    }

    #[tokio::test]
    async fn the_team_header_a_request_carries_is_the_one_the_core_reads() {
        let host = DecliningHost {
            jwks: Some(jwks()),
            ..DecliningHost::default()
        };
        let auth = authenticator(host, jwt_auth(true), true);
        let token = Credential::Token(SecretValue::from(admin_jwt()));
        let request = HttpRequest::post("/v1/messages")
            .header(TEAM_HEADER, "team-1")
            .body(())
            .unwrap()
            .into_parts()
            .0;

        let _ = auth.authenticate(&token, &request, Model::Unread).await;

        assert_eq!(
            auth.host.team_asked.get().map(String::as_str),
            Some("team-1")
        );
    }
}
