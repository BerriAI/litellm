mod admission;
pub mod authn;
pub mod authz;
mod commit;
mod error;
mod host;
pub mod identity;
mod outcome;
mod pipeline;
pub mod plan;
mod public;
mod request;
pub mod routes;
mod settings;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod testing;
mod token;
mod value;

pub use error::{
    AgentIdentityFailure, AssertionRejection, BudgetExceeded, ClaimDenial, Error, JwtDenial,
    KeyDenial, KeyError, KeyFailure, ModelAccessDenied, PremiumRequired, RbacDenial, Searched,
    TeamDenial, TokenError, VerifyDenial,
};
pub use host::{
    AgentRow, AuthHost, EndUserRow, KeyRow, MembershipRow, ObjectPermissionRow, OperatorHooks,
    OrgRow, ProjectRow, Row, Rows, TeamRow, UserRow,
};
pub use outcome::{Identity, KeyOutcome, Outcome, Refused};
pub use public::{ErrorType, PublicDenial};
pub use request::Request;
pub use routes::{
    ManagedRoute, MissingRouteGroup, RouteAllowlist, RouteGroup, RouteGroups, matches_wildcard,
};
pub use settings::{HostFacts, Settings};
pub use token::{KeyHash, MasterKey, Token};

use crate::{
    authn::key::Credential,
    identity::{
        jwt::{ClaimIdentity, JwtIdentity, JwtMapping, JwtOutcome, Provisioning},
        key::KeyIdentity,
    },
    plan::Plan,
};

type Authenticated<H> = Result<Outcome<<H as AuthHost>::Rows>, Error>;

/// Which path authenticates the request, in the proxy's order. Custom auth runs before it, in the
/// host, since it depends only on the host's configuration.
pub fn plan(request: &Request<'_>, settings: &Settings) -> Result<Plan, PremiumRequired> {
    plan::plan(&request.plan(), &settings.plan)
}

/// The master key and virtual key path, authenticating `key`, which a pass-through endpoint may
/// have read from another header than the token the plan was made on.
pub async fn authenticate_key<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    key: Option<&Token>,
    settings: &Settings,
) -> Authenticated<H> {
    let credential = Credential::Presented(key);
    key_outcome(
        pipeline::key::authenticate(
            host,
            &request.key(),
            &settings.key,
            &settings.policy(),
            credential,
        )
        .await,
    )
}

/// The JWT path alone, for callers that already know the token is a JWT (MCP token flows,
/// pass-through endpoints). A JWT mapped to a virtual key continues on the key path.
pub async fn authenticate_jwt<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Authenticated<H> {
    let token = request.token_or_empty();
    let policy = settings.policy();
    let outcome =
        match pipeline::jwt::build(host, &token, &request.jwt(), &settings.jwt, &policy).await {
            Ok(outcome) => outcome,
            Err(error) => return denied(error, None),
        };
    match outcome {
        JwtOutcome::Admin(identity) => Ok(Outcome::Authenticated(Identity::AdminJwt(Box::new(
            identity,
        )))),
        JwtOutcome::Finished(token) => Ok(Outcome::Authenticated(Identity::FinishedJwt(token))),
        JwtOutcome::MappedKey(key, jwt) => {
            let credential = Credential::Resolved {
                key,
                end_user_id: jwt.end_user_id.as_deref(),
            };
            let outcome = pipeline::key::authenticate(
                host,
                &request.key(),
                &settings.key,
                &policy,
                credential,
            )
            .await;
            match outcome {
                Ok(KeyIdentity::VirtualKey(context)) => {
                    Ok(Outcome::Authenticated(Identity::MappedJwtKey(jwt, context)))
                }
                outcome => key_outcome(outcome),
            }
        }
    }
}

pub fn is_public_route(route: &str, settings: &Settings) -> bool {
    plan::is_public_route(route, &settings.plan)
}

/// Whether the planner sends this token to OAuth2 introspection, for callers that pick a validator
/// themselves (the MCP identity check).
pub fn oauth2_introspection_applies(request: &Request<'_>, settings: &Settings) -> bool {
    plan::oauth2_introspection_applies(&request.plan(), &settings.plan)
}

/// A denial ends the call as `Outcome::Denied`. Only a host failure leaves it as an error.
fn denied<R: Rows>(error: Error, key: Option<R::Key>) -> Result<Outcome<R>, Error> {
    match error.is_host_failure() {
        true => Err(error),
        false => Ok(Outcome::Denied(error, key)),
    }
}

fn key_outcome<R: Rows>(outcome: KeyOutcome<R>) -> Result<Outcome<R>, Error> {
    match outcome {
        Ok(identity) => Ok(Outcome::Authenticated(Identity::Key(identity))),
        Err(refused) => denied(*refused.error, refused.key),
    }
}

/// `JWTAuthManager.authorize_jwt` for the MCP token flows, which pass no provisioning.
pub async fn authorize_jwt<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
    provisioning: Option<Provisioning>,
) -> Result<JwtIdentity<H::Rows>, Error> {
    let token = request.token_or_empty();
    let policy = settings.policy();
    pipeline::jwt::authorize_jwt(
        host,
        &token,
        &request.jwt(),
        &settings.jwt,
        &policy,
        provisioning,
    )
    .await
}

/// `JWTAuthManager.resolve_identity`, the user an MCP bridge JWT names.
pub async fn resolve_jwt_identity<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Result<ClaimIdentity<UserRow<H>>, Error> {
    let token = request.token_or_empty();
    pipeline::jwt::resolve_identity(host, &token, request.now, &settings.jwt, &settings.policy())
        .await
}

/// The virtual key a JWT maps to, without provisioning one, for the MCP bridge.
pub async fn resolve_jwt_mapping<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Result<JwtMapping<KeyRow<H>>, Error> {
    pipeline::jwt::resolve_mapping(host, &request.token_or_empty(), request.now, &settings.jwt)
        .await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use http::Method;
    use litellm_config::{GeneralSettings, JwtAuthSettings, UnregisteredJwtClientBehavior};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        authn::{
            jwt::{JwtSettings, MappedKey},
            key::{KeyRecord, Keyed},
        },
        identity::jwt::User,
        testing::ScriptedHost,
    };

    fn settings() -> Settings {
        let general = GeneralSettings::default();
        let facts = HostFacts {
            master_key: Some(MasterKey::new("sk-master")),
            premium: false,
            has_db: true,
            db_configured: true,
            ui_login: false,
            proxy_budget: false,
            metrics_require_auth: false,
            route_groups: crate::test_support::route_groups(),
        };
        Settings::new(&general, facts, &|_| None)
    }

    #[tokio::test]
    async fn a_failed_host_step_is_left_to_the_host() {
        let host = ScriptedHost {
            failing: HashMap::from([("db_key", Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        let token = Token::new("sk-1");
        let request = Request {
            route: "/chat/completions",
            token: Some(&token),
            ..Request::default()
        };
        assert_eq!(
            authenticate_key(&host, &request, request.token, &settings()).await,
            Err(Error::HostFailed)
        );
    }

    fn mapped(hash: &'static str, record: KeyRecord) -> (&'static str, MappedKey<String>) {
        let key = Keyed {
            record: KeyRecord {
                token: Some(KeyHash::new(hash)),
                ..record
            },
            row: format!("{hash} row"),
        };
        (
            hash,
            MappedKey {
                key,
                user_email: None,
            },
        )
    }

    /// JWTs map to a virtual key by `claim_field`.
    fn jwt_settings(claim_field: &str, behavior: UnregisteredJwtClientBehavior) -> Settings {
        let base = settings();
        Settings {
            jwt: JwtSettings {
                config: JwtAuthSettings {
                    oidc_userinfo_enabled: true,
                    oidc_userinfo_endpoint: Some("https://idp/userinfo".to_owned()),
                    virtual_key_claim_field: Some(claim_field.into()),
                    unregistered_jwt_client_behavior: behavior,
                    user_id_jwt_field: Some("sub".into()),
                    end_user_id_jwt_field: Some("end".into()),
                    ..JwtAuthSettings::default()
                },
                ..base.jwt.clone()
            },
            ..base
        }
    }

    /// A chat request whose body names the end user `from-body`.
    async fn chat(host: &ScriptedHost, settings: &Settings) -> Authenticated<ScriptedHost> {
        let token = Token::new("opaque");
        let request = Request {
            route: "/chat/completions",
            method: Some(&Method::POST),
            token: Some(&token),
            end_user_id: Some("from-body"),
            runs_inference_checks: true,
            ..Request::default()
        };
        authenticate_jwt(host, &request, settings).await
    }

    /// The end user the key path validated and warmed the cache for.
    fn key_path_end_user(host: &ScriptedHost) -> Vec<String> {
        host.calls()
            .into_iter()
            .filter(|call| {
                call.starts_with("validated_end_user_id") || call.starts_with("prefetch_identity")
            })
            .collect()
    }

    #[rstest]
    #[case::the_claim_wins_over_the_request(json!("from-claim"), "from-claim")]
    #[case::an_empty_claim_leaves_the_requests(json!(""), "from-body")]
    #[case::a_missing_claim_leaves_the_requests(Value::Null, "from-body")]
    #[tokio::test]
    async fn a_mapped_jwt_names_the_end_user_its_key_is_checked_for(
        #[case] claim: Value,
        #[case] expected: &str,
    ) {
        let host = ScriptedHost {
            userinfo: Some(json!({"client_id": "app", "end": claim})),
            mappings: HashMap::from([("client_id=app -".to_owned(), "h1".to_owned())]),
            keys: HashMap::from([mapped("h1", KeyRecord::default())]),
            ..ScriptedHost::default()
        };

        let outcome = chat(
            &host,
            &jwt_settings("client_id", UnregisteredJwtClientBehavior::Reject),
        )
        .await;

        assert!(
            matches!(
                outcome,
                Ok(Outcome::Authenticated(Identity::MappedJwtKey(..)))
            ),
            "{outcome:?}"
        );
        assert_eq!(
            key_path_end_user(&host),
            [
                format!("validated_end_user_id {expected}"),
                format!("prefetch_identity {expected} resolved"),
            ]
        );
    }

    #[tokio::test]
    async fn a_jwt_linked_to_an_existing_key_names_the_end_user_that_key_is_checked_for() {
        let host = ScriptedHost {
            userinfo: Some(json!({"sub": "u1", "end": "from-claim"})),
            users: HashMap::from([(
                "u1",
                User {
                    user_id: "u1".to_owned(),
                    user_email: None,
                    user_role: Some("internal_user".to_owned()),
                    teams: Vec::new(),
                    deactivated: false,
                    row: "u1".to_owned(),
                },
            )]),
            reusable_key: Some("reused"),
            keys: HashMap::from([mapped("reused", KeyRecord::default())]),
            ..ScriptedHost::default()
        };

        let outcome = chat(
            &host,
            &jwt_settings("sub", UnregisteredJwtClientBehavior::LinkOnly),
        )
        .await;

        assert!(
            matches!(
                outcome,
                Ok(Outcome::Authenticated(Identity::MappedJwtKey(..)))
            ),
            "{outcome:?}"
        );
        assert_eq!(
            key_path_end_user(&host),
            [
                "validated_end_user_id from-claim",
                "prefetch_identity from-claim resolved"
            ]
        );
    }

    #[tokio::test]
    async fn a_mapped_jwt_whose_key_is_blocked_is_refused_with_that_key() {
        let blocked = KeyRecord {
            blocked: true,
            ..KeyRecord::default()
        };
        let host = ScriptedHost {
            userinfo: Some(json!({"client_id": "app"})),
            mappings: HashMap::from([("client_id=app -".to_owned(), "h1".to_owned())]),
            keys: HashMap::from([mapped("h1", blocked)]),
            ..ScriptedHost::default()
        };

        let outcome = chat(
            &host,
            &jwt_settings("client_id", UnregisteredJwtClientBehavior::Reject),
        )
        .await;

        assert_eq!(
            outcome,
            Ok(Outcome::Denied(
                KeyDenial::Blocked.into(),
                Some("h1 row".to_owned())
            ))
        );
    }
}
