pub mod authorize;
pub mod budgets;
pub mod end_user;
mod error;
mod glob;
mod host;
pub mod jwt;
pub mod key;
pub mod key_budget;
mod outcome;
pub mod passthrough;
pub mod plan;
mod public;
mod request;
pub mod route_access;
mod routes;
mod settings;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod testing;
mod value;

pub use error::{
    AuthorizeDenial, BudgetExceeded, Dependency, Error, KeyDenial, PremiumRequired, RouteDenial, SessionDenial,
    Unavailable,
};
pub use host::{
    AgentRow, AuthHost, EndUserRow, KeyRow, MembershipRow, ObjectPermissionRow, OrgRow, ProjectRow,
    Row, Rows, TeamRow, UserRow,
};
pub use outcome::{Identity, Outcome};
pub use public::{ErrorType, PublicDenial};
pub use request::Request;
pub use routes::{ManagedRoute, managed_agent_route_allowed, managed_route};
pub use settings::{HostFacts, Settings};

use crate::{
    jwt::{ClaimIdentity, JwtIdentity, JwtMapping, JwtOutcome, Provisioning},
    key::{KeyIdentity, KeyOutcome},
    plan::Plan,
};

type Authenticated<H> = Result<Outcome<<H as AuthHost>::Rows>, Error>;

/// Which path authenticates the request, in the proxy's order. Custom auth runs before it, in the
/// host, since it depends only on the host's configuration.
pub fn plan(request: &Request<'_>, settings: &Settings) -> Result<Plan, PremiumRequired> {
    plan::plan(&request.plan(), &settings.plan)
}

/// The master key and virtual key path, authenticating `key`.
pub async fn authenticate_key<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    key: Option<&str>,
    settings: &Settings,
) -> Authenticated<H> {
    served(
        key_outcome(key::authenticate(host, &request.key(key), &settings.key, None).await),
        settings,
    )
}

/// The JWT path alone, for callers that already know the token is a JWT (MCP token flows,
/// pass-through endpoints). A JWT mapped to a virtual key continues on the key path.
pub async fn authenticate_jwt<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Authenticated<H> {
    served(jwt_path(host, request, settings).await, settings)
}

/// `common_checks` for a request the host already authenticated.
pub async fn authorize<H: AuthHost>(
    host: &H,
    request: &authorize::AuthorizeRequest<'_>,
) -> Result<(), Error> {
    authorize::authorize(host, request).await
}

/// Whether the planner sends this token to OAuth2 introspection, for callers that pick a validator
/// themselves (the MCP identity check).
pub fn oauth2_introspection_applies(request: &Request<'_>, settings: &Settings) -> bool {
    plan::oauth2_introspection_applies(&request.plan(), &settings.plan)
}

/// `handle_authentication_error`: an unreachable database lets the request through as a
/// restricted user when the operator allows it, and is a 503 otherwise.
fn served<R: Rows>(
    outcome: Result<Outcome<R>, Error>,
    settings: &Settings,
) -> Result<Outcome<R>, Error> {
    match outcome {
        Err(Error::Unavailable(unavailable))
            if unavailable.transient && settings.allow_requests_on_db_unavailable =>
        {
            Ok(Outcome::Authenticated(Identity::DbUnavailable(unavailable)))
        }
        Err(unavailable @ Error::Unavailable(_)) => Ok(Outcome::Denied(unavailable, None)),
        outcome => outcome,
    }
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
        Err(refused) => denied(refused.error, refused.key),
    }
}

async fn jwt_path<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Authenticated<H> {
    let token = request.token.unwrap_or_default();
    let outcome = match jwt::authenticate(host, token, &request.jwt(), &settings.jwt).await {
        Ok(outcome) => outcome,
        Err(error) => return denied(error, None),
    };
    match outcome {
        JwtOutcome::Admin(identity) => Ok(Outcome::Authenticated(Identity::AdminJwt(Box::new(
            identity,
        )))),
        JwtOutcome::Finished(token) => Ok(Outcome::Authenticated(Identity::FinishedJwt(token))),
        JwtOutcome::MappedKey(key, jwt) => {
            let token = key.record.token.clone();
            match key::authenticate(
                host,
                &request.key(token.as_deref()),
                &settings.key,
                Some(key),
            )
            .await
            {
                Ok(KeyIdentity::VirtualKey(context)) => {
                    Ok(Outcome::Authenticated(Identity::MappedJwtKey(jwt, context)))
                }
                outcome => key_outcome(outcome),
            }
        }
    }
}

/// `JWTAuthManager.authorize_jwt` for the MCP token flows, which pass no provisioning.
pub async fn authorize_jwt<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
    provisioning: Option<Provisioning>,
) -> Result<JwtIdentity<H::Rows>, Error> {
    let token = request.token.unwrap_or_default();
    jwt::authorize_jwt(host, token, &request.jwt(), &settings.jwt, provisioning).await
}

/// `JWTAuthManager.resolve_identity`, the user an MCP bridge JWT names.
pub async fn resolve_jwt_identity<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Result<ClaimIdentity<UserRow<H>>, Error> {
    jwt::resolve_identity(
        host,
        request.token.unwrap_or_default(),
        request.now,
        &settings.jwt,
    )
    .await
}

/// The virtual key a JWT maps to, without provisioning one, for the MCP bridge.
pub async fn resolve_jwt_mapping<H: AuthHost>(
    host: &H,
    request: &Request<'_>,
    settings: &Settings,
) -> Result<JwtMapping<KeyRow<H>>, Error> {
    jwt::resolve_mapping(
        host,
        request.token.unwrap_or_default(),
        request.now,
        &settings.jwt,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use litellm_config::{GeneralSettings, LenientFlag};
    use litellm_secret_value::SecretValue;
    use rstest::rstest;

    use super::*;
    use crate::{
        key::KeyRecord,
        testing::{Labels, ScriptedHost},
    };

    fn settings(allow_requests_on_db_unavailable: bool) -> Settings {
        let general = GeneralSettings {
            allow_requests_on_db_unavailable: LenientFlag(allow_requests_on_db_unavailable),
            ..GeneralSettings::default()
        };
        let facts = HostFacts {
            master_key: Some(SecretValue::new("sk-master")),
            premium: false,
            has_db: true,
            ui_login: false,
            proxy_budget: false,
            metrics_require_auth: false,
        };
        Settings::new(&general, facts, &|_| None)
    }

    fn unavailable(transient: bool) -> Unavailable {
        Unavailable {
            dependency: Dependency::Db,
            transient,
            message: "database down".to_owned(),
        }
    }

    #[rstest]
    #[case::allowed_transient(
        true,
        true,
        Outcome::Authenticated(Identity::DbUnavailable(unavailable(true)))
    )]
    #[case::not_allowed(
        false,
        true,
        Outcome::Denied(Error::Unavailable(unavailable(true)), None)
    )]
    #[case::allowed_but_permanent(
        true,
        false,
        Outcome::Denied(Error::Unavailable(unavailable(false)), None)
    )]
    #[tokio::test]
    async fn unreachable_database_is_served_only_when_allowed_and_transient(
        #[case] allow: bool,
        #[case] transient: bool,
        #[case] expected: Outcome<Labels>,
    ) {
        let host = ScriptedHost {
            db: Some(KeyRecord::default()),
            failing: HashMap::from([("db_key", Error::Unavailable(unavailable(transient)))]),
            ..ScriptedHost::default()
        };
        let request = Request {
            route: "/chat/completions",
            token: Some("sk-1"),
            ..Request::default()
        };
        assert_eq!(
            authenticate_key(&host, &request, request.token, &settings(allow)).await,
            Ok(expected)
        );
    }

    #[tokio::test]
    async fn other_host_failures_are_left_to_the_host() {
        let host = ScriptedHost {
            failing: HashMap::from([("db_key", Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        let request = Request {
            route: "/chat/completions",
            token: Some("sk-1"),
            ..Request::default()
        };
        assert_eq!(
            authenticate_key(&host, &request, request.token, &settings(true)).await,
            Err(Error::HostFailed)
        );
    }
}
