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
            let credential = Credential::Resolved(key);
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

    use litellm_config::GeneralSettings;

    use super::*;
    use crate::testing::ScriptedHost;

    fn settings() -> Settings {
        let general = GeneralSettings::default();
        let facts = HostFacts {
            master_key: Some(MasterKey::new("sk-master")),
            premium: false,
            has_db: true,
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
}
