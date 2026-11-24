mod admission;
pub mod authn;
pub mod authz;
mod belongs;
pub mod body;
mod caller;
pub mod client_address;
mod commit;
mod error;
mod host;
pub mod identity;
mod outcome;
mod pipeline;
pub mod plan;
mod public;
pub mod record;
mod request;
pub mod routes;
mod settings;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod testing;
mod token;
mod value;

pub use belongs::{Belongs, Holder, Named, Team, TeamCopy};
pub use caller::{Caller, Credential, NotACaller, StoredKey};
pub use error::{
    AddressNotAllowed, AgentIdentityFailure, AssertionRejection, AuthzDenial, BudgetExceeded,
    ClaimDenial, DeniedList, Error, JwtDenial, KeyDenial, KeyError, KeyFailure, ModelAccessDenied,
    PremiumRequired, RbacDenial, RequestDenial, Searched, TeamDenial, TokenError, VerifyDenial,
};
pub use host::{
    AgentRow, AuthHost, EndUserRow, KeyRow, MembershipRow, ObjectPermissionRow, OperatorHooks,
    OrgRow, ProjectRow, Row, Rows, TeamRow, UserRow,
};
pub use outcome::{Identity, KeyOutcome, Outcome, Refused};
pub use public::{ErrorType, PublicDenial};
pub use request::Request;
pub use routes::{
    ExposedRoutes, ManagedRoute, MissingRouteGroup, RouteAllowlist, RouteGroup, RouteGroups,
    RouteNotExposed, matches_wildcard,
};
pub use settings::{Allowlists, HostFacts, Settings};
pub use token::{KeyHash, MasterKey, Token};

use crate::{
    authn::key::KeySource,
    client_address::Origin,
    identity::{
        jwt::{ClaimIdentity, JwtIdentity, JwtMapping, JwtOutcome, Provisioning},
        key::KeyIdentity,
    },
    plan::Plan,
};

type Authenticated<H> = Result<Outcome<<H as AuthHost>::Rows>, Error>;

/// Whether the proxy takes the request at all, decided before any credential is read: its address,
/// then its route, as `pre_db_read_auth_checks` orders them.
pub fn check_request(
    route: &str,
    origin: Origin<'_>,
    allowlists: &Allowlists,
) -> Result<(), RequestDenial> {
    allowlists.client_addresses.permits(origin)?;
    allowlists.exposed_routes.permits(route)?;
    Ok(())
}

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
    let credential = KeySource::Presented(key);
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
        JwtOutcome::Jwt(identity) => Ok(Outcome::Authenticated(Identity::Jwt(Box::new(identity)))),
        JwtOutcome::AutoRegistered(registered) => Ok(Outcome::Authenticated(
            Identity::AutoRegisteredJwt(Box::new(registered)),
        )),
        JwtOutcome::MappedKey(key, jwt) => {
            let credential = KeySource::Resolved {
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
    use litellm_config::{GeneralSettings, JwtAuthSettings, RouteList};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        authn::{
            jwt::{JwtSettings, MappedKey},
            key::{KeyRecord, Keyed},
        },
        testing::ScriptedHost,
    };

    fn settings() -> Settings {
        settings_from(&GeneralSettings::default())
    }

    fn settings_from(general: &GeneralSettings) -> Settings {
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
        Settings::new(general, facts, &|_| None)
    }

    fn allowlists() -> Allowlists {
        let list = |entry: &str| RouteList::Valid(Box::from([entry.to_owned()]));
        Allowlists::new(&GeneralSettings {
            allowed_ips: Some(list("10.0.0.1")),
            allowed_routes: Some(list("/health")),
            ..GeneralSettings::default()
        })
    }

    fn from(peer: &str) -> Origin<'_> {
        Origin {
            peer: Some(peer),
            forwarded_for: None,
        }
    }

    #[rstest]
    #[case::listed_address_and_route("10.0.0.1", "/health", None)]
    #[case::unlisted_route(
        "10.0.0.1",
        "/v1/messages",
        Some("Access forbidden: Route /v1/messages not allowed")
    )]
    #[case::unlisted_address(
        "10.0.0.9",
        "/health",
        Some("Access forbidden: IP address 10.0.0.9 not allowed.")
    )]
    #[case::the_address_is_refused_before_the_route(
        "10.0.0.9",
        "/v1/messages",
        Some("Access forbidden: IP address 10.0.0.9 not allowed.")
    )]
    fn a_request_is_refused_with_the_403_the_proxy_raised(
        #[case] peer: &str,
        #[case] route: &str,
        #[case] detail: Option<&str>,
    ) {
        let refused = check_request(route, from(peer), &allowlists())
            .err()
            .map(|denial| denial.public());

        assert_eq!(
            refused,
            detail.map(|detail| PublicDenial::Http {
                status: 403,
                detail: detail.to_owned(),
            })
        );
    }

    #[rstest]
    fn no_allowlist_takes_every_request() {
        assert_eq!(
            check_request("/v1/messages", from("10.0.0.9"), &Allowlists::default()),
            Ok(())
        );
    }

    #[rstest]
    #[case::a_route_the_public_group_lists("/health/liveliness", true)]
    #[case::any_other_route("/chat/completions", false)]
    fn a_route_is_public_when_the_public_group_lists_it(
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(is_public_route(route, &settings()), expected);
    }

    #[rstest]
    #[case::on_an_llm_route(true, true)]
    #[case::on_a_route_introspection_does_not_cover(false, false)]
    fn an_opaque_token_is_introspected_once_oauth2_auth_is_on(
        #[case] is_llm_api_route: bool,
        #[case] expected: bool,
    ) {
        let general = GeneralSettings {
            enable_oauth2_auth: true,
            ..GeneralSettings::default()
        };
        let token = Token::new("opaque");
        let request = Request {
            route: "/chat/completions",
            token: Some(&token),
            is_llm_api_route,
            ..Request::default()
        };
        assert_eq!(
            oauth2_introspection_applies(&request, &settings_from(&general)),
            expected
        );
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

    fn host_mapping_the_client(claims: Value, record: KeyRecord) -> ScriptedHost {
        let key = Keyed {
            record: KeyRecord {
                token: Some(KeyHash::new("h1")),
                ..record
            },
            row: "h1 row".to_owned(),
        };
        let mapped = MappedKey {
            key,
            user_email: None,
        };
        ScriptedHost {
            userinfo: Some(claims),
            mappings: HashMap::from([("client_id=app -".to_owned(), "h1".to_owned())]),
            keys: HashMap::from([("h1", mapped)]),
            ..ScriptedHost::default()
        }
    }

    async fn chat(host: &ScriptedHost) -> Authenticated<ScriptedHost> {
        let base = settings();
        let settings = Settings {
            jwt: JwtSettings {
                config: JwtAuthSettings {
                    oidc_userinfo_enabled: true,
                    oidc_userinfo_endpoint: Some("https://idp/userinfo".to_owned()),
                    virtual_key_claim_field: Some("client_id".into()),
                    end_user_id_jwt_field: Some("end".into()),
                    ..JwtAuthSettings::default()
                },
                ..base.jwt.clone()
            },
            ..base
        };
        let token = Token::new("opaque");
        let request = Request {
            route: "/chat/completions",
            method: Some(&Method::POST),
            token: Some(&token),
            end_user_id: Some("from-body"),
            runs_inference_checks: true,
            ..Request::default()
        };
        authenticate_jwt(host, &request, &settings).await
    }

    #[rstest]
    #[tokio::test]
    async fn a_mapped_jwt_has_its_key_checked_for_the_end_user_its_claim_names() {
        let host = host_mapping_the_client(
            json!({"client_id": "app", "end": "from-claim"}),
            KeyRecord::default(),
        );

        let outcome = chat(&host).await;

        assert!(
            matches!(
                outcome,
                Ok(Outcome::Authenticated(Identity::MappedJwtKey(..)))
            ),
            "{outcome:?}"
        );
        let end_user_reads: Vec<String> = host
            .calls()
            .into_iter()
            .filter(|call| {
                call.starts_with("validated_end_user_id") || call.starts_with("prefetch_identity")
            })
            .collect();
        assert_eq!(
            end_user_reads,
            [
                "validated_end_user_id from-claim",
                "prefetch_identity from-claim resolved"
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_mapped_jwt_whose_key_is_blocked_is_refused_with_that_key() {
        let blocked = KeyRecord {
            blocked: true,
            ..KeyRecord::default()
        };
        let host = host_mapping_the_client(json!({"client_id": "app"}), blocked);

        assert_eq!(
            chat(&host).await,
            Ok(Outcome::Denied(
                KeyDenial::Blocked.into(),
                Some("h1 row".to_owned())
            ))
        );
    }
}
