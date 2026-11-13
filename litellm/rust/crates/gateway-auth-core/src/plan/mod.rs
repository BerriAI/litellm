mod introspection;
mod public_routes;

use litellm_config::GeneralSettings;
use strum::VariantArray;

pub use self::{introspection::Introspection, public_routes::PublicRoutes};
use crate::{PremiumRequired, credential::key::Token, routes::RouteGroups};

#[derive(Clone, Debug, Default)]
pub struct Settings {
    public_routes: PublicRoutes,
    introspection: Introspection,
    /// [`general_settings.enable_oauth2_proxy_auth`](GeneralSettings::enable_oauth2_proxy_auth)
    oauth2_proxy: bool,
    /// [`general_settings.enable_jwt_auth`](GeneralSettings::enable_jwt_auth)
    jwt_auth: bool,
    /// [`litellm_jwtauth.custom_validate`](litellm_config::JwtAuthSettings::custom_validate) is set
    jwt_operator_code: bool,
    /// An enterprise license ([`HostFacts::premium`](crate::HostFacts::premium)), which JWT auth and
    /// OAuth2 introspection need.
    premium: bool,
}

impl Settings {
    pub fn new(
        general: &GeneralSettings,
        premium: bool,
        metrics_require_auth: bool,
        route_groups: &RouteGroups,
    ) -> Self {
        Self {
            public_routes: PublicRoutes::new(general, premium, metrics_require_auth, route_groups),
            introspection: Introspection::new(general),
            oauth2_proxy: general.enable_oauth2_proxy_auth == Some(true),
            jwt_auth: general.enable_jwt_auth == Some(true),
            jwt_operator_code: general
                .litellm_jwtauth
                .as_ref()
                .is_some_and(|jwt| jwt.custom_validate.is_some()),
            premium,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Request<'a> {
    pub route: &'a str,
    pub token: Option<&'a Token>,
    pub is_llm_api_route: bool,
    pub is_info_route: bool,
    pub pass_through_open: bool,
    /// Custom auth handed back the token, so an open pass-through route still authenticates it.
    pub from_custom_auth: bool,
}

/// Declared in the proxy's order: the first plan that applies wins. Custom auth runs before all of
/// them, in the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, VariantArray)]
pub enum Plan {
    Public,
    OAuth2Introspection,
    OAuth2Proxy,
    /// A JWT the operator's `custom_validate` also judges. Only a host with
    /// [`OperatorHooks`](crate::OperatorHooks) authenticates it, and any other hands the request
    /// to one before authenticating.
    JwtWithOperatorCode,
    Jwt,
    /// A configured pass-through endpoint with `auth: false`.
    PassThroughOpen,
    Key,
}

impl Plan {
    fn applies(self, request: &Request<'_>, settings: &Settings) -> bool {
        match self {
            Self::Public => settings.public_routes.contains(request.route),
            Self::OAuth2Introspection => oauth2_introspection_applies(request, settings),
            Self::OAuth2Proxy => settings.oauth2_proxy,
            Self::JwtWithOperatorCode => {
                settings.jwt_operator_code
                    && matches!(bearer_token(request, settings), BearerToken::Jwt(_))
            }
            Self::Jwt => matches!(bearer_token(request, settings), BearerToken::Jwt(_)),
            Self::PassThroughOpen => request.pass_through_open && !request.from_custom_auth,
            Self::Key => true,
        }
    }

    fn license(self) -> Option<PremiumRequired> {
        match self {
            Self::OAuth2Introspection => Some(PremiumRequired::OAuth2),
            Self::JwtWithOperatorCode | Self::Jwt => Some(PremiumRequired::Jwt),
            Self::Public | Self::OAuth2Proxy | Self::PassThroughOpen | Self::Key => None,
        }
    }
}

/// With JWT auth off, a JWT-shaped token is just an opaque token.
#[derive(Clone, Copy, Debug)]
enum BearerToken<'a> {
    Opaque,
    Jwt(&'a Token),
}

fn bearer_token<'a>(request: &Request<'a>, settings: &Settings) -> BearerToken<'a> {
    match request.token {
        Some(token) if settings.jwt_auth && token.is_jwt() => BearerToken::Jwt(token),
        _ => BearerToken::Opaque,
    }
}

pub(crate) fn oauth2_introspection_applies(request: &Request<'_>, settings: &Settings) -> bool {
    settings
        .introspection
        .applies(request, bearer_token(request, settings))
}

pub fn plan(request: &Request<'_>, settings: &Settings) -> Result<Plan, PremiumRequired> {
    let plan = Plan::VARIANTS
        .iter()
        .copied()
        .find(|plan| plan.applies(request, settings))
        .unwrap_or(Plan::Key);
    match plan.license() {
        Some(feature) if !settings.premium => Err(feature),
        _ => Ok(plan),
    }
}

#[cfg(test)]
mod tests {
    use litellm_config::{
        GeneralSettings, JwtAuthSettings, JwtRoutingOverride, OneOrMany, RoutingPath,
    };
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::test_support::{route_groups, unsigned_jwt};

    const JWT: &str = "header.claims.signature";
    const KEY: &str = "sk-1234";
    const ISSUER: &str = "oauth-issuer.example.com";

    fn configured(premium: bool, metrics_require_auth: bool) -> Settings {
        let general = GeneralSettings {
            enable_jwt_auth: Some(true),
            public_routes: Some(Box::new([
                "/public-api/*".to_owned(),
                "/spend/calculate".to_owned(),
                "/metrics".to_owned(),
                "/metrics/".to_owned(),
            ])),
            ..GeneralSettings::default()
        };
        Settings::new(&general, premium, metrics_require_auth, &route_groups())
    }

    #[fixture]
    fn settings() -> Settings {
        configured(true, true)
    }

    fn request<'a>(route: &'a str, token: &'a Token) -> Request<'a> {
        Request {
            route,
            token: Some(token),
            ..Request::default()
        }
    }

    #[rstest]
    #[case::jwt(JWT, Ok(Plan::Jwt))]
    #[case::key(KEY, Ok(Plan::Key))]
    #[case::two_dots_is_not_a_jwt("a.b", Ok(Plan::Key))]
    #[case::four_parts_is_not_a_jwt("a.b.c.d", Ok(Plan::Key))]
    #[case::empty_parts_still_count("..", Ok(Plan::Jwt))]
    fn tells_jwt_from_key(
        settings: Settings,
        #[case] token: &str,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        assert_eq!(
            plan(&request("/chat/completions", &Token::new(token)), &settings),
            expected
        );
    }

    #[rstest]
    fn jwt_off_sends_jwts_to_keys(settings: Settings) {
        let settings = Settings {
            jwt_auth: false,
            ..settings
        };
        assert_eq!(
            plan(&request("/chat/completions", &Token::new(JWT)), &settings),
            Ok(Plan::Key)
        );
    }

    #[rstest]
    #[case::a_jwt_needs_the_operators_code(JWT, Ok(Plan::JwtWithOperatorCode))]
    #[case::a_key_does_not(KEY, Ok(Plan::Key))]
    fn custom_validate_plans_jwts_for_a_host_with_operator_code(
        #[case] token: &str,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        let general = GeneralSettings {
            enable_jwt_auth: Some(true),
            litellm_jwtauth: Some(JwtAuthSettings {
                custom_validate: Some("hooks.validate".to_owned()),
                ..JwtAuthSettings::default()
            }),
            ..GeneralSettings::default()
        };
        let settings = Settings::new(&general, true, true, &route_groups());
        assert_eq!(
            plan(&request("/chat/completions", &Token::new(token)), &settings),
            expected
        );
    }

    #[rstest]
    fn jwt_without_license_is_refused(settings: Settings) {
        let settings = Settings {
            premium: false,
            ..settings
        };
        assert_eq!(
            plan(&request("/chat/completions", &Token::new(JWT)), &settings),
            Err(PremiumRequired::Jwt)
        );
        assert_eq!(
            plan(&request("/chat/completions", &Token::new(KEY)), &settings),
            Ok(Plan::Key)
        );
    }

    #[rstest]
    #[case::builtin("/", true, Ok(Plan::Public))]
    #[case::additional_exact("/spend/calculate", true, Ok(Plan::Public))]
    #[case::additional_wildcard("/public-api/users/1", true, Ok(Plan::Public))]
    #[case::additional_needs_license("/spend/calculate", false, Ok(Plan::Key))]
    #[case::builtin_without_license("/", false, Ok(Plan::Public))]
    fn public_routes(
        #[case] route: &str,
        #[case] premium: bool,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        assert_eq!(
            plan(
                &request(route, &Token::new(KEY)),
                &configured(premium, true)
            ),
            expected
        );
    }

    #[rstest]
    #[case::requires_auth("/metrics", true, Ok(Plan::Key))]
    #[case::trailing_slash_requires_auth("/metrics/", true, Ok(Plan::Key))]
    #[case::trailing_slash_public("/metrics/", false, Ok(Plan::Public))]
    #[case::public("/metrics", false, Ok(Plan::Public))]
    fn metrics(
        #[case] route: &str,
        #[case] metrics_require_auth: bool,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        assert_eq!(
            plan(
                &request(route, &Token::new(KEY)),
                &configured(true, metrics_require_auth)
            ),
            expected
        );
    }

    #[rstest]
    #[case::llm_route_key("/chat/completions", true, false, KEY, Ok(Plan::OAuth2Introspection))]
    #[case::info_route_key("/key/info", false, true, KEY, Ok(Plan::OAuth2Introspection))]
    #[case::lens_route("/lens/x", false, false, KEY, Ok(Plan::OAuth2Introspection))]
    #[case::lens_lookalike("/lensx", false, false, KEY, Ok(Plan::Key))]
    #[case::uncovered_route("/team/new", false, false, KEY, Ok(Plan::Key))]
    #[case::jwt_stays_jwt("/chat/completions", true, false, JWT, Ok(Plan::Jwt))]
    fn oauth2_introspection(
        settings: Settings,
        #[case] route: &str,
        #[case] is_llm_api_route: bool,
        #[case] is_info_route: bool,
        #[case] token: &str,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        let token = Token::new(token);
        let request = Request {
            is_llm_api_route,
            is_info_route,
            ..request(route, &token)
        };
        let settings = Settings {
            introspection: Introspection::new(&GeneralSettings {
                enable_oauth2_auth: Some(true),
                ..GeneralSettings::default()
            }),
            ..settings
        };
        assert_eq!(plan(&request, &settings), expected);
    }

    #[rstest]
    fn routed_jwt_needs_a_license(settings: Settings) {
        let settings = Settings {
            premium: false,
            introspection: Introspection::new(&GeneralSettings {
                litellm_jwtauth: Some(JwtAuthSettings {
                    routing_overrides: Some(Box::new([JwtRoutingOverride {
                        iss: OneOrMany::One(ISSUER.to_owned()),
                        client_id: None,
                        scope: None,
                        aud: None,
                        path: RoutingPath::OAuth2,
                    }])),
                    ..JwtAuthSettings::default()
                }),
                ..GeneralSettings::default()
            }),
            ..settings
        };
        let token = Token::new(unsigned_jwt(&json!({"iss": ISSUER})));
        let routed = Request {
            is_llm_api_route: true,
            ..request("/chat/completions", &token)
        };
        assert_eq!(plan(&routed, &settings), Err(PremiumRequired::OAuth2));
        assert_eq!(
            plan(&request("/team/new", &token), &settings),
            Err(PremiumRequired::Jwt)
        );
    }

    #[rstest]
    fn oauth2_proxy_wins_over_jwt(settings: Settings) {
        let settings = Settings {
            oauth2_proxy: true,
            ..settings
        };
        assert_eq!(
            plan(&request("/chat/completions", &Token::new(JWT)), &settings),
            Ok(Plan::OAuth2Proxy)
        );
    }

    #[rstest]
    #[case::key_on_an_open_route(KEY, Ok(Plan::PassThroughOpen))]
    #[case::jwt_is_planned_before_it(JWT, Ok(Plan::Jwt))]
    fn auth_off_pass_through_route(
        settings: Settings,
        #[case] token: &str,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        let token = Token::new(token);
        let open = Request {
            pass_through_open: true,
            ..request("/forwarder", &token)
        };
        assert_eq!(plan(&open, &settings), expected);
    }

    #[rstest]
    fn key_from_custom_auth_skips_the_open_pass_through(settings: Settings) {
        let key = Token::new(KEY);
        let handed_back = Request {
            pass_through_open: true,
            from_custom_auth: true,
            ..request("/forwarder", &key)
        };
        assert_eq!(plan(&handed_back, &settings), Ok(Plan::Key));
    }

    #[rstest]
    fn public_route_wins_over_an_open_pass_through(settings: Settings) {
        let key = Token::new(KEY);
        let open = Request {
            pass_through_open: true,
            ..request("/", &key)
        };
        assert_eq!(plan(&open, &settings), Ok(Plan::Public));
    }
}
