mod introspection;
mod public_routes;

use litellm_config::GeneralSettings;
use strum::VariantArray;

pub use self::{introspection::Introspection, public_routes::PublicRoutes};
use crate::{PremiumRequired, jwt::is_jwt};

const LENS_ROUTE: &str = "/lens";

#[derive(Clone, Debug, Default)]
pub struct Settings {
    public_routes: PublicRoutes,
    introspection: Introspection,
    /// [`general_settings.enable_oauth2_proxy_auth`](GeneralSettings::enable_oauth2_proxy_auth)
    oauth2_proxy: bool,
    /// [`general_settings.enable_jwt_auth`](GeneralSettings::enable_jwt_auth)
    jwt_auth: bool,
    /// An enterprise license ([`HostFacts::premium`](crate::HostFacts::premium)), which JWT auth and
    /// OAuth2 introspection need.
    premium: bool,
}

impl Settings {
    pub fn new(general: &GeneralSettings, premium: bool, metrics_require_auth: bool) -> Self {
        Self {
            public_routes: PublicRoutes::new(general, premium, metrics_require_auth),
            introspection: Introspection::new(general),
            oauth2_proxy: general.enable_oauth2_proxy_auth == Some(true),
            jwt_auth: general.enable_jwt_auth == Some(true),
            premium,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Request<'a> {
    pub route: &'a str,
    pub token: &'a str,
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
            Self::Jwt => matches!(bearer_token(request, settings), BearerToken::Jwt(_)),
            Self::PassThroughOpen => request.pass_through_open && !request.from_custom_auth,
            Self::Key => true,
        }
    }

    fn license(self) -> Option<PremiumRequired> {
        match self {
            Self::OAuth2Introspection => Some(PremiumRequired::OAuth2),
            Self::Jwt => Some(PremiumRequired::Jwt),
            Self::Public | Self::OAuth2Proxy | Self::PassThroughOpen | Self::Key => None,
        }
    }
}

/// With JWT auth off, a JWT-shaped token is just an opaque token.
#[derive(Clone, Copy, Debug)]
enum BearerToken<'a> {
    Opaque,
    Jwt(&'a str),
}

fn bearer_token<'a>(request: &Request<'a>, settings: &Settings) -> BearerToken<'a> {
    match settings.jwt_auth && is_jwt(request.token) {
        true => BearerToken::Jwt(request.token),
        false => BearerToken::Opaque,
    }
}

fn introspection_route(request: &Request<'_>) -> bool {
    request.is_llm_api_route
        || request.is_info_route
        || request.route == LENS_ROUTE
        || request.route.starts_with("/lens/")
}

pub(crate) fn oauth2_introspection_applies(request: &Request<'_>, settings: &Settings) -> bool {
    introspection_route(request)
        && settings
            .introspection
            .validates(bearer_token(request, settings))
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
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use litellm_config::{
        GeneralSettings, JwtAuthSettings, JwtRoutingOverride, OneOrMany, RoutingPath,
    };
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::test_support::{many, one};

    const JWT: &str = "header.claims.signature";
    const KEY: &str = "sk-1234";

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
        Settings::new(&general, premium, metrics_require_auth)
    }

    #[fixture]
    fn settings() -> Settings {
        configured(true, true)
    }

    fn request<'a>(route: &'a str, token: &'a str) -> Request<'a> {
        Request {
            route,
            token,
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
            plan(&request("/chat/completions", token), &settings),
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
            plan(&request("/chat/completions", JWT), &settings),
            Ok(Plan::Key)
        );
    }

    #[rstest]
    fn jwt_without_license_is_refused(settings: Settings) {
        let settings = Settings {
            premium: false,
            ..settings
        };
        assert_eq!(
            plan(&request("/chat/completions", JWT), &settings),
            Err(PremiumRequired::Jwt)
        );
        assert_eq!(
            plan(&request("/chat/completions", KEY), &settings),
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
            plan(&request(route, KEY), &configured(premium, true)),
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
                &request(route, KEY),
                &configured(true, metrics_require_auth)
            ),
            expected
        );
    }

    #[rstest]
    #[case::llm_route_key(Request { is_llm_api_route: true, ..request("/chat/completions", KEY) }, Ok(Plan::OAuth2Introspection))]
    #[case::info_route_key(Request { is_info_route: true, ..request("/key/info", KEY) }, Ok(Plan::OAuth2Introspection))]
    #[case::lens_route(request("/lens/x", KEY), Ok(Plan::OAuth2Introspection))]
    #[case::lens_lookalike(request("/lensx", KEY), Ok(Plan::Key))]
    #[case::uncovered_route(request("/team/new", KEY), Ok(Plan::Key))]
    #[case::jwt_stays_jwt(Request { is_llm_api_route: true, ..request("/chat/completions", JWT) }, Ok(Plan::Jwt))]
    fn oauth2_introspection(
        settings: Settings,
        #[case] request: Request<'_>,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        let settings = Settings {
            introspection: Introspection::new(&GeneralSettings {
                enable_oauth2_auth: Some(true),
                ..GeneralSettings::default()
            }),
            ..settings
        };
        assert_eq!(plan(&request, &settings), expected);
    }

    fn unsigned(claims: Value) -> String {
        let segment = |value: &Value| URL_SAFE_NO_PAD.encode(serde_json::to_vec(value).unwrap());
        format!(
            "{}.{}.AAAA",
            segment(&json!({"alg": "RS256"})),
            segment(&claims)
        )
    }

    fn routing(
        client_id: Option<OneOrMany>,
        scope: Option<OneOrMany>,
        aud: Option<OneOrMany>,
    ) -> JwtRoutingOverride {
        JwtRoutingOverride {
            iss: OneOrMany::One(ISSUER.to_owned()),
            client_id,
            scope,
            aud,
            path: RoutingPath::OAuth2,
        }
    }

    const ISSUER: &str = "oauth-issuer.example.com";

    fn routed(routing: JwtRoutingOverride) -> Introspection {
        Introspection::new(&GeneralSettings {
            litellm_jwtauth: Some(JwtAuthSettings {
                routing_overrides: Some(Box::new([routing])),
                ..JwtAuthSettings::default()
            }),
            ..GeneralSettings::default()
        })
    }

    #[rstest]
    #[case::issuer_only(routing(None, None, None), json!({"iss": ISSUER}), Ok(Plan::OAuth2Introspection))]
    #[case::other_issuer(routing(None, None, None), json!({"iss": "other.example.com"}), Ok(Plan::Jwt))]
    #[case::scope_narrows(routing(None, one("App:LiteLLM"), None), json!({"iss": ISSUER, "scope": "App:LiteLLM openid"}), Ok(Plan::OAuth2Introspection))]
    #[case::client_id_wildcard(routing(one("*MID_LITELLM"), None, None), json!({"iss": ISSUER, "client_id": "BATCH_MID_LITELLM"}), Ok(Plan::OAuth2Introspection))]
    #[case::client_id_miss(routing(one("*MID_LITELLM"), None, None), json!({"iss": ISSUER, "client_id": "BATCH_PORTAL"}), Ok(Plan::Jwt))]
    #[case::list_audience(routing(None, None, many(&["api://litellm", "api://fallback"])), json!({"iss": ISSUER, "aud": ["api://other", "api://litellm"]}), Ok(Plan::OAuth2Introspection))]
    #[case::selectors_are_anded(routing(one("*MID_LITELLM"), one("App:LiteLLM"), None), json!({"iss": ISSUER, "scope": "App:Other openid", "client_id": "BATCH_MID_LITELLM"}), Ok(Plan::Jwt))]
    fn routing_override_sends_jwt_to_oauth2(
        settings: Settings,
        #[case] routing: JwtRoutingOverride,
        #[case] claims: Value,
        #[case] expected: Result<Plan, PremiumRequired>,
    ) {
        let settings = Settings {
            introspection: routed(routing),
            ..settings
        };
        let token = unsigned(claims);
        let routed = Request {
            is_llm_api_route: true,
            ..request("/chat/completions", &token)
        };
        assert_eq!(plan(&routed, &settings), expected);
    }

    #[rstest]
    fn routed_jwt_needs_a_license(settings: Settings) {
        let settings = Settings {
            premium: false,
            introspection: routed(routing(None, None, None)),
            ..settings
        };
        let token = unsigned(json!({"iss": ISSUER}));
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
            plan(&request("/chat/completions", JWT), &settings),
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
        let open = Request {
            pass_through_open: true,
            ..request("/forwarder", token)
        };
        assert_eq!(plan(&open, &settings), expected);
    }

    #[rstest]
    fn key_from_custom_auth_skips_the_open_pass_through(settings: Settings) {
        let handed_back = Request {
            pass_through_open: true,
            from_custom_auth: true,
            ..request("/forwarder", KEY)
        };
        assert_eq!(plan(&handed_back, &settings), Ok(Plan::Key));
    }

    #[rstest]
    fn public_route_wins_over_an_open_pass_through(settings: Settings) {
        let open = Request {
            pass_through_open: true,
            ..request("/", KEY)
        };
        assert_eq!(plan(&open, &settings), Ok(Plan::Public));
    }
}
