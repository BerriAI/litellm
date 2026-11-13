//! Which bearer tokens LiteLLM validates by calling the IdP's OAuth2 introspection endpoint instead
//! of validating them itself. A deployment can run JWT auth for one IdP and introspection for
//! another whose access tokens are also JWTs, so `general_settings.litellm_jwtauth.routing_overrides`
//! lists the JWTs that belong to introspection. Their claims are read unverified, only to pick the
//! validator, and introspection then authenticates the token.

use litellm_config::{GeneralSettings, JwtRoutingOverride, OneOrMany, RoutingPath};
use litellm_fnmatch::Pattern;
use serde_json::Value;

use super::{BearerToken, Request};
use crate::{
    credential::jwt::{Claims, unverified_claims},
    routes::table::is_lens_route,
    value::{id_text, is_separator, trim},
};

#[derive(Clone, Debug, Default)]
pub struct Introspection {
    /// [`general_settings.enable_oauth2_auth`](GeneralSettings::enable_oauth2_auth)
    opaque_tokens: bool,
    /// [`general_settings.litellm_jwtauth.routing_overrides`](litellm_config::JwtAuthSettings::routing_overrides),
    /// each a [`JwtRoutingOverride`]
    jwts: Box<[JwtSelector]>,
}

impl Introspection {
    pub fn new(general: &GeneralSettings) -> Self {
        Self {
            opaque_tokens: general.enable_oauth2_auth == Some(true),
            jwts: general
                .litellm_jwtauth
                .as_ref()
                .and_then(|jwt| jwt.routing_overrides.as_deref())
                .unwrap_or_default()
                .iter()
                .map(JwtSelector::new)
                .collect(),
        }
    }

    pub(super) fn applies(&self, request: &Request<'_>, token: BearerToken<'_>) -> bool {
        (request.is_llm_api_route || request.is_info_route || is_lens_route(request.route))
            && self.validates(token)
    }

    fn validates(&self, token: BearerToken<'_>) -> bool {
        match token {
            BearerToken::Opaque => self.opaque_tokens,
            BearerToken::Jwt(token) => {
                !self.jwts.is_empty()
                    && unverified_claims(token.expose())
                        .is_some_and(|claims| self.jwts.iter().any(|jwt| jwt.matches(&claims)))
            }
        }
    }
}

/// One `routing_overrides` entry. Every selector it sets has to match.
#[derive(Clone, Debug)]
struct JwtSelector {
    iss: ClaimPattern,
    client_id: Option<ClaimPattern>,
    aud: Option<ClaimPattern>,
    scope: Option<ScopePattern>,
}

impl JwtSelector {
    fn new(routing: &JwtRoutingOverride) -> Self {
        let JwtRoutingOverride {
            iss,
            client_id,
            scope,
            aud,
            path: RoutingPath::OAuth2,
        } = routing;
        Self {
            iss: ClaimPattern(selectors(iss)),
            client_id: client_id
                .as_ref()
                .map(|values| ClaimPattern(selectors(values))),
            aud: aud.as_ref().map(|values| ClaimPattern(selectors(values))),
            scope: scope.as_ref().map(|values| ScopePattern(selectors(values))),
        }
    }

    fn matches(&self, claims: &Claims) -> bool {
        let claim = |name: &str| claims.0.get(name);
        self.iss.matches(claim("iss"))
            && self
                .client_id
                .as_ref()
                .is_none_or(|pattern| pattern.matches(claim("client_id")))
            && self
                .aud
                .as_ref()
                .is_none_or(|pattern| pattern.matches(claim("aud")))
            && self
                .scope
                .as_ref()
                .is_none_or(|pattern| pattern.matches(claim("scope")))
    }
}

#[derive(Clone, Debug)]
enum Selector {
    Exact(String),
    Glob(Pattern),
}

fn selectors(values: &OneOrMany) -> Box<[Selector]> {
    values
        .values()
        .iter()
        .map(|value| match value.contains(['*', '?']) {
            true => Selector::Glob(Pattern::new(value)),
            false => Selector::Exact(value.clone()),
        })
        .collect()
}

/// Matches a claim value whole. A wildcard never spans whitespace, so `trusted.*` cannot match an
/// unsigned `trusted.example.com evil.com`.
#[derive(Clone, Debug)]
struct ClaimPattern(Box<[Selector]>);

impl ClaimPattern {
    fn matches(&self, claim: Option<&Value>) -> bool {
        whole_values(claim).iter().any(|value| {
            self.0.iter().any(|selector| match selector {
                Selector::Exact(exact) => exact == value,
                Selector::Glob(glob) => !value.contains(is_separator) && glob.matches(value),
            })
        })
    }
}

/// OAuth sends `scope` as one space-separated string, so each scope is matched on its own.
#[derive(Clone, Debug)]
struct ScopePattern(Box<[Selector]>);

impl ScopePattern {
    fn matches(&self, claim: Option<&Value>) -> bool {
        scope_values(claim).iter().any(|value| {
            self.0.iter().any(|selector| match selector {
                Selector::Exact(exact) => exact == value,
                Selector::Glob(glob) => glob.matches(value),
            })
        })
    }
}

fn whole_values(claim: Option<&Value>) -> Vec<String> {
    match claim {
        Some(Value::Array(items)) => items.iter().filter_map(id_text).collect(),
        Some(other) => id_text(other).into_iter().collect(),
        None => Vec::new(),
    }
}

fn scope_values(claim: Option<&Value>) -> Vec<String> {
    match claim {
        Some(Value::String(text)) if trim(text).contains(' ') => trim(text)
            .split(' ')
            .filter(|scope| !scope.is_empty())
            .map(str::to_owned)
            .collect(),
        other => whole_values(other),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        credential::key::Token,
        test_support::{many, one, unsigned_jwt},
    };

    const ISSUER: &str = "oauth-issuer.example.com";

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

    fn claim_pattern(values: Option<OneOrMany>) -> ClaimPattern {
        ClaimPattern(selectors(&values.unwrap()))
    }

    fn scope_pattern(values: Option<OneOrMany>) -> ScopePattern {
        ScopePattern(selectors(&values.unwrap()))
    }

    #[rstest]
    #[case::exact(one("issuer.example.com"), Some(json!("issuer.example.com")), true)]
    #[case::other_value(one("issuer.example.com"), Some(json!("other-issuer.example.com")), false)]
    #[case::not_split(one("trusted.example.com"), Some(json!("trusted.example.com attacker.example.com")), false)]
    #[case::wildcard_never_spans_spaces(one("trusted.*"), Some(json!("trusted.example.com attacker.example.com")), false)]
    #[case::wildcard(one("trusted.*"), Some(json!("trusted.example.com")), true)]
    #[case::any_selector(many(&["issuer-a.example.com", "issuer-b.example.com"]), Some(json!("issuer-b.example.com")), true)]
    #[case::leading_star(one("*MID_LITELLM"), Some(json!("STREAM_MID_LITELLM")), true)]
    #[case::leading_star_miss(one("*MID_LITELLM"), Some(json!("REDIS_LITELLM")), false)]
    #[case::question_marks(one("machine-??"), Some(json!("machine-01")), true)]
    #[case::question_marks_count(one("machine-??"), Some(json!("machine-001")), false)]
    #[case::wildcard_is_case_sensitive(one("*litellm"), Some(json!("BATCH_LITELLM")), false)]
    #[case::wildcard_same_case(one("*LITELLM"), Some(json!("BATCH_LITELLM")), true)]
    #[case::any_wildcard(many(&["service-*", "batch-*"]), Some(json!("batch-123")), true)]
    #[case::no_wildcard_matches(many(&["service-*", "batch-*"]), Some(json!("other-123")), false)]
    #[case::list_claim(one("App:LiteLLM"), Some(json!(["openid", "App:LiteLLM"])), true)]
    #[case::missing_claim(one("App:LiteLLM"), None, false)]
    #[case::null_claim(one("App:LiteLLM"), Some(Value::Null), false)]
    #[case::number_claim(one("42"), Some(json!(42)), true)]
    #[case::bool_claim_is_no_value(one("True"), Some(json!(true)), false)]
    fn claim_patterns_match_whole_values(
        #[case] values: Option<OneOrMany>,
        #[case] claim: Option<Value>,
        #[case] expected: bool,
    ) {
        assert_eq!(claim_pattern(values).matches(claim.as_ref()), expected);
    }

    #[rstest]
    #[case::split(one("App:LiteLLM"), Some(json!("App:LiteLLM openid")), true)]
    #[case::wildcard(one("App:*"), Some(json!("App:LiteLLM openid")), true)]
    #[case::any_selector(many(&["openid", "App:LiteLLM"]), Some(json!("openid profile")), true)]
    #[case::list(one("App:LiteLLM"), Some(json!(["openid", "App:LiteLLM"])), true)]
    #[case::missing(one("App:LiteLLM"), None, false)]
    #[case::no_scope_matches(one("App:LiteLLM"), Some(json!("openid profile")), false)]
    fn scope_patterns_match_each_scope(
        #[case] values: Option<OneOrMany>,
        #[case] claim: Option<Value>,
        #[case] expected: bool,
    ) {
        assert_eq!(scope_pattern(values).matches(claim.as_ref()), expected);
    }

    #[rstest]
    #[case::issuer_only(routing(None, None, None), json!({"iss": ISSUER}), true)]
    #[case::other_issuer(routing(None, None, None), json!({"iss": "other.example.com"}), false)]
    #[case::scope_narrows(routing(None, one("App:LiteLLM"), None), json!({"iss": ISSUER, "scope": "App:LiteLLM openid"}), true)]
    #[case::client_id_wildcard(routing(one("*MID_LITELLM"), None, None), json!({"iss": ISSUER, "client_id": "BATCH_MID_LITELLM"}), true)]
    #[case::client_id_miss(routing(one("*MID_LITELLM"), None, None), json!({"iss": ISSUER, "client_id": "BATCH_PORTAL"}), false)]
    #[case::list_audience(routing(None, None, many(&["api://litellm", "api://fallback"])), json!({"iss": ISSUER, "aud": ["api://other", "api://litellm"]}), true)]
    #[case::selectors_are_anded(routing(one("*MID_LITELLM"), one("App:LiteLLM"), None), json!({"iss": ISSUER, "scope": "App:Other openid", "client_id": "BATCH_MID_LITELLM"}), false)]
    fn routing_override_selects_jwts(
        #[case] routing: JwtRoutingOverride,
        #[case] claims: Value,
        #[case] expected: bool,
    ) {
        let introspection = Introspection::new(&GeneralSettings {
            litellm_jwtauth: Some(litellm_config::JwtAuthSettings {
                routing_overrides: Some(Box::new([routing])),
                ..litellm_config::JwtAuthSettings::default()
            }),
            ..GeneralSettings::default()
        });
        let token = Token::new(unsigned_jwt(&claims));
        assert_eq!(introspection.validates(BearerToken::Jwt(&token)), expected);
    }
}
