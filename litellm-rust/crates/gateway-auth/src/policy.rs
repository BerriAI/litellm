use std::{sync::Arc, time::SystemTime};

use fancy_regex::Regex;

use crate::AccessRequest;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Denied {
    #[error("key is expired")]
    Expired,
    #[error("key is blocked")]
    KeyBlocked,
    #[error("team is blocked")]
    TeamBlocked,
    #[error("key model access denied")]
    KeyModel,
    #[error("team model access denied")]
    TeamModel,
    #[error("user model access denied")]
    UserModel,
    #[error("team member model access denied")]
    MemberModel,
    #[error("route access denied")]
    Route,
    #[error("request type is unsupported")]
    UnsupportedRequest,
}

#[derive(Debug, thiserror::Error)]
pub enum InvalidPolicy {
    #[error("invalid model pattern {pattern}")]
    ModelPattern {
        pattern: String,
        #[source]
        source: fancy_regex::Error,
    },
    #[error("unsupported route pattern {0}")]
    RoutePattern(String),
    #[error("unsupported model sentinel in an expanded grant")]
    ModelSentinel,
}

const ALL_PROXY_MODELS: &str = "all-proxy-models";
const ALL_TEAM_MODELS: &str = "all-team-models";

#[derive(Clone, Debug)]
enum ModelPattern {
    Exact(String),
    Regex(Regex),
}

impl ModelPattern {
    fn new(pattern: String) -> Result<Self, InvalidPolicy> {
        if !pattern.contains('*') {
            return Ok(Self::Exact(pattern));
        }
        let expression = format!("^{}$", pattern.replace('*', ".*"));
        let regex = Regex::new(&expression).map_err(|source| InvalidPolicy::ModelPattern {
            pattern: pattern.clone(),
            source,
        })?;
        Ok(Self::Regex(regex))
    }

    fn matches(&self, model: &str) -> bool {
        match self {
            Self::Exact(pattern) => pattern == model,
            Self::Regex(regex) => regex.is_match(model).unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug)]
enum ModelScopeKind {
    Unrestricted,
    Patterns(Arc<[ModelPattern]>),
    ExpandedEmpty,
    AllTeamModels,
}

#[derive(Clone, Debug)]
pub struct ModelScope(ModelScopeKind);

impl ModelScope {
    pub fn unrestricted() -> Self {
        Self(ModelScopeKind::Unrestricted)
    }

    pub fn patterns<I, S>(patterns: I) -> Result<Self, InvalidPolicy>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let patterns: Vec<String> = patterns.into_iter().map(Into::into).collect();
        if patterns.is_empty() {
            return Ok(Self::unrestricted());
        }
        let has_all_team_models = patterns.iter().any(|pattern| pattern == ALL_TEAM_MODELS);
        let has_all_proxy_models = patterns.iter().any(|pattern| pattern == ALL_PROXY_MODELS);
        let patterns: Result<Vec<ModelPattern>, InvalidPolicy> =
            patterns.into_iter().map(ModelPattern::new).collect();
        let patterns = patterns?;
        if has_all_team_models {
            return Ok(Self(ModelScopeKind::AllTeamModels));
        }
        if has_all_proxy_models {
            return Ok(Self::unrestricted());
        }
        Ok(Self(ModelScopeKind::Patterns(patterns.into())))
    }

    /// Builds a model scope from caller-resolved model names and access groups.
    ///
    /// The caller must resolve provider aliases and access groups before construction. This helper does not provide full runtime resolver parity.
    pub fn expanded<I, S>(models: I) -> Result<Self, InvalidPolicy>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let models: Vec<String> = models.into_iter().map(Into::into).collect();
        if models.is_empty() {
            return Ok(Self(ModelScopeKind::ExpandedEmpty));
        }
        if models
            .iter()
            .any(|model| model == ALL_PROXY_MODELS || model == ALL_TEAM_MODELS)
        {
            return Err(InvalidPolicy::ModelSentinel);
        }
        let patterns: Result<Vec<ModelPattern>, InvalidPolicy> =
            models.into_iter().map(ModelPattern::new).collect();
        Ok(Self(ModelScopeKind::Patterns(patterns?.into())))
    }

    pub fn is_all_team_models(&self) -> bool {
        matches!(self.0, ModelScopeKind::AllTeamModels)
    }

    fn allows(&self, model: &str) -> bool {
        match &self.0 {
            ModelScopeKind::Unrestricted => true,
            ModelScopeKind::ExpandedEmpty => false,
            ModelScopeKind::AllTeamModels => false,
            ModelScopeKind::Patterns(patterns) => {
                patterns.iter().any(|pattern| pattern.matches(model))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TeamPolicy {
    models: ModelScope,
    aliases: Arc<[String]>,
    blocked: bool,
}

impl TeamPolicy {
    pub fn new(models: ModelScope) -> Result<Self, InvalidPolicy> {
        if models.is_all_team_models() {
            return Err(InvalidPolicy::ModelSentinel);
        }
        Ok(Self {
            models,
            aliases: Arc::default(),
            blocked: false,
        })
    }

    pub fn with_aliases<I, S>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        // Python parity: a team alias is granted only when present in the resolved alias map (auth_checks.py:5167-5191, 2026-10-25).
        self.aliases = aliases
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>()
            .into();
        self
    }

    pub fn blocked(mut self, blocked: bool) -> Self {
        self.blocked = blocked;
        self
    }

    fn allows(&self, model: &str) -> bool {
        self.models.allows(model) || self.allows_alias(model)
    }

    fn allows_alias(&self, model: &str) -> bool {
        self.aliases.iter().any(|alias| alias == model)
    }
}

#[derive(Clone, Debug)]
enum RoutePattern {
    Literal(String),
    WildcardPrefix(String),
}

impl RoutePattern {
    fn new(pattern: String) -> Result<Self, InvalidPolicy> {
        if pattern.is_empty() || pattern.contains('{') || pattern.contains('}') {
            return Err(InvalidPolicy::RoutePattern(pattern));
        }
        if pattern != "*" && !pattern.starts_with('/') {
            return Err(InvalidPolicy::RoutePattern(pattern));
        }
        if let Some(prefix) = pattern.strip_suffix('*') {
            return Ok(Self::WildcardPrefix(prefix.to_owned()));
        }
        if pattern.contains('*') {
            return Err(InvalidPolicy::RoutePattern(pattern));
        }
        Ok(Self::Literal(pattern))
    }

    fn matches(&self, route: &str) -> bool {
        match self {
            Self::Literal(path) => route == path || route.starts_with(&format!("{path}/")),
            Self::WildcardPrefix(prefix) => route.starts_with(prefix),
        }
    }
}

#[derive(Clone, Debug)]
enum RouteScopeKind {
    Unrestricted,
    ExpandedEmpty,
    Patterns(Arc<[RoutePattern]>),
}

#[derive(Clone, Debug)]
pub struct RouteScope(RouteScopeKind);

impl RouteScope {
    pub fn unrestricted() -> Self {
        Self(RouteScopeKind::Unrestricted)
    }

    pub fn allowed<I, S>(routes: I) -> Result<Self, InvalidPolicy>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let routes: Result<Vec<RoutePattern>, InvalidPolicy> = routes
            .into_iter()
            .map(|route| RoutePattern::new(route.into()))
            .collect();
        let routes = routes?;
        if routes.is_empty() {
            return Ok(Self::unrestricted());
        }
        Ok(Self(RouteScopeKind::Patterns(routes.into())))
    }

    /// Builds a route scope from caller-expanded named route groups.
    ///
    /// The caller must resolve named route groups before construction. This helper does not provide full runtime resolver parity.
    pub fn expanded<I, S>(routes: I) -> Result<Self, InvalidPolicy>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let routes: Result<Vec<RoutePattern>, InvalidPolicy> = routes
            .into_iter()
            .map(|route| RoutePattern::new(route.into()))
            .collect();
        let routes = routes?;
        if routes.is_empty() {
            return Ok(Self(RouteScopeKind::ExpandedEmpty));
        }
        Ok(Self(RouteScopeKind::Patterns(routes.into())))
    }

    fn allows(&self, route: &str) -> bool {
        match &self.0 {
            RouteScopeKind::Unrestricted => true,
            RouteScopeKind::ExpandedEmpty => false,
            RouteScopeKind::Patterns(patterns) => {
                patterns.iter().any(|pattern| pattern.matches(route))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct KeyPolicy {
    key_models: ModelScope,
    team: Option<TeamPolicy>,
    user_models: Option<ModelScope>,
    member_models: Option<ModelScope>,
    blocked: bool,
    expires_at: Option<SystemTime>,
    routes: RouteScope,
}

impl KeyPolicy {
    pub fn new(
        key_models: ModelScope,
        team: Option<TeamPolicy>,
        user_models: Option<ModelScope>,
        member_models: Option<ModelScope>,
        blocked: bool,
        expires_at: Option<SystemTime>,
        routes: RouteScope,
    ) -> Result<Self, InvalidPolicy> {
        if team
            .as_ref()
            .is_some_and(|team| team.models.is_all_team_models())
        {
            return Err(InvalidPolicy::ModelSentinel);
        }
        Ok(Self {
            key_models,
            team,
            user_models,
            member_models,
            blocked,
            expires_at,
            routes,
        })
    }

    pub fn unrestricted() -> Self {
        Self {
            key_models: ModelScope::unrestricted(),
            team: None,
            user_models: None,
            member_models: None,
            blocked: false,
            expires_at: None,
            routes: RouteScope::unrestricted(),
        }
    }

    pub fn check(&self, request: &AccessRequest, now: SystemTime) -> Result<(), Denied> {
        if matches!(request, AccessRequest::Mcp { .. } | AccessRequest::Ui(_)) {
            return Err(Denied::UnsupportedRequest);
        }
        if self.expires_at.is_some_and(|expires_at| expires_at <= now) {
            return Err(Denied::Expired);
        }
        if self.blocked {
            return Err(Denied::KeyBlocked);
        }
        if self.team.as_ref().is_some_and(|team| team.blocked) {
            return Err(Denied::TeamBlocked);
        }
        match request {
            AccessRequest::Route { path, .. } => {
                if self.routes.allows(path) {
                    Ok(())
                } else {
                    Err(Denied::Route)
                }
            }
            AccessRequest::Model { name, .. } => {
                let key_models = if self.key_models.is_all_team_models() {
                    self.team
                        .as_ref()
                        .map_or(ModelScope::unrestricted(), |team| team.models.clone())
                } else {
                    self.key_models.clone()
                };
                // Python parity: resolved team aliases participate in key model access checks (auth_checks.py:4970-4984, 2026-10-25).
                let allows_team_alias = !matches!(self.key_models.0, ModelScopeKind::ExpandedEmpty)
                    && self
                        .team
                        .as_ref()
                        .is_some_and(|team| team.allows_alias(name));
                if !key_models.allows(name) && !allows_team_alias {
                    return Err(Denied::KeyModel);
                }
                if let Some(team) = &self.team
                    && !team.allows(name)
                {
                    return Err(Denied::TeamModel);
                }
                if self
                    .user_models
                    .as_ref()
                    .is_some_and(|models| !models.allows(name))
                {
                    return Err(Denied::UserModel);
                }
                if self
                    .member_models
                    .as_ref()
                    .is_some_and(|models| !models.allows(name))
                {
                    return Err(Denied::MemberModel);
                }
                Ok(())
            }
            AccessRequest::Mcp { .. } | AccessRequest::Ui(_) => Err(Denied::UnsupportedRequest),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use rstest::{fixture, rstest};

    use crate::{McpAction, UiAction};

    use super::*;

    #[fixture]
    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_000)
    }

    fn model_request(name: &str) -> AccessRequest {
        AccessRequest::Model {
            name: name.to_owned(),
            deployment: "resolved-by-caller".to_owned(),
        }
    }

    fn assert_error_kind(result: Result<(), Denied>, expected: &Denied) {
        assert_eq!(result, Err(*expected));
    }

    #[rstest]
    #[case::key_and_team_intersection("gpt-4o", Denied::TeamModel)]
    #[case::key_denial("claude-3", Denied::KeyModel)]
    fn model_scopes_intersect(#[case] name: &str, #[case] denial: Denied, now: SystemTime) {
        let key = ModelScope::patterns(["gpt-*"]).expect("valid key scope");
        let team = TeamPolicy::new(ModelScope::patterns(["claude-*"]).expect("valid team scope"))
            .expect("valid team");
        let policy = KeyPolicy::new(
            key,
            Some(team),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");

        assert_error_kind(policy.check(&model_request(name), now), &denial);
    }

    #[rstest]
    fn model_pattern_uses_python_regex_semantics(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::patterns(["gpt-4.o*"]).expect("valid regex pattern"),
            None,
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("gpt-4ao-fine"), now).is_ok());
    }

    #[rstest]
    #[case::all_proxy("gpt-4o", vec!["all-proxy-models"])]
    #[case::empty_input("gpt-4o", Vec::<&str>::new())]
    fn unrestricted_sentinels_allow(
        #[case] name: &str,
        #[case] models: Vec<&str>,
        now: SystemTime,
    ) {
        let policy = KeyPolicy::new(
            ModelScope::patterns(models).expect("valid scope"),
            None,
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request(name), now).is_ok());
    }

    #[rstest]
    fn all_team_models_replaces_key_list(now: SystemTime) {
        let team = TeamPolicy::new(ModelScope::patterns(["gpt-*"]).expect("valid team scope"))
            .expect("valid team");
        let policy = KeyPolicy::new(
            ModelScope::patterns(["all-team-models"]).expect("valid sentinel"),
            Some(team),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("gpt-4o"), now).is_ok());
        assert_error_kind(
            policy.check(&model_request("claude-3"), now),
            &Denied::KeyModel,
        );
    }

    #[rstest]
    fn all_team_models_without_team_are_unrestricted(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::patterns(["all-team-models"]).expect("valid sentinel"),
            None,
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("gpt-4o"), now).is_ok());
    }

    #[rstest]
    fn aliases_match_only_caller_supplied_alias_names(now: SystemTime) {
        let team = TeamPolicy::new(ModelScope::expanded(["gpt-4o"]).expect("valid team scope"))
            .expect("valid team")
            .with_aliases(["team-alias"]);
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            Some(team),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("team-alias"), now).is_ok());
        assert_error_kind(
            policy.check(&model_request("resolved-by-caller"), now),
            &Denied::TeamModel,
        );
    }

    #[rstest]
    fn restricted_key_accepts_team_alias_without_bypassing_member_scope(now: SystemTime) {
        let team = TeamPolicy::new(ModelScope::patterns(["team-*"]).expect("valid team scope"))
            .expect("valid team")
            .with_aliases(["team-alias"]);
        let policy = KeyPolicy::new(
            ModelScope::patterns(["key-*"]).expect("valid key scope"),
            Some(team.clone()),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("team-alias"), now).is_ok());
        assert_error_kind(
            policy.check(&model_request("unknown-alias"), now),
            &Denied::KeyModel,
        );

        let restricted = KeyPolicy::new(
            ModelScope::patterns(["key-*"]).expect("valid key scope"),
            Some(team),
            None,
            Some(ModelScope::expanded(["member-*"]).expect("member scope")),
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            restricted.check(&model_request("team-alias"), now),
            &Denied::MemberModel,
        );
    }

    #[rstest]
    fn all_team_models_inherits_team_aliases(now: SystemTime) {
        let team = TeamPolicy::new(ModelScope::patterns(["team-*"]).expect("valid team scope"))
            .expect("valid team")
            .with_aliases(["team-alias"]);
        let policy = KeyPolicy::new(
            ModelScope::patterns(["all-team-models"]).expect("valid sentinel"),
            Some(team),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("team-alias"), now).is_ok());
    }

    #[rstest]
    fn expanded_empty_key_scope_rejects_team_alias(now: SystemTime) {
        let team = TeamPolicy::new(ModelScope::patterns(["team-*"]).expect("valid team scope"))
            .expect("valid team")
            .with_aliases(["team-alias"]);
        let policy = KeyPolicy::new(
            ModelScope::expanded(Vec::<&str>::new()).expect("empty expansion"),
            Some(team),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            policy.check(&model_request("team-alias"), now),
            &Denied::KeyModel,
        );
    }

    #[rstest]
    fn user_scope_intersects_key(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            Some(ModelScope::expanded(["gpt-4o"]).expect("user scope")),
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            policy.check(&model_request("claude-3"), now),
            &Denied::UserModel,
        );
    }

    #[rstest]
    fn non_empty_member_scope_intersects_key(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            Some(ModelScope::expanded(["gpt-4o"]).expect("member scope")),
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            policy.check(&model_request("claude-3"), now),
            &Denied::MemberModel,
        );
    }

    #[rstest]
    fn empty_member_scope_does_not_restrict(now: SystemTime) {
        let team = TeamPolicy::new(ModelScope::patterns(["gpt-*"]).expect("valid team scope"))
            .expect("valid team");
        let member = ModelScope::patterns(Vec::<&str>::new()).expect("empty scope");
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            Some(team),
            None,
            Some(member),
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert!(policy.check(&model_request("gpt-4o"), now).is_ok());
    }

    #[rstest]
    fn expanded_empty_member_scope_denies(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            Some(ModelScope::expanded(Vec::<&str>::new()).expect("empty expansion")),
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            policy.check(&model_request("gpt-4o"), now),
            &Denied::MemberModel,
        );
    }

    #[rstest]
    fn expanded_empty_key_scope_denies(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::expanded(Vec::<&str>::new()).expect("empty expansion"),
            None,
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            policy.check(&model_request("gpt-4o"), now),
            &Denied::KeyModel,
        );
    }

    #[rstest]
    #[case::configured_empty(
        RouteScope::allowed(Vec::<&str>::new()).expect("empty configured routes"),
        true
    )]
    #[case::expanded_empty(
        RouteScope::expanded(Vec::<&str>::new()).expect("empty expanded routes"),
        false
    )]
    fn empty_route_scopes_preserve_resolution_semantics(
        #[case] routes: RouteScope,
        #[case] expected: bool,
        now: SystemTime,
    ) {
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            None,
            false,
            None,
            routes,
        )
        .expect("valid policy");
        assert_eq!(
            policy
                .check(
                    &AccessRequest::Route {
                        method: "POST".to_owned(),
                        path: "/v1/chat/completions".to_owned()
                    },
                    now
                )
                .is_ok(),
            expected
        );
    }

    #[rstest]
    fn route_prefix_obeys_boundary(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            None,
            false,
            None,
            RouteScope::allowed(["/v1/chat"]).expect("valid route"),
        )
        .expect("valid policy");
        assert!(
            policy
                .check(
                    &AccessRequest::Route {
                        method: "POST".to_owned(),
                        path: "/v1/chat/completions".to_owned()
                    },
                    now
                )
                .is_ok()
        );
        assert_error_kind(
            policy.check(
                &AccessRequest::Route {
                    method: "POST".to_owned(),
                    path: "/v1/chatter".to_owned(),
                },
                now,
            ),
            &Denied::Route,
        );
    }

    #[rstest]
    fn wildcard_route_matches_python_prefix(now: SystemTime) {
        let policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            None,
            false,
            None,
            RouteScope::allowed(["/v1/chat*"]).expect("valid route"),
        )
        .expect("valid policy");
        assert!(
            policy
                .check(
                    &AccessRequest::Route {
                        method: "POST".to_owned(),
                        path: "/v1/chatter".to_owned()
                    },
                    now
                )
                .is_ok()
        );
    }

    #[rstest]
    fn blocked_expired_and_unsupported_fail_closed(now: SystemTime) {
        let blocked = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            None,
            true,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            blocked.check(&model_request("gpt-4o"), now),
            &Denied::KeyBlocked,
        );

        let blocked_team = TeamPolicy::new(ModelScope::unrestricted())
            .expect("valid team")
            .blocked(true);
        let team_blocked_policy = KeyPolicy::new(
            ModelScope::unrestricted(),
            Some(blocked_team),
            None,
            None,
            false,
            None,
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            team_blocked_policy.check(&model_request("gpt-4o"), now),
            &Denied::TeamBlocked,
        );

        let expired = KeyPolicy::new(
            ModelScope::unrestricted(),
            None,
            None,
            None,
            false,
            Some(now),
            RouteScope::unrestricted(),
        )
        .expect("valid policy");
        assert_error_kind(
            expired.check(&model_request("gpt-4o"), now),
            &Denied::Expired,
        );

        let unsupported = KeyPolicy::unrestricted();
        assert_error_kind(
            unsupported.check(
                &AccessRequest::Mcp {
                    server: "server".to_owned(),
                    action: McpAction::Connect,
                },
                now,
            ),
            &Denied::UnsupportedRequest,
        );
        assert_error_kind(
            unsupported.check(&AccessRequest::Ui(UiAction::Logout), now),
            &Denied::UnsupportedRequest,
        );
    }

    #[rstest]
    fn unsupported_patterns_fail_construction() {
        assert!(matches!(
            ModelScope::patterns(["[*"]),
            Err(InvalidPolicy::ModelPattern { .. })
        ));
        assert!(matches!(
            RouteScope::allowed(["/threads/{id}"]),
            Err(InvalidPolicy::RoutePattern(_))
        ));
        assert!(matches!(
            RouteScope::allowed(["/v1/ch*t"]),
            Err(InvalidPolicy::RoutePattern(_))
        ));
        assert!(matches!(
            RouteScope::allowed(["chat-completions"]),
            Err(InvalidPolicy::RoutePattern(_))
        ));
        assert!(matches!(
            ModelScope::expanded(["all-proxy-models"]),
            Err(InvalidPolicy::ModelSentinel)
        ));
    }
}
