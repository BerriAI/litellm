use std::{future::Future, pin::Pin, sync::Arc, time::SystemTime};

use litellm_auth_types::SecretValue;
use serde_json::{Map, Value};

use crate::{KeyError, hash_token};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KeyHash(String);

impl KeyHash {
    pub fn from_token(token: &SecretValue) -> Self {
        Self(hash_token(token.expose()))
    }

    /// A hash as the proxy stores and broadcasts it: 64 lowercase hex digits.
    pub fn from_hashed(hashed: &str) -> Option<Self> {
        (hashed.len() == 64
            && hashed
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
        .then(|| Self(hashed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Model names that grant every model, as in the proxy's `check_model_access_helper`.
const ALL_MODELS: [&str; 2] = ["*", "all-proxy-models"];

/// The model groups a key may call. Empty grants every model, like an empty `models` column.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAccess(Arc<[String]>);

impl ModelAccess {
    pub fn new(models: impl Into<Arc<[String]>>) -> Self {
        Self(models.into())
    }

    pub fn models(&self) -> &[String] {
        &self.0
    }

    /// Exact names, `*`/`all-proxy-models`, and `*` wildcards such as `claude-*` or `openai/*`.
    pub fn allows(&self, model: &str) -> bool {
        self.0.is_empty()
            || self.0.iter().any(|allowed| {
                ALL_MODELS.contains(&allowed.as_str())
                    || allowed == model
                    || (allowed.contains('*') && wildcard_match(allowed, model))
            })
    }
}

fn wildcard_match(pattern: &str, candidate: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return false;
    };
    let Some(mut rest) = candidate.strip_prefix(first) else {
        return false;
    };
    let mut parts = parts.peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            return rest.ends_with(part);
        }
        match rest.find(part) {
            Some(index) => rest = &rest[index + part.len()..],
            None => return false,
        }
    }
    rest.is_empty()
}

/// A key's own `models`, narrowed by its team's when it has one: the proxy requires both
/// `can_key_call_model` and `can_team_access_model` to pass.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyModels {
    pub key: ModelAccess,
    pub team: Option<ModelAccess>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelDenial {
    Key,
    Team,
}

impl KeyModels {
    pub fn check(&self, model: &str) -> Result<(), ModelDenial> {
        if !self.key.allows(model) {
            return Err(ModelDenial::Key);
        }
        if self.team.as_ref().is_some_and(|team| !team.allows(model)) {
            return Err(ModelDenial::Team);
        }
        Ok(())
    }
}

impl From<ModelAccess> for KeyModels {
    fn from(key: ModelAccess) -> Self {
        Self { key, team: None }
    }
}

/// The gateway's routes in the proxy's `openai_routes` preset.
const OPENAI_ROUTES: [&str; 18] = [
    "/chat/completions",
    "/v1/chat/completions",
    "/engines/{model}/chat/completions",
    "/openai/deployments/{model}/chat/completions",
    "/completions",
    "/v1/completions",
    "/engines/{model}/completions",
    "/openai/deployments/{model}/completions",
    "/embeddings",
    "/v1/embeddings",
    "/engines/{model}/embeddings",
    "/openai/deployments/{model}/embeddings",
    "/audio/transcriptions",
    "/v1/audio/transcriptions",
    "/responses",
    "/v1/responses",
    "/ocr",
    "/v1/ocr",
];

/// The gateway's routes in the proxy's `anthropic_routes` preset.
const ANTHROPIC_ROUTES: [&str; 2] = ["/v1/messages", "/v1/messages/count_tokens"];

/// A key's `allowed_routes`. Empty allows every route.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AllowedRoutes(Arc<[String]>);

impl AllowedRoutes {
    pub fn new(routes: impl Into<Arc<[String]>>) -> Self {
        Self(routes.into())
    }

    /// The proxy's `RouteChecks.is_virtual_key_allowed_to_call_route` for the routes this gateway
    /// serves. Presets other than `llm_api_routes`, `openai_routes` and `anthropic_routes` grant
    /// nothing here.
    pub fn allows(&self, route: &str) -> bool {
        self.0.is_empty()
            || self.0.iter().any(|allowed| {
                route == allowed
                    || route
                        .strip_prefix(allowed.as_str())
                        .is_some_and(|rest| rest.starts_with('/'))
                    || allowed
                        .strip_suffix('*')
                        .is_some_and(|prefix| route.starts_with(prefix))
                    || preset(allowed).any(|template| template_matches(template, route))
            })
    }

    /// The list as the proxy prints it, a Python list repr.
    pub fn python_repr(&self) -> String {
        let items: Vec<String> = self.0.iter().map(|route| python_repr(route)).collect();
        format!("[{}]", items.join(", "))
    }
}

fn preset(name: &str) -> impl Iterator<Item = &'static str> {
    let routes: &[&[&'static str]] = match name {
        "llm_api_routes" => &[&OPENAI_ROUTES, &ANTHROPIC_ROUTES],
        "openai_routes" => &[&OPENAI_ROUTES],
        "anthropic_routes" => &[&ANTHROPIC_ROUTES],
        _ => &[],
    };
    routes.iter().flat_map(|group| group.iter().copied())
}

/// `{name}` matches one non-empty segment, as in the proxy's `_route_matches_pattern`.
fn template_matches(template: &str, route: &str) -> bool {
    let template: Vec<&str> = template.split('/').collect();
    let route: Vec<&str> = route.split('/').collect();
    template.len() == route.len()
        && template.iter().zip(&route).all(|(expected, actual)| {
            if expected.starts_with('{') && expected.ends_with('}') {
                !actual.is_empty()
            } else {
                expected == actual
            }
        })
}

/// Python's `repr` of a string, for proxy messages that print one.
pub(crate) fn python_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let escaped: String = value
        .chars()
        .flat_map(|character| match character {
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            c if c == quote => vec!['\\', c],
            c => vec![c],
        })
        .collect();
    format!("{quote}{escaped}{quote}")
}

/// Why a found, unblocked key is refused whatever it asks for, with the proxy's messages.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Refusal {
    #[error(
        "{{'error': \"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call.\"}}"
    )]
    TeamMissing { team_id: String },
    #[error(
        "Authentication Error, Team={team_id} is blocked. Update via `/team/unblock` if you're an admin."
    )]
    TeamBlocked { team_id: String },
    #[error("Budget has been exceeded! {owner} Current cost: {spend}, Max budget: {max_budget}")]
    BudgetExceeded {
        /// `Key=<alias>` or `Team=<id>`.
        owner: String,
        /// Python float reprs.
        spend: String,
        max_budget: String,
    },
    /// A restriction the proxy enforces and this gateway does not yet.
    #[error(
        "Authentication Error, keys with {restriction} set on {owner} are not supported by this gateway yet."
    )]
    Unsupported {
        restriction: Restriction,
        owner: Owner,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::Display, strum::VariantArray)]
pub enum Restriction {
    #[strum(serialize = "rate limits")]
    RateLimits,
    #[strum(serialize = "guardrails or policies")]
    Guardrails,
    #[strum(serialize = "tags")]
    Tags,
    #[strum(serialize = "tool allowlists")]
    ToolAllowlists,
    #[strum(serialize = "cache control allowlists")]
    CacheControls,
    #[strum(serialize = "enforced params")]
    EnforcedParams,
    #[strum(serialize = "priority reservations")]
    Priority,
    #[strum(serialize = "max_iterations limits")]
    MaxIterations,
    #[strum(serialize = "required trace IDs")]
    RequiredTraceIds,
    #[strum(serialize = "model aliases or router settings")]
    ModelAliases,
    #[strum(serialize = "object permissions")]
    ObjectPermissions,
    #[strum(serialize = "model lists")]
    ModelLists,
    #[strum(serialize = "a project")]
    Project,
    #[strum(serialize = "an organization")]
    Organization,
    #[strum(serialize = "an agent")]
    Agent,
    #[strum(serialize = "team-based model access for personal keys")]
    PersonalKeyTeamModels,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::Display, strum::VariantArray)]
pub enum Owner {
    #[strum(serialize = "the key")]
    Key,
    #[strum(serialize = "its team")]
    Team,
    #[strum(serialize = "its team membership")]
    TeamMembership,
    #[strum(serialize = "its user")]
    User,
    #[strum(serialize = "the proxy settings")]
    Proxy,
}

/// Who a key belongs to beyond what gates it, for hooks and spend records.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyProfile {
    pub alias: Option<String>,
    pub user_id: Option<String>,
    pub team_alias: Option<String>,
    pub metadata: Arc<Map<String, Value>>,
    pub team_metadata: Arc<Map<String, Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KeyStatus {
    Active {
        expires_at: Option<SystemTime>,
        models: KeyModels,
        allowed_routes: AllowedRoutes,
        /// The key's team, whose team-scoped deployments the key may use.
        team_id: Option<String>,
        profile: Arc<KeyProfile>,
    },
    Blocked,
    Revoked,
    Refused {
        expires_at: Option<SystemTime>,
        refusal: Refusal,
    },
}

impl KeyStatus {
    /// An active key with no team and no route restriction.
    pub fn active(expires_at: Option<SystemTime>, models: ModelAccess) -> Self {
        Self::Active {
            expires_at,
            models: models.into(),
            allowed_routes: AllowedRoutes::default(),
            team_id: None,
            profile: Arc::default(),
        }
    }
}

/// A key that passed [`verify_key`]: found, unblocked, unexpired and not refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedKey {
    pub hash: KeyHash,
    pub expires_at: Option<SystemTime>,
    pub models: KeyModels,
    pub allowed_routes: AllowedRoutes,
    pub team_id: Option<String>,
    pub profile: Arc<KeyProfile>,
}

pub type LookupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<KeyStatus>, KeyError>> + Send + 'a>>;

pub type EntryFuture<'a> = Pin<Box<dyn Future<Output = Result<KeyEntry, KeyError>> + Send + 'a>>;

/// A key's status and the rows it was read from besides its own, whose changes make it stale.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyEntry {
    pub status: Option<KeyStatus>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
}

impl From<Option<KeyStatus>> for KeyEntry {
    fn from(status: Option<KeyStatus>) -> Self {
        let team_id = match &status {
            Some(KeyStatus::Active { team_id, .. }) => team_id.clone(),
            Some(KeyStatus::Refused {
                refusal: Refusal::TeamMissing { team_id } | Refusal::TeamBlocked { team_id },
                ..
            }) => Some(team_id.clone()),
            _ => None,
        };
        Self {
            status,
            team_id,
            user_id: None,
        }
    }
}

pub trait KeyLookup: Send + Sync {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a>;

    /// [`KeyLookup::lookup`] with the team and user the answer depends on. The default reads
    /// the team from the status alone; a store that knows both should say so, so that a cache
    /// can drop the answer when either changes.
    fn lookup_entry<'a>(&'a self, hash: &'a KeyHash) -> EntryFuture<'a> {
        Box::pin(async move { self.lookup(hash).await.map(KeyEntry::from) })
    }
}

pub async fn verify_key(
    lookup: &dyn KeyLookup,
    token: &SecretValue,
    now: impl FnOnce() -> SystemTime,
) -> Result<VerifiedKey, KeyError> {
    let hash = KeyHash::from_token(token);
    let expired = |expires_at: Option<SystemTime>| expires_at.is_some_and(|expiry| expiry <= now());
    match lookup.lookup(&hash).await? {
        Some(KeyStatus::Active {
            expires_at,
            models,
            allowed_routes,
            team_id,
            profile,
        }) => {
            if expired(expires_at) {
                return Err(KeyError::Expired);
            }
            Ok(VerifiedKey {
                hash,
                expires_at,
                models,
                allowed_routes,
                team_id,
                profile,
            })
        }
        Some(KeyStatus::Refused {
            expires_at,
            refusal,
        }) => Err(if expired(expires_at) {
            KeyError::Expired
        } else {
            KeyError::Refused(refusal)
        }),
        Some(KeyStatus::Blocked) => Err(KeyError::Blocked),
        Some(KeyStatus::Revoked) | None => Err(KeyError::Invalid),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn access(models: &[&str]) -> ModelAccess {
        ModelAccess::new(
            models
                .iter()
                .map(|model| (*model).to_owned())
                .collect::<Vec<_>>(),
        )
    }

    #[rstest]
    #[case::empty_list_grants_all(&[], "gpt-a", true)]
    #[case::exact(&["gpt-a", "gpt-b"], "gpt-b", true)]
    #[case::not_listed(&["gpt-a"], "gpt-b", false)]
    #[case::not_a_prefix_match(&["gpt-a"], "gpt-a-mini", false)]
    #[case::star(&["*"], "anything", true)]
    #[case::all_proxy_models(&["all-proxy-models"], "anything", true)]
    #[case::suffix_wildcard(&["claude-*"], "claude-opus-5-5-0", true)]
    #[case::suffix_wildcard_other_family(&["claude-*"], "gpt-a", false)]
    #[case::provider_wildcard(&["bedrock/*"], "bedrock/us.model-v1:0", true)]
    #[case::provider_wildcard_needs_separator(&["bedrock/*"], "bedrockz/us.model", false)]
    #[case::inner_wildcard(&["gpt-*-mini"], "gpt-6-mini", true)]
    #[case::inner_wildcard_suffix_mismatch(&["gpt-*-mini"], "gpt-6-large", false)]
    fn model_access(#[case] models: &[&str], #[case] model: &str, #[case] allowed: bool) {
        assert_eq!(access(models).allows(model), allowed);
    }

    #[rstest]
    #[case::no_team(&["gpt-a"], None, "gpt-a", Ok(()))]
    #[case::key_checked_first(&["gpt-a"], Some(&["gpt-b"][..]), "gpt-b", Err(ModelDenial::Key))]
    #[case::team_narrows_all_models_key(&[], Some(&["gpt-a"][..]), "gpt-b", Err(ModelDenial::Team))]
    #[case::both_allow(&["gpt-*"], Some(&["gpt-a"][..]), "gpt-a", Ok(()))]
    #[case::empty_team_list_allows_all(&["gpt-a"], Some(&[][..]), "gpt-a", Ok(()))]
    #[case::team_star(&[], Some(&["*"][..]), "anything", Ok(()))]
    #[case::team_all_proxy_models(&[], Some(&["all-proxy-models"][..]), "anything", Ok(()))]
    #[case::team_wildcard_other_family(&[], Some(&["claude-*"][..]), "gpt-a", Err(ModelDenial::Team))]
    fn team_models_narrow_key_models(
        #[case] key: &[&str],
        #[case] team: Option<&[&str]>,
        #[case] model: &str,
        #[case] expected: Result<(), ModelDenial>,
    ) {
        let models = KeyModels {
            key: access(key),
            team: team.map(access),
        };
        assert_eq!(models.check(model), expected);
    }

    fn routes(routes: &[&str]) -> AllowedRoutes {
        AllowedRoutes::new(
            routes
                .iter()
                .map(|route| (*route).to_owned())
                .collect::<Vec<_>>(),
        )
    }

    #[rstest]
    #[case::empty_allows_all(&[], "/v1/messages", true)]
    #[case::exact(&["/v1/messages"], "/v1/messages", true)]
    #[case::exact_other(&["/v1/messages"], "/v1/chat/completions", false)]
    #[case::prefix(&["/v1"], "/v1/messages", true)]
    #[case::prefix_needs_separator(&["/v1/mess"], "/v1/messages", false)]
    #[case::wildcard(&["/v1/*"], "/v1/chat/completions", true)]
    #[case::wildcard_other(&["/v2/*"], "/v1/chat/completions", false)]
    #[case::llm_api_routes_messages(&["llm_api_routes"], "/v1/messages", true)]
    #[case::llm_api_routes_chat(&["llm_api_routes"], "/chat/completions", true)]
    #[case::llm_api_routes_model_path(
        &["llm_api_routes"],
        "/openai/deployments/{model}/chat/completions",
        true
    )]
    #[case::llm_api_routes_not_mcp(&["llm_api_routes"], "/mcp/sse", false)]
    #[case::openai_routes_chat(&["openai_routes"], "/v1/chat/completions", true)]
    #[case::openai_routes_responses(&["openai_routes"], "/v1/responses", true)]
    #[case::openai_routes_not_messages(&["openai_routes"], "/v1/messages", false)]
    #[case::anthropic_routes_messages(&["anthropic_routes"], "/v1/messages", true)]
    #[case::anthropic_routes_not_chat(&["anthropic_routes"], "/chat/completions", false)]
    #[case::template_needs_a_segment(&["openai_routes"], "/engines//chat/completions", false)]
    #[case::template_extra_segment(&["openai_routes"], "/engines/a/b/chat/completions", false)]
    #[case::other_presets_grant_nothing(&["info_routes", "management_routes"], "/v1/messages", false)]
    #[case::any_entry(&["/key/info", "anthropic_routes"], "/v1/messages", true)]
    fn allowed_routes(#[case] allowed: &[&str], #[case] route: &str, #[case] expected: bool) {
        assert_eq!(routes(allowed).allows(route), expected);
    }

    #[rstest]
    #[case::plain(&["llm_api_routes", "/v1/messages"], "['llm_api_routes', '/v1/messages']")]
    #[case::empty(&[], "[]")]
    #[case::single_quote(&["it's"], r#"["it's"]"#)]
    #[case::both_quotes(&[r#"a'b"c"#], r#"['a\'b"c']"#)]
    #[case::backslash(&[r"a\b"], r"['a\\b']")]
    fn prints_routes_like_python(#[case] allowed: &[&str], #[case] expected: &str) {
        assert_eq!(routes(allowed).python_repr(), expected);
    }
}
