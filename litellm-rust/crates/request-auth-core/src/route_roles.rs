use serde::{Deserialize, Serialize};

use crate::{Error, patterns};

#[derive(
    Clone, Copy, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
enum Role {
    #[strum(serialize = "proxy_admin")]
    Admin,
    #[strum(serialize = "proxy_admin_viewer")]
    Viewer,
    #[strum(serialize = "org_admin")]
    Organization,
    #[strum(serialize = "internal_user")]
    Internal,
    #[strum(serialize = "internal_user_viewer")]
    InternalViewer,
    #[strum(serialize = "team")]
    Team,
    #[strum(serialize = "customer")]
    Customer,
}

#[derive(Deserialize)]
pub struct RoleFacts {
    pub user_present: bool,
    pub role: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct RoleDecision {
    pub effective: &'static str,
    pub user_role: Option<&'static str>,
    pub admin: bool,
}

pub fn role(facts: &RoleFacts) -> RoleDecision {
    let parsed = facts
        .role
        .as_deref()
        .and_then(|value| value.parse::<Role>().ok());
    let effective = parsed.unwrap_or(Role::Internal).into();
    RoleDecision {
        effective,
        user_role: facts.user_present.then_some(effective),
        admin: facts.user_present && parsed == Some(Role::Admin),
    }
}

#[derive(Deserialize)]
pub struct ApiFacts {
    pub token_present: bool,
    pub admin_read: bool,
    pub user: RoleFacts,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum ApiDecision {
    InvalidToken,
    ReadAdmin,
    NonAdmin,
    Allow,
}

pub fn api(facts: &ApiFacts) -> ApiDecision {
    if !facts.token_present {
        ApiDecision::InvalidToken
    } else if !facts.admin_read {
        ApiDecision::ReadAdmin
    } else if role(&facts.user).admin {
        ApiDecision::Allow
    } else {
        ApiDecision::NonAdmin
    }
}

#[derive(Deserialize)]
pub struct JwtFacts {
    pub role: Option<String>,
    pub config_read: bool,
    pub config_missing: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum JwtDecision {
    ReadAdmin,
    ReadTeam,
    DefaultTeam,
    Check,
    Deny,
}

pub fn jwt(facts: &JwtFacts) -> JwtDecision {
    match facts.role.as_deref() {
        Some("proxy_admin") if !facts.config_read => JwtDecision::ReadAdmin,
        Some("team") if !facts.config_read => JwtDecision::ReadTeam,
        Some("team") if facts.config_missing => JwtDecision::DefaultTeam,
        Some("proxy_admin" | "team") => JwtDecision::Check,
        _ => JwtDecision::Deny,
    }
}

#[derive(Deserialize)]
pub struct InsideFacts {
    pub role: Option<String>,
    pub user_id: Option<String>,
    pub requested_user_id: Option<String>,
}

pub fn inside(facts: &InsideFacts) -> bool {
    matches!(
        facts.role.as_deref(),
        Some("proxy_admin" | "proxy_admin_viewer")
    ) || facts.user_id.is_some() && facts.user_id == facts.requested_user_id
}

#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    pub text: Option<String>,
    pub python_type: String,
    pub hash_failure: Option<String>,
    pub templates: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct ListedFacts {
    pub route: String,
    pub entry: Option<Entry>,
    pub exhausted: bool,
    pub template_index: usize,
    pub compiled: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Action {
    ReadEntry,
    ReadTemplate,
    Allow,
    Deny,
    Keep,
    Expand,
    Failure,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum FailureKind {
    TypeError,
    AttributeError,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Failure {
    pub kind: FailureKind,
    pub message: String,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct EntryDecision {
    pub action: Action,
    pub template_index: usize,
    pub failure: Option<Failure>,
}

impl EntryDecision {
    fn action(action: Action) -> Self {
        Self {
            action,
            template_index: 0,
            failure: None,
        }
    }

    fn failure(kind: FailureKind, message: String) -> Self {
        Self {
            action: Action::Failure,
            template_index: 0,
            failure: Some(Failure { kind, message }),
        }
    }
}

pub fn expand(entry: &Entry) -> EntryDecision {
    if let Some(message) = &entry.hash_failure {
        EntryDecision::failure(FailureKind::TypeError, message.clone())
    } else if entry.templates.is_some() {
        EntryDecision::action(Action::Expand)
    } else {
        EntryDecision::action(Action::Keep)
    }
}

pub fn listed(facts: &ListedFacts) -> Result<EntryDecision, Error> {
    if facts.exhausted {
        return Ok(EntryDecision::action(Action::Deny));
    }
    let Some(entry) = &facts.entry else {
        return Ok(EntryDecision::action(Action::ReadEntry));
    };
    if let Some(message) = &entry.hash_failure {
        return Ok(EntryDecision::failure(
            FailureKind::TypeError,
            message.clone(),
        ));
    }
    let Some(text) = &entry.text else {
        return Ok(EntryDecision::failure(
            FailureKind::AttributeError,
            format!("'{}' object has no attribute 'endswith'", entry.python_type),
        ));
    };
    let Some(templates) = &entry.templates else {
        return Ok(EntryDecision::action(
            if patterns::wildcard(&facts.route, text) {
                Action::Allow
            } else {
                Action::ReadEntry
            },
        ));
    };
    let next = if let Some(compiled) = &facts.compiled {
        let expression = compiled
            .strip_suffix('$')
            .map_or_else(|| compiled.clone(), |prefix| format!("{prefix}(?=\\n?\\z)"));
        if patterns::regex_match(&expression, &facts.route)? {
            return Ok(EntryDecision::action(Action::Allow));
        }
        facts.template_index + 1
    } else {
        facts.template_index
    };
    Ok(if next < templates.len() {
        EntryDecision {
            action: Action::ReadTemplate,
            template_index: next,
            failure: None,
        }
    } else {
        EntryDecision::action(Action::ReadEntry)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::absent(false, None, None, false)]
    #[case::default(true, None, Some("internal_user"), false)]
    #[case::unknown(true, Some("ADMIN"), Some("internal_user"), false)]
    #[case::admin(true, Some("proxy_admin"), Some("proxy_admin"), true)]
    #[case::viewer(true, Some("proxy_admin_viewer"), Some("proxy_admin_viewer"), false)]
    fn roles(
        #[case] present: bool,
        #[case] raw: Option<&str>,
        #[case] expected: Option<&str>,
        #[case] admin: bool,
    ) {
        let result = role(&RoleFacts {
            user_present: present,
            role: raw.map(str::to_owned),
        });
        assert_eq!(result.user_role, expected);
        assert_eq!(result.admin, admin);
    }

    #[rstest]
    #[case::invalid(false, false, ApiDecision::InvalidToken)]
    #[case::read(true, false, ApiDecision::ReadAdmin)]
    #[case::nonadmin(true, true, ApiDecision::NonAdmin)]
    fn api_order(#[case] token: bool, #[case] read: bool, #[case] expected: ApiDecision) {
        assert_eq!(
            api(&ApiFacts {
                token_present: token,
                admin_read: read,
                user: RoleFacts {
                    user_present: false,
                    role: None
                }
            }),
            expected
        );
    }

    #[rstest]
    #[case::read("team", false, false, JwtDecision::ReadTeam)]
    #[case::default("team", true, true, JwtDecision::DefaultTeam)]
    #[case::empty("team", true, false, JwtDecision::Check)]
    #[case::other("internal_user", false, false, JwtDecision::Deny)]
    fn jwt_selection(
        #[case] role: &str,
        #[case] read: bool,
        #[case] missing: bool,
        #[case] expected: JwtDecision,
    ) {
        assert_eq!(
            jwt(&JwtFacts {
                role: Some(role.into()),
                config_read: read,
                config_missing: missing
            }),
            expected
        );
    }

    #[rstest]
    #[case::self_empty(Some(""), Some(""), true)]
    #[case::absent(None, None, false)]
    #[case::different(Some("a"), Some("b"), false)]
    fn inside_identity(
        #[case] user: Option<&str>,
        #[case] requested: Option<&str>,
        #[case] expected: bool,
    ) {
        assert_eq!(
            inside(&InsideFacts {
                role: None,
                user_id: user.map(str::to_owned),
                requested_user_id: requested.map(str::to_owned)
            }),
            expected
        );
    }
}
