#[derive(
    Clone, Debug, PartialEq, Eq, Hash, strum::EnumString, strum::Display, strum::VariantArray,
)]
pub enum CustomerScope {
    #[strum(serialize = "customer_org")]
    Organization,
    #[strum(serialize = "customer_team")]
    Team,
    #[strum(serialize = "customer_member")]
    Member,
}

/// One budget's spend counter. `parse` and `key` round-trip `main`'s `spend:*` keys exactly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpendCounter {
    /// The charges-only twin of a key, team, user, org or end user lifetime counter, which budget
    /// holds never touch: `spend:committed:` and the twin's key past its `spend:`.
    Committed(Box<SpendCounter>),
    Key {
        token: String,
        window: Option<String>,
    },
    Team {
        team_id: String,
        window: Option<String>,
    },
    TeamMember {
        user_id: String,
        team_id: String,
    },
    CustomerWindow {
        scope: CustomerScope,
        entity_id: String,
        window: String,
    },
    User(String),
    EndUser(String),
    Org(String),
    Tag(String),
    Project(String),
    ModelAccessGroup(String),
    ShadowEval(String),
    CustomerOrg(String),
    CustomerTeam(String),
    CustomerMember(String),
}

#[derive(Debug, thiserror::Error)]
#[error("{0:?} is not a spend counter")]
pub struct NotASpendCounter(pub String);

impl SpendCounter {
    pub fn key(&self) -> String {
        let windowed = |scope: &str, id: &str, window: &Option<String>| match window {
            Some(window) => format!("spend:{scope}:{id}:window:{window}"),
            None => format!("spend:{scope}:{id}"),
        };
        match self {
            Self::Committed(counter) => {
                format!("spend:committed:{}", &counter.key()["spend:".len()..])
            }
            Self::Key { token, window } => windowed("key", token, window),
            Self::Team { team_id, window } => windowed("team", team_id, window),
            Self::TeamMember { user_id, team_id } => {
                format!("spend:team_member:{user_id}:{team_id}")
            }
            Self::CustomerWindow {
                scope,
                entity_id,
                window,
            } => {
                format!("spend:{scope}:{entity_id}:window:{window}")
            }
            Self::User(id) => format!("spend:user:{id}"),
            Self::EndUser(id) => format!("spend:end_user:{id}"),
            Self::Org(id) => format!("spend:org:{id}"),
            Self::Tag(name) => format!("spend:tag:{name}"),
            Self::Project(id) => format!("spend:project:{id}"),
            Self::ModelAccessGroup(name) => format!("spend:model_access_group:{name}"),
            Self::ShadowEval(job_id) => format!("spend:shadow_eval:{job_id}"),
            Self::CustomerOrg(id) => format!("spend:customer_org:{id}"),
            Self::CustomerTeam(id) => format!("spend:customer_team:{id}"),
            Self::CustomerMember(id) => format!("spend:customer_member:{id}"),
        }
    }

    pub fn parse(key: &str) -> Result<Self, NotASpendCounter> {
        let invalid = || NotASpendCounter(key.to_owned());
        let rest = key.strip_prefix("spend:").ok_or_else(invalid)?;
        if let Some(twin) = rest.strip_prefix("committed:") {
            return match Self::parse(&format!("spend:{twin}")).map_err(|_| invalid())? {
                counter @ (Self::Key { window: None, .. }
                | Self::Team { window: None, .. }
                | Self::User(_)
                | Self::Org(_)
                | Self::EndUser(_)) => Ok(Self::Committed(Box::new(counter))),
                _ => Err(invalid()),
            };
        }
        let (scope, id) = rest.split_once(':').ok_or_else(invalid)?;
        if let Ok(scope) = scope.parse::<CustomerScope>() {
            return match id.rsplit_once(":window:") {
                Some((entity_id, window)) if !entity_id.is_empty() && !window.is_empty() => {
                    Ok(Self::CustomerWindow {
                        scope,
                        entity_id: entity_id.to_owned(),
                        window: window.to_owned(),
                    })
                }
                Some(_) => Err(invalid()),
                None => Ok(match scope {
                    CustomerScope::Organization => Self::CustomerOrg(id.to_owned()),
                    CustomerScope::Team => Self::CustomerTeam(id.to_owned()),
                    CustomerScope::Member => Self::CustomerMember(id.to_owned()),
                }),
            };
        }
        let windowed = || match id.rsplit_once(":window:") {
            Some((id, window)) if !window.is_empty() => (id.to_owned(), Some(window.to_owned())),
            _ => (id.to_owned(), None),
        };
        Ok(match scope {
            "key" => {
                let (token, window) = windowed();
                Self::Key { token, window }
            }
            "team" => {
                let (team_id, window) = windowed();
                Self::Team { team_id, window }
            }
            "team_member" => {
                let (user_id, team_id) = id.rsplit_once(':').ok_or_else(invalid)?;
                Self::TeamMember {
                    user_id: user_id.to_owned(),
                    team_id: team_id.to_owned(),
                }
            }
            "user" => Self::User(id.to_owned()),
            "end_user" => Self::EndUser(id.to_owned()),
            "org" => Self::Org(id.to_owned()),
            "tag" => Self::Tag(id.to_owned()),
            "project" => Self::Project(id.to_owned()),
            "model_access_group" => Self::ModelAccessGroup(id.to_owned()),
            "shadow_eval" => Self::ShadowEval(id.to_owned()),
            _ => return Err(invalid()),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::SpendCounter;
    use rstest::rstest;

    #[rstest]
    #[case::member("spend:customer_member:member%3A1:window:1mo@2026-11-01T05:30:00+00:00")]
    #[case::team("spend:customer_team:team:window:12mo@2026-01-01T05:30:00+00:00")]
    #[case::organization("spend:customer_org:org:window:1mo@2026-11-01T00:00:00+00:00")]
    #[case::lifetime_organization("spend:customer_org:tenant:lifetime")]
    #[case::period_team("spend:customer_team:tenant:team:2026-01-01T00:00:00+00:00")]
    #[case::lifetime_member("spend:customer_member:user@example.com:lifetime")]
    fn customer_counter_keys_round_trip(#[case] key: &str) {
        assert_eq!(
            SpendCounter::parse(key).expect("customer counter").key(),
            key
        );
    }

    #[rstest]
    #[case::no_identity("spend:customer_team::window:1mo")]
    #[case::empty_window("spend:customer_org:org:window:")]
    #[case::committed("spend:committed:customer_member:member:window:1mo")]
    fn rejects_invalid_customer_window(#[case] key: &str) {
        assert!(SpendCounter::parse(key).is_err());
    }
}
