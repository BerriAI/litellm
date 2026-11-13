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
            "customer_org" => Self::CustomerOrg(id.to_owned()),
            "customer_team" => Self::CustomerTeam(id.to_owned()),
            "customer_member" => Self::CustomerMember(id.to_owned()),
            _ => return Err(invalid()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SpendCounter;
    use rstest::rstest;

    #[rstest]
    #[case::organization("spend:customer_org:tenant:lifetime")]
    #[case::team("spend:customer_team:tenant:team:2026-01-01T00:00:00+00:00")]
    #[case::member("spend:customer_member:user@example.com:lifetime")]
    fn customer_counter_keys_round_trip(#[case] key: &str) {
        assert_eq!(SpendCounter::parse(key).unwrap().key(), key);
    }
}
