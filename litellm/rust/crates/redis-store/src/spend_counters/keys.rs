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
    PersonWindow {
        person_id: String,
        window: String,
    },
    EndUser(String),
    Org(String),
    Tag(String),
    Project(String),
    ModelAccessGroup(String),
    ShadowEval(String),
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
            Self::PersonWindow { person_id, window } => {
                format!("spend:person:{person_id}:window:{window}")
            }
            Self::EndUser(id) => format!("spend:end_user:{id}"),
            Self::Org(id) => format!("spend:org:{id}"),
            Self::Tag(name) => format!("spend:tag:{name}"),
            Self::Project(id) => format!("spend:project:{id}"),
            Self::ModelAccessGroup(name) => format!("spend:model_access_group:{name}"),
            Self::ShadowEval(job_id) => format!("spend:shadow_eval:{job_id}"),
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
            "person" => {
                let (person_id, window) = windowed();
                Self::PersonWindow {
                    person_id,
                    window: window.ok_or_else(invalid)?,
                }
            }
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
    #[case::monthly_user("spend:person:user:person%3Aone:window:1mo@2026-11-01T05:30:00+00:00")]
    #[case::annual_customer("spend:person:customer_member:person%2Ftwo:window:12mo@2026-01-01T08:00:00+00:00")]
    fn personal_windows_round_trip_without_a_committed_twin(#[case] key: &str) {
        let counter = SpendCounter::parse(key).expect("valid personal window");
        assert!(matches!(counter, SpendCounter::PersonWindow { .. }));
        assert_eq!(counter.key(), key);
        assert!(SpendCounter::parse(&key.replacen("spend:", "spend:committed:", 1)).is_err());
    }
}
