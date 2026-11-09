/// One budget's spend counter. `parse` and `key` round-trip `main`'s `spend:*` keys exactly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpendCounter {
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
        }
    }

    pub fn parse(key: &str) -> Result<Self, NotASpendCounter> {
        let invalid = || NotASpendCounter(key.to_owned());
        let rest = key.strip_prefix("spend:").ok_or_else(invalid)?;
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
            _ => return Err(invalid()),
        })
    }
}
