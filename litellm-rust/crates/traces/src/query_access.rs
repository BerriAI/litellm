use serde::{Deserialize, Serialize};

use crate::InvalidScope;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryScope {
    Admin,
    Team {
        team_id: String,
    },
    Logs {
        user_id: String,
        team_ids: Vec<String>,
        api_key_hash: String,
    },
    Key {
        team_id: String,
        api_key_hash: String,
    },
}

impl QueryScope {
    pub fn validate(&self) -> Result<(), InvalidScope> {
        match self {
            Self::Admin => Ok(()),
            Self::Team { team_id } if !team_id.is_empty() => Ok(()),
            Self::Key { api_key_hash, .. } if !api_key_hash.is_empty() => Ok(()),
            Self::Logs {
                user_id,
                team_ids,
                api_key_hash,
            } if (!user_id.is_empty() || !team_ids.is_empty() || !api_key_hash.is_empty())
                && team_ids.iter().all(|team| !team.is_empty()) =>
            {
                Ok(())
            }
            _ => Err(InvalidScope),
        }
    }
}
