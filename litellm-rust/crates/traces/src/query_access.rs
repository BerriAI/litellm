use serde::{Deserialize, Serialize};

use crate::InvalidScope;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryScope {
    Admin,
    Team {
        team_id: String,
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
            _ => Err(InvalidScope),
        }
    }
}
