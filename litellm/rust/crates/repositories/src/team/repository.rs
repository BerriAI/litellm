//! The team repository: one team by id, or the teams sharing an alias, each with its model aliases, read when the auth
//! cache misses.

use litellm_db::Call;

use super::queries::{self, Team};
use crate::{Database, Side};

const FIND_TEAM: Call = Call {
    call_type: "_get_team_db_check",
    table: None,
    operation: None,
};

const FIND_TEAMS_BY_ALIAS: Call = Call {
    call_type: "get_team_object_by_alias",
    table: None,
    operation: None,
};

/// Reads `LiteLLM_TeamTable` joined to the team's `LiteLLM_ModelTable` aliases.
pub struct TeamRepository {
    database: Database,
}

impl TeamRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The team `team_id`, or `None` when no row has that id.
    pub async fn find_with_model_aliases(
        &self,
        team_id: &str,
        side: Side,
    ) -> Result<Option<Team>, litellm_db::Error> {
        self.database
            .run(FIND_TEAM, || async {
                let mut session = self.database.session(side).await?;
                queries::find_team(session.executor(), team_id).await
            })
            .await
    }

    /// Up to two teams named `alias`, so the caller can refuse an alias two teams share.
    pub async fn find_up_to_two_by_alias(
        &self,
        alias: &str,
    ) -> Result<Vec<Team>, litellm_db::Error> {
        self.database
            .run(FIND_TEAMS_BY_ALIAS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_teams_by_alias(session.executor(), alias).await
            })
            .await
    }
}
