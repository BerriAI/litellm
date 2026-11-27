//! The team repository: one team by id, or the teams sharing an alias, each with its model aliases, read when the auth
//! cache misses, and the insert `team_id_upsert` makes for a team no row has.

use jiff::Timestamp;
use litellm_db::Call;

use super::queries::{self, Team};
use crate::{Database, Inserted, Side, database::Connections};

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

const CREATE_TEAM: Call = Call {
    call_type: "create",
    table: Some("LiteLLM_TeamTable"),
    operation: Some("insert"),
};

/// A team to create under `team_id_upsert`, with what `new_team` takes from `default_team_params` and
/// `default_team_settings` for a request that names only the id. The host resolves them, the reset time of
/// `budget_duration` included. `None` leaves the column at what `new_team` stores for it.
#[derive(Clone, Debug, Default)]
pub struct NewTeam {
    pub team_id: String,
    pub organization_id: Option<String>,
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub budget_reset_at: Option<Timestamp>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub team_member_permissions: Option<Vec<String>>,
    pub access_group_ids: Option<Vec<String>>,
}

/// What a team insert stored: the team, or the one a concurrent writer stored first, and the access groups whose
/// teams the insert reconciled, whose caches the host drops once it returns.
#[derive(Clone, Debug, PartialEq)]
pub struct InsertedTeam {
    pub team: Inserted<Team>,
    pub access_groups: Vec<String>,
}

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

    /// Creates `team` as `new_team` does for a proxy admin under `team_id_upsert`, in one transaction: the row,
    /// then, under the team's advisory lock, each access group's `assigned_team_ids` reconciled with the team's
    /// `access_group_ids`, as `reconcile_team_access_group_membership` does. A team id a concurrent writer stored
    /// first gets that team, and nothing else is written.
    pub async fn insert(&self, team: &NewTeam) -> Result<InsertedTeam, litellm_db::Error> {
        self.database
            .run(CREATE_TEAM, || async {
                self.database
                    .transaction(async |transaction| insert_once(transaction, team).await)
                    .await
            })
            .await
    }
}

/// The steps of a team insert, in the order `new_team` makes them.
pub(super) async fn insert_once(
    connections: &mut impl Connections,
    team: &NewTeam,
) -> Result<InsertedTeam, litellm_db::Error> {
    let created = connections
        .with(Side::Writer, async |db| {
            queries::insert_team(db, team).await
        })
        .await?;
    if created.is_none() {
        let stored = connections
            .with(Side::Writer, async |db| {
                queries::find_team(db, &team.team_id).await
            })
            .await?;
        return Ok(InsertedTeam {
            team: Inserted::Conflict(stored),
            access_groups: Vec::new(),
        });
    }
    let access_groups = reconcile_access_groups(connections, &team.team_id).await?;
    let stored = connections
        .with(Side::Writer, async |db| {
            queries::find_team(db, &team.team_id).await
        })
        .await?
        .ok_or(sqlx::Error::RowNotFound)?;
    Ok(InsertedTeam {
        team: Inserted::Created(stored),
        access_groups,
    })
}

/// Mirrors the team's `access_group_ids` into each access group's `assigned_team_ids`, and returns every group on
/// either side of it.
async fn reconcile_access_groups(
    connections: &mut impl Connections,
    team_id: &str,
) -> Result<Vec<String>, litellm_db::Error> {
    connections
        .with(Side::Writer, async |db| {
            queries::lock_team(db, team_id).await
        })
        .await?;
    let wanted = connections
        .with(Side::Writer, async |db| {
            queries::lock_team_access_group_ids(db, team_id).await
        })
        .await?
        .unwrap_or_default();
    let affected = connections
        .with(Side::Writer, async |db| {
            queries::access_groups_of_team(db, team_id, &wanted).await
        })
        .await?;
    let additions: Vec<String> = affected
        .iter()
        .filter(|group| !group.attached)
        .map(|group| group.access_group_id.clone())
        .collect();
    let removals: Vec<String> = affected
        .iter()
        .filter(|group| !wanted.contains(&group.access_group_id))
        .map(|group| group.access_group_id.clone())
        .collect();
    let changed: Vec<String> = additions.iter().chain(&removals).cloned().collect();
    if !changed.is_empty() {
        connections
            .with(Side::Writer, async |db| {
                queries::lock_access_groups(db, &changed).await
            })
            .await?;
    }
    if !additions.is_empty() {
        connections
            .with(Side::Writer, async |db| {
                queries::attach_team_to_access_groups(db, team_id, &additions).await
            })
            .await?;
    }
    if !removals.is_empty() {
        connections
            .with(Side::Writer, async |db| {
                queries::detach_team_from_access_groups(db, team_id, &removals).await
            })
            .await?;
    }
    Ok(affected
        .into_iter()
        .map(|group| group.access_group_id)
        .collect())
}
