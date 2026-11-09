use crate::Error;
use sqlx::{Executor, Postgres};

#[derive(Clone, PartialEq, serde::Serialize, sqlx::Type, veil::Redact)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct AuthData(#[redact(with = "[REDACTED]")] pub serde_json::Value);

#[macro_rules_attribute::apply(crate::db_row)]
pub struct AuthRow {
    pub data: AuthData,
}

#[derive(Clone, Copy, Debug, strum::EnumString, strum::VariantArray)]
pub enum AuthRead {
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "team")]
    Team,
    #[strum(serialize = "org")]
    Org,
    #[strum(serialize = "project")]
    Project,
    #[strum(serialize = "end_user")]
    EndUser,
    #[strum(serialize = "budget")]
    Budget,
    #[strum(serialize = "object_permission")]
    ObjectPermission,
    #[strum(serialize = "access_group")]
    AccessGroup,
    #[strum(serialize = "tag")]
    Tag,
    #[strum(serialize = "model_access_group")]
    ModelAccessGroup,
    #[strum(serialize = "agent")]
    Agent,
    #[strum(serialize = "retired_agent")]
    RetiredAgent,
    #[strum(serialize = "key")]
    Key,
    #[strum(serialize = "member")]
    Member,
    #[strum(serialize = "agent_identity")]
    AgentIdentity,
    #[strum(serialize = "retired_agent_identity")]
    RetiredAgentIdentity,
    #[strum(serialize = "verified_subject")]
    VerifiedSubject,
    #[strum(serialize = "team_alias")]
    TeamAlias,
    #[strum(serialize = "org_alias")]
    OrgAlias,
    #[strum(serialize = "user_sso")]
    UserSso,
    #[strum(serialize = "user_email")]
    UserEmail,
    #[strum(serialize = "tag_registry")]
    TagRegistry,
    #[strum(serialize = "model_access_group_registry")]
    ModelAccessGroupRegistry,
    #[strum(serialize = "end_user_restricted_registry")]
    EndUserRestrictedRegistry,
    #[strum(serialize = "prefetch")]
    Prefetch,
    #[strum(serialize = "budget_window")]
    BudgetWindow,
    #[strum(serialize = "window_spend")]
    WindowSpend,
}

#[derive(Clone, Copy, Debug, strum::EnumString, strum::VariantArray)]
pub enum AuthSpend {
    #[strum(serialize = "key")]
    Key,
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "team")]
    Team,
    #[strum(serialize = "org")]
    Org,
    #[strum(serialize = "project")]
    Project,
    #[strum(serialize = "end_user")]
    EndUser,
    #[strum(serialize = "tag")]
    Tag,
    #[strum(serialize = "model_access_group")]
    ModelAccessGroup,
    #[strum(serialize = "member")]
    Member,
}

fn exact(ids: &[String], count: usize) -> Result<(), Error> {
    if ids.len() == count {
        Ok(())
    } else {
        Err(Error::InvalidAuthArguments)
    }
}

fn hashed_key(id: &str) -> Result<(), Error> {
    if id.len() == 64 && id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(Error::InvalidAuthArguments)
    }
}

fn entity(value: &str) -> Result<&'static str, Error> {
    match value {
        "Key" => Ok("key"),
        "Team" => Ok("team"),
        _ => Err(Error::InvalidAuthArguments),
    }
}

pub async fn auth_read<'e>(
    executor: impl Executor<'e, Database = Postgres>,
    kind: AuthRead,
    ids: &[String],
) -> Result<Vec<AuthRow>, Error> {
    if ids.len() > 1000 || ids.iter().any(|id| id.len() > 65536) {
        return Err(Error::InvalidAuthArguments);
    }
    let rows = match kind {
        AuthRead::User => {
            sqlx::query_file_as!(AuthRow, "queries/auth/user.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Team => {
            sqlx::query_file_as!(AuthRow, "queries/auth/team.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Org => {
            sqlx::query_file_as!(AuthRow, "queries/auth/org.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Project => {
            sqlx::query_file_as!(AuthRow, "queries/auth/project.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::EndUser => {
            sqlx::query_file_as!(AuthRow, "queries/auth/end_user.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Budget => {
            sqlx::query_file_as!(AuthRow, "queries/auth/budget.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::ObjectPermission => {
            sqlx::query_file_as!(AuthRow, "queries/auth/object_permission.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::AccessGroup => {
            sqlx::query_file_as!(AuthRow, "queries/auth/access_group.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Tag => {
            sqlx::query_file_as!(AuthRow, "queries/auth/tag.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::ModelAccessGroup => {
            sqlx::query_file_as!(AuthRow, "queries/auth/model_access_group.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Agent => {
            sqlx::query_file_as!(AuthRow, "queries/auth/agent.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::RetiredAgent => {
            sqlx::query_file_as!(AuthRow, "queries/auth/retired_agent.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Key => {
            ids.iter().try_for_each(|id| hashed_key(id))?;
            sqlx::query_file_as!(AuthRow, "queries/auth/key.sql", ids)
                .fetch_all(executor)
                .await?
        }
        AuthRead::Member => {
            exact(ids, 2)?;
            sqlx::query_file_as!(AuthRow, "queries/auth/member.sql", &ids[0], &ids[1])
                .fetch_all(executor)
                .await?
        }
        AuthRead::AgentIdentity => {
            exact(ids, 3)?;
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/agent_identity.sql",
                &ids[0],
                &ids[1],
                &ids[2]
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::RetiredAgentIdentity => {
            exact(ids, 3)?;
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/retired_agent_identity.sql",
                &ids[0],
                &ids[1],
                &ids[2]
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::VerifiedSubject => {
            exact(ids, 3)?;
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/verified_subject.sql",
                &ids[0],
                &ids[1],
                &ids[2]
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::TeamAlias => {
            exact(ids, 1)?;
            sqlx::query_file_as!(AuthRow, "queries/auth/team_alias.sql", &ids[0])
                .fetch_all(executor)
                .await?
        }
        AuthRead::OrgAlias => {
            exact(ids, 1)?;
            sqlx::query_file_as!(AuthRow, "queries/auth/org_alias.sql", &ids[0])
                .fetch_all(executor)
                .await?
        }
        AuthRead::UserSso => {
            exact(ids, 1)?;
            sqlx::query_file_as!(AuthRow, "queries/auth/user_sso.sql", &ids[0])
                .fetch_all(executor)
                .await?
        }
        AuthRead::UserEmail => {
            exact(ids, 1)?;
            sqlx::query_file_as!(AuthRow, "queries/auth/user_email.sql", &ids[0])
                .fetch_all(executor)
                .await?
        }
        AuthRead::TagRegistry => {
            exact(ids, 1)?;
            let limit: i64 = ids[0]
                .parse()
                .ok()
                .filter(|value| (1..=100_001).contains(value))
                .ok_or(Error::InvalidAuthArguments)?;
            sqlx::query_file_as!(AuthRow, "queries/auth/tag_registry.sql", limit)
                .fetch_all(executor)
                .await?
        }
        AuthRead::ModelAccessGroupRegistry => {
            exact(ids, 1)?;
            let limit: i64 = ids[0]
                .parse()
                .ok()
                .filter(|value| (1..=100_001).contains(value))
                .ok_or(Error::InvalidAuthArguments)?;
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/model_access_group_registry.sql",
                limit
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::EndUserRestrictedRegistry => {
            exact(ids, 1)?;
            let limit: i64 = ids[0]
                .parse()
                .ok()
                .filter(|value| (1..=100_001).contains(value))
                .ok_or(Error::InvalidAuthArguments)?;
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/end_user_restricted_registry.sql",
                limit
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::Prefetch => {
            exact(ids, 1)?;
            let refs: [Option<String>; 5] =
                serde_json::from_str(&ids[0]).map_err(|_| Error::InvalidAuthArguments)?;
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/prefetch.sql",
                refs[0].as_deref(),
                refs[1].as_deref(),
                refs[2].as_deref(),
                refs[3].as_deref(),
                refs[4].as_deref()
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::BudgetWindow => {
            exact(ids, 3)?;
            let entity = entity(&ids[0])?;
            if entity == "key" {
                hashed_key(&ids[1])?;
            }
            sqlx::query_file_as!(
                AuthRow,
                "queries/auth/budget_window.sql",
                entity,
                &ids[1],
                &ids[2]
            )
            .fetch_all(executor)
            .await?
        }
        AuthRead::WindowSpend => {
            exact(ids, 3)?;
            let start = chrono::DateTime::parse_from_rfc3339(&ids[2])
                .map(|value| value.naive_utc())
                .or_else(|_| {
                    chrono::NaiveDateTime::parse_from_str(&ids[2], "%Y-%m-%dT%H:%M:%S%.f")
                })?;
            match entity(&ids[0])? {
                "key" => {
                    hashed_key(&ids[1])?;
                    sqlx::query_file_as!(
                        AuthRow,
                        "queries/auth/window_key_spend.sql",
                        &ids[1],
                        start
                    )
                    .fetch_all(executor)
                    .await?
                }
                _ => {
                    sqlx::query_file_as!(
                        AuthRow,
                        "queries/auth/window_team_spend.sql",
                        &ids[1],
                        start
                    )
                    .fetch_all(executor)
                    .await?
                }
            }
        }
    };
    Ok(rows)
}

pub async fn auth_spend<'e>(
    executor: impl Executor<'e, Database = Postgres>,
    kind: AuthSpend,
    id: &str,
) -> Result<Option<f64>, Error> {
    if id.len() > 65536 {
        return Err(Error::InvalidAuthArguments);
    }
    let row = match kind {
        AuthSpend::Key => {
            hashed_key(id)?;
            sqlx::query_file!("queries/auth/spend_key.sql", id)
                .fetch_optional(executor)
                .await?
                .map(|row| row.spend)
        }
        AuthSpend::User => sqlx::query_file!("queries/auth/spend_user.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|row| row.spend),
        AuthSpend::Team => sqlx::query_file!("queries/auth/spend_team.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|row| row.spend),
        AuthSpend::Org => sqlx::query_file!("queries/auth/spend_org.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|row| row.spend),
        AuthSpend::Project => sqlx::query_file!("queries/auth/spend_project.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|row| row.spend),
        AuthSpend::EndUser => sqlx::query_file!("queries/auth/spend_end_user.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|row| row.spend),
        AuthSpend::Tag => sqlx::query_file!("queries/auth/spend_tag.sql", id)
            .fetch_optional(executor)
            .await?
            .map(|row| row.spend),
        AuthSpend::ModelAccessGroup => {
            sqlx::query_file!("queries/auth/spend_model_access_group.sql", id)
                .fetch_optional(executor)
                .await?
                .map(|row| row.spend)
        }
        AuthSpend::Member => {
            let (user_id, team_id) = id.rsplit_once(':').ok_or(Error::InvalidAuthArguments)?;
            sqlx::query_file!("queries/auth/spend_member.sql", user_id, team_id)
                .fetch_optional(executor)
                .await?
                .map(|row| row.spend)
        }
    };
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::{AuthRead, AuthSpend, auth_read, auth_spend};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use serde_json::{Value, json};
    use sqlx::{PgPool, Postgres, Transaction};
    use std::error::Error as StdError;

    async fn seeded(pool: &PgPool) -> Result<Transaction<'_, Postgres>, sqlx::Error> {
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Transaction-local test fixture outside the production query set"
        )]
        sqlx::raw_sql(include_str!("../queries/auth/fixture.sql"))
            .execute(&mut *tx)
            .await?;
        Ok(tx)
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[rstest]
    #[case::user(AuthRead::User, &["auth-u"], "/organization_memberships/0/user_role", json!("org_admin"))]
    #[case::team(AuthRead::Team, &["auth-team"], "/litellm_model_table/model_aliases/alias", json!("actual"))]
    #[case::team_null_lists(AuthRead::Team, &["auth-team-empty"], "/models", json!([]))]
    #[case::team_missing_relation(AuthRead::Team, &["auth-team-empty"], "/object_permission", json!(null))]
    #[case::org(AuthRead::Org, &["auth-org"], "/litellm_budget_table/max_budget", json!(13.0))]
    #[case::project(AuthRead::Project, &["auth-project"], "/litellm_budget_table/max_budget", json!(0.0))]
    #[case::end_user(AuthRead::EndUser, &["auth-end"], "/models", json!([]))]
    #[case::budget(AuthRead::Budget, &["auth-b"], "/allowed_models", json!([]))]
    #[case::object_permission(AuthRead::ObjectPermission, &["auth-op"], "/mcp_servers", json!([]))]
    #[case::access_group(AuthRead::AccessGroup, &["auth-access"], "/access_model_names", json!(["group-model"]))]
    #[case::tag(AuthRead::Tag, &["auth-tag"], "/models", json!([]))]
    #[case::model_access_group(AuthRead::ModelAccessGroup, &["auth-model-group"], "/litellm_budget_table/max_budget", json!(0.0))]
    #[case::member(AuthRead::Member, &["auth-u","auth-team"], "/litellm_budget_table/rpm_limit", json!(33))]
    #[case::agent(AuthRead::Agent, &["auth-agent"], "/identity/active", json!(false))]
    #[case::retired_agent(AuthRead::RetiredAgent, &["auth-retired-agent"], "/original_agent_id", json!("auth-retired-agent"))]
    #[case::agent_identity(AuthRead::AgentIdentity, &["microsoft_entra","auth-tenant","auth-client"], "/required_roles", json!([]))]
    #[case::retired_agent_identity(AuthRead::RetiredAgentIdentity, &["microsoft_entra","auth-tenant","auth-retired-client"], "/binding_id", json!("auth-retired-binding"))]
    #[case::verified_subject(AuthRead::VerifiedSubject, &["auth-issuer","auth-tenant","auth-oid"], "/user_id", json!("auth-u"))]
    #[case::team_alias(AuthRead::TeamAlias, &["Auth Team"], "/team_id", json!("auth-team"))]
    #[case::org_alias(AuthRead::OrgAlias, &["Auth Org"], "/organization_id", json!("auth-org"))]
    #[case::sso(AuthRead::UserSso, &["auth-sso"], "/user_id", json!("auth-u"))]
    #[case::email_literal_insensitive(AuthRead::UserEmail, &["test_%@example.COM"], "/user_id", json!("auth-u"))]
    #[case::tags_registry(AuthRead::TagRegistry, &["5001"], "/tag_name", json!("auth-tag"))]
    #[case::model_registry(AuthRead::ModelAccessGroupRegistry, &["5001"], "/access_group_name", json!("auth-model-group"))]
    #[case::restricted_registry(AuthRead::EndUserRestrictedRegistry, &["5001"], "/user_id", json!("auth-end"))]
    #[case::budget_window(AuthRead::BudgetWindow, &["Team","auth-team","1h"], "/spend", json!(2.5))]
    #[case::window_spend_offset(AuthRead::WindowSpend, &["Team","auth-team","2025-01-01T02:00:00+01:00"], "/spend", json!(6.0))]
    #[case::window_spend_naive(AuthRead::WindowSpend, &["Team","auth-team","2025-01-01T01:00:00"], "/spend", json!(6.0))]
    #[case::window_spend_no_rows(AuthRead::WindowSpend, &["Team","absent","2025-01-01T01:00:00Z"], "/spend", json!(0.0))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn projects_auth_rows(
        #[case] kind: AuthRead,
        #[case] ids: &[&str],
        #[case] pointer: &str,
        #[case] expected: Value,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        let rows = auth_read(&mut *tx, kind, &strings(ids)).await?;
        assert_eq!(rows.len(), 1);
        let actual = rows[0].data.0.pointer(pointer).expect("projected field");
        if expected.is_number() {
            assert_eq!(actual.as_f64(), expected.as_f64());
        } else {
            assert_eq!(actual, &expected);
        }
        Ok(())
    }

    #[rstest]
    #[case::active('a', true)]
    #[case::deprecated('b', true)]
    #[case::expired('c', false)]
    #[case::no_chaining('d', false)]
    #[case::missing('f', false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn combined_key_join_and_deprecation(
        #[case] token: char,
        #[case] found: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        let rows = auth_read(&mut *tx, AuthRead::Key, &[token.to_string().repeat(64)]).await?;
        assert_eq!(rows.len(), usize::from(found));
        if !found {
            return Ok(());
        }
        let row = &rows[0].data.0;
        assert_eq!(row["token"], "a".repeat(64));
        assert_eq!(row["team_spend"], 2.5);
        assert_eq!(row["team_blocked"], true);
        assert_eq!(row["org_id"], "auth-org");
        assert_eq!(row["team_models"], json!(["team-model"]));
        assert_eq!(row["team_model_aliases"], json!({"alias":"actual"}));
        assert_eq!(row["team_member_spend"], 6.5);
        assert_eq!(row["team_member_tpm_limit"], 22);
        assert_eq!(row["litellm_budget_table_max_budget"], 4.5);
        assert_eq!(row["organization_max_budget"], 13.0);
        assert_eq!(row["project_alias"], "Auth Project");
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn combined_key_keeps_null_join_values(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        let rows = auth_read(&mut *tx, AuthRead::Key, &["e".repeat(64)]).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].data.0["team_models"], Value::Null);
        assert_eq!(rows[0].data.0["team_blocked"], Value::Null);
        assert_eq!(rows[0].data.0["organization_max_budget"], Value::Null);
        assert_eq!(rows[0].data.0["spend"], 0.0);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn prefetch_has_same_nested_projections(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        let refs = json!(["auth-u", "auth-team", "auth-u", "auth-org", "auth-project"]).to_string();
        let rows = auth_read(&mut *tx, AuthRead::Prefetch, &[refs]).await?;
        let user = auth_read(&mut *tx, AuthRead::User, &strings(&["auth-u"])).await?;
        let team = auth_read(&mut *tx, AuthRead::Team, &strings(&["auth-team"])).await?;
        let member = auth_read(
            &mut *tx,
            AuthRead::Member,
            &strings(&["auth-u", "auth-team"]),
        )
        .await?;
        let org = auth_read(&mut *tx, AuthRead::Org, &strings(&["auth-org"])).await?;
        let project = auth_read(&mut *tx, AuthRead::Project, &strings(&["auth-project"])).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].data.0,
            json!({
                "user_row": user[0].data.0, "team_row": team[0].data.0,
                "membership_row": member[0].data.0, "organization_row": org[0].data.0,
                "project_row": project[0].data.0
            })
        );
        let absent = auth_read(
            &mut *tx,
            AuthRead::Prefetch,
            &[json!([null, null, null, null, null]).to_string()],
        )
        .await?;
        assert_eq!(
            absent[0].data.0,
            json!({"user_row":null,"team_row":null,"membership_row":null,"organization_row":null,"project_row":null})
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn batch_ids_preserve_tenant_boundaries_and_missing_rows(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        let rows = auth_read(
            &mut *tx,
            AuthRead::User,
            &strings(&["auth-u", "missing", "auth-u"]),
        )
        .await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].data.0["user_id"], "auth-u");
        assert!(auth_read(&mut *tx, AuthRead::User, &[]).await?.is_empty());
        assert!(
            auth_read(
                &mut *tx,
                AuthRead::Member,
                &strings(&["auth-other", "auth-team"])
            )
            .await?
            .is_empty()
        );
        assert!(
            auth_read(
                &mut *tx,
                AuthRead::AgentIdentity,
                &strings(&["microsoft_entra", "other-tenant", "auth-client"])
            )
            .await?
            .is_empty()
        );
        assert!(
            auth_read(
                &mut *tx,
                AuthRead::UserEmail,
                &strings(&["test%@example.com"])
            )
            .await?
            .is_empty()
        );
        assert!(
            auth_read(&mut *tx, AuthRead::User, &strings(&["' OR 1=1 --"]))
                .await?
                .is_empty()
        );
        Ok(())
    }

    #[rstest]
    #[case::key(AuthSpend::Key, "a".repeat(64), 4.5)]
    #[case::user(AuthSpend::User, "auth-u".into(), 3.5)]
    #[case::zero(AuthSpend::User, "auth-other".into(), 0.0)]
    #[case::team(AuthSpend::Team, "auth-team".into(), 2.5)]
    #[case::org(AuthSpend::Org, "auth-org".into(), 13.5)]
    #[case::project(AuthSpend::Project, "auth-project".into(), 8.5)]
    #[case::end_user(AuthSpend::EndUser, "auth-end".into(), 10.5)]
    #[case::tag(AuthSpend::Tag, "auth-tag".into(), 11.5)]
    #[case::model_group(AuthSpend::ModelAccessGroup, "auth-model-group".into(), 12.5)]
    #[case::member(AuthSpend::Member, "auth-u:auth-team".into(), 6.5)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn spend_reads(
        #[case] kind: AuthSpend,
        #[case] id: String,
        #[case] expected: f64,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        assert_eq!(auth_spend(&mut *tx, kind, &id).await?, Some(expected));
        let absent = match kind {
            AuthSpend::Key => "f".repeat(64),
            AuthSpend::Member => "missing:missing".into(),
            _ => "missing".into(),
        };
        assert_eq!(auth_spend(&mut *tx, kind, &absent).await?, None);
        Ok(())
    }

    #[rstest]
    #[case::key_window(AuthRead::BudgetWindow, vec!["Key".into(), "a".repeat(64), "1h".into()], 2.5)]
    #[case::key_logs(AuthRead::WindowSpend, vec!["Key".into(), "a".repeat(64), "2025-01-01T01:00:00Z".into()], 6.0)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn key_window_spend(
        #[case] kind: AuthRead,
        #[case] ids: Vec<String>,
        #[case] expected: f64,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = seeded(&pool).await?;
        let rows = auth_read(&mut *tx, kind, &ids).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].data.0["spend"], expected);
        Ok(())
    }

    #[rstest]
    #[case::member(AuthRead::Member, vec!["one".into()])]
    #[case::registry(AuthRead::TagRegistry, vec!["0".into()])]
    #[case::huge_registry(AuthRead::TagRegistry, vec!["100002".into()])]
    #[case::raw_token(AuthRead::Key, vec!["sk-raw-secret".into()])]
    #[case::prefetch_shape(AuthRead::Prefetch, vec!["[]".into()])]
    #[case::unknown_window(AuthRead::BudgetWindow, strings(&["Unknown","id","1h"]))]
    #[tokio::test]
    async fn invalid_arguments_fail_before_queries(
        #[case] kind: AuthRead,
        #[case] ids: Vec<String>,
    ) {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgresql://localhost/unused")
            .unwrap();
        pool.close().await;
        assert!(matches!(
            auth_read(&pool, kind, &ids).await,
            Err(crate::Error::InvalidAuthArguments)
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn database_errors_are_not_misses() {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgresql://localhost/unused")
            .unwrap();
        pool.close().await;
        assert!(matches!(
            auth_read(&pool, AuthRead::User, &strings(&["id"])).await,
            Err(crate::Error::Query(sqlx::Error::PoolClosed))
        ));
        assert!(matches!(
            auth_spend(&pool, AuthSpend::User, "id").await,
            Err(crate::Error::Query(sqlx::Error::PoolClosed))
        ));
    }

    #[rstest]
    fn projected_credentials_are_redacted() {
        let row = super::AuthRow {
            data: super::AuthData(json!({"password":"credential-value"})),
        };
        let debug = format!("{row:?}");
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
    }
}
