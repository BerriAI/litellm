//! JWT claim to virtual key mappings in `LiteLLM_JWTKeyMapping`.

use sqlx::PgExecutor;

use litellm_db::Error;

use super::repository::NewKeyMapping;

/// The issuer scope and claim of one mapping. An empty issuer is the issuer-agnostic scope.
#[macro_rules_attribute::apply(crate::db_row)]
pub struct MappedClaim {
    pub jwt_issuer: String,
    pub jwt_claim_name: String,
    pub jwt_claim_value: String,
}

/// The hashed key an active mapping gives the claim within exactly the issuer scope `jwt_issuer`.
pub(super) async fn find_mapped_token(
    executor: impl PgExecutor<'_>,
    jwt_claim_name: &str,
    jwt_claim_value: &str,
    jwt_issuer: &str,
) -> Result<Option<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/jwt_key_mapping/queries/find_mapped_token.sql",
        jwt_claim_name,
        jwt_claim_value,
        jwt_issuer
    )
    .fetch_optional(executor)
    .await?)
}

/// Every claim, active or not, mapped to any of `tokens`.
pub(super) async fn mapped_claims_for_tokens(
    executor: impl PgExecutor<'_>,
    tokens: &[String],
) -> Result<Vec<MappedClaim>, Error> {
    Ok(sqlx::query_file_as!(
        MappedClaim,
        "src/jwt_key_mapping/queries/mapped_claims_for_tokens.sql",
        tokens
    )
    .fetch_all(executor)
    .await?)
}

/// Inserts the mapping under `id` in the scope `jwt_issuer`; the hashed key now mapped, or `None` when the claim
/// is already mapped in that scope.
pub(super) async fn insert_mapping(
    executor: impl PgExecutor<'_>,
    id: &str,
    jwt_issuer: &str,
    mapping: NewKeyMapping<'_>,
) -> Result<Option<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/jwt_key_mapping/queries/insert_mapping.sql",
        id,
        jwt_issuer,
        mapping.jwt_claim_name,
        mapping.jwt_claim_value,
        mapping.token,
        mapping.written_by
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{MappedClaim, find_mapped_token, mapped_claims_for_tokens};
    use crate::Inserted;
    use crate::jwt_key_mapping::repository::{NewKeyMapping, insert_once};
    use crate::test_support::{OneTransaction, database_pool, leave_utc};
    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    async fn insert_mappings(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(r#"INSERT INTO "LiteLLM_VerificationToken" (token) VALUES ('jwt-key-a'), ('jwt-key-b')"#)
            .execute(&mut **tx)
            .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_JWTKeyMapping" (id, jwt_issuer, jwt_claim_name, jwt_claim_value, token, is_active)
            VALUES ('m1', '', 'sub', 'alice', 'jwt-key-a', true),
                   ('m2', 'https://idp', 'sub', 'alice', 'jwt-key-b', true),
                   ('m3', '', 'sub', 'bob', 'jwt-key-b', false)
            "#,
        )
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[case::agnostic_scope("sub", "alice", "", Some("jwt-key-a"))]
    #[case::issuer_scope("sub", "alice", "https://idp", Some("jwt-key-b"))]
    #[case::no_fallback_to_agnostic("sub", "alice", "https://other", None)]
    #[case::inactive("sub", "bob", "", None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn mapped_token_matches_one_active_scope(
        #[case] name: &str,
        #[case] value: &str,
        #[case] issuer: &str,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_mappings(&mut tx).await?;

        let token = find_mapped_token(&mut *tx, name, value, issuer).await?;

        assert_eq!(token.as_deref(), expected);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn mapped_claims_include_inactive_mappings(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_mappings(&mut tx).await?;

        let mut claims = mapped_claims_for_tokens(&mut *tx, &["jwt-key-b".to_owned()]).await?;
        claims.sort_by(|a, b| a.jwt_claim_value.cmp(&b.jwt_claim_value));

        assert_eq!(
            claims,
            vec![
                MappedClaim {
                    jwt_issuer: "https://idp".to_owned(),
                    jwt_claim_name: "sub".to_owned(),
                    jwt_claim_value: "alice".to_owned(),
                },
                MappedClaim {
                    jwt_issuer: String::new(),
                    jwt_claim_name: "sub".to_owned(),
                    jwt_claim_value: "bob".to_owned(),
                },
            ]
        );
        Ok(())
    }

    #[derive(Debug, PartialEq, sqlx::FromRow)]
    struct StoredMapping {
        token: String,
        is_active: bool,
        description: Option<String>,
        created_by: Option<String>,
        updated_by: Option<String>,
    }

    async fn stored_mappings(
        tx: &mut Transaction<'_, Postgres>,
        name: &str,
        value: &str,
        issuer: &str,
    ) -> Result<Vec<StoredMapping>, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        sqlx::query_as(
            r#"
            SELECT token, is_active, description, created_by, updated_by
            FROM "LiteLLM_JWTKeyMapping"
            WHERE jwt_claim_name = $1 AND jwt_claim_value = $2 AND jwt_issuer = $3
            "#,
        )
        .bind(name)
        .bind(value)
        .bind(issuer)
        .fetch_all(&mut **tx)
        .await
    }

    #[rstest]
    #[case::issuer_agnostic_scope("carol", None, "")]
    #[case::issuer_scope_beside_the_same_claim_in_the_agnostic_one(
        "bob",
        Some("https://idp"),
        "https://idp"
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_maps_a_claim_in_exactly_its_issuer_scope(
        #[case] value: &str,
        #[case] issuer: Option<&str>,
        #[case] stored_issuer: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_mappings(&mut tx).await?;
        let mapping = NewKeyMapping {
            jwt_claim_name: "sub",
            jwt_claim_value: value,
            jwt_issuer: issuer,
            token: "jwt-key-a",
            written_by: "auto_register",
        };

        let inserted = insert_once(&mut OneTransaction(&mut tx), mapping).await?;

        assert_eq!(
            (
                inserted,
                stored_mappings(&mut tx, "sub", value, stored_issuer).await?
            ),
            (
                Inserted::Created("jwt-key-a".to_owned()),
                vec![StoredMapping {
                    token: "jwt-key-a".to_owned(),
                    is_active: true,
                    description: None,
                    created_by: Some("auto_register".to_owned()),
                    updated_by: Some("auto_register".to_owned()),
                }]
            )
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_gives_the_mapping_a_random_uuid_and_stamps_it_in_utc(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_mappings(&mut tx).await?;
        leave_utc(&mut tx).await?;
        let mapping = NewKeyMapping {
            jwt_claim_name: "sub",
            jwt_claim_value: "carol",
            jwt_issuer: None,
            token: "jwt-key-a",
            written_by: "auto_register",
        };

        insert_once(&mut OneTransaction(&mut tx), mapping).await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let (id, created_at_drift, updated_at_drift): (String, f64, f64) = sqlx::query_as(
            r#"
            SELECT id,
                   abs(extract(epoch FROM created_at - (now() AT TIME ZONE 'UTC')))::float8,
                   abs(extract(epoch FROM updated_at - (now() AT TIME ZONE 'UTC')))::float8
            FROM "LiteLLM_JWTKeyMapping"
            WHERE jwt_claim_value = 'carol'
            "#,
        )
        .fetch_one(&mut *tx)
        .await?;
        let parsed = uuid::Uuid::parse_str(&id)?;
        assert_eq!(
            (parsed.get_version_num(), parsed.hyphenated().to_string()),
            (4, id)
        );
        assert!(
            created_at_drift < 1.0 && updated_at_drift < 1.0,
            "created_at is {created_at_drift}s and updated_at {updated_at_drift}s from UTC now"
        );
        Ok(())
    }

    #[rstest]
    #[case::an_active_mapping_wins("alice", Some("jwt-key-a"), "jwt-key-a")]
    #[case::an_inactive_mapping_holds_the_claim_without_a_key("bob", None, "jwt-key-b")]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_of_a_mapped_claim_returns_the_stored_key_and_writes_nothing(
        #[case] value: &str,
        #[case] winner: Option<&str>,
        #[case] stored_token: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_mappings(&mut tx).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(r#"INSERT INTO "LiteLLM_VerificationToken" (token) VALUES ('jwt-key-c')"#)
            .execute(&mut *tx)
            .await?;
        let before = stored_mappings(&mut tx, "sub", value, "").await?;
        let mapping = NewKeyMapping {
            jwt_claim_name: "sub",
            jwt_claim_value: value,
            jwt_issuer: None,
            token: "jwt-key-c",
            written_by: "auto_register",
        };

        let inserted = insert_once(&mut OneTransaction(&mut tx), mapping).await?;

        let after = stored_mappings(&mut tx, "sub", value, "").await?;
        assert_eq!(
            (
                inserted,
                after.first().map(|stored| stored.token.as_str()),
                &after
            ),
            (
                Inserted::Conflict(winner.map(str::to_owned)),
                Some(stored_token),
                &before
            )
        );
        Ok(())
    }
}
