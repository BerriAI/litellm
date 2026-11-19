//! JWT claim to virtual key mappings in `LiteLLM_JWTKeyMapping`.

use sqlx::PgExecutor;

use litellm_db::Error;

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

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{MappedClaim, find_mapped_token, mapped_claims_for_tokens};
    use crate::test_support::database_pool;
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
}
