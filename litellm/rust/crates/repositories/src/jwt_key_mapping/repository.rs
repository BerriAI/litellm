//! The JWT key mapping repository: which virtual key a JWT claim maps to, and which claims map to given keys.

use litellm_db::Call;

use super::queries::{self, MappedClaim};
use crate::{Database, Side};

const MAPPED_TOKEN: Call = Call {
    call_type: "get_jwt_key_mapping_object",
    table: Some("LiteLLM_JWTKeyMapping"),
    operation: Some("find_first"),
};

const MAPPED_CLAIMS: Call = Call {
    call_type: "get_jwt_key_mapping_cache_keys_for_tokens",
    table: Some("LiteLLM_JWTKeyMapping"),
    operation: Some("find_many"),
};

/// Reads `LiteLLM_JWTKeyMapping`.
pub struct JwtKeyMappingRepository {
    database: Database,
}

impl JwtKeyMappingRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The hashed key an active mapping gives the claim in exactly the scope of `jwt_issuer`, where `None` is
    /// the issuer-agnostic scope. It never falls back from an issuer's scope to the agnostic one.
    pub async fn mapped_token(
        &self,
        jwt_claim_name: &str,
        jwt_claim_value: &str,
        jwt_issuer: Option<&str>,
    ) -> Result<Option<String>, litellm_db::Error> {
        let issuer = jwt_issuer.unwrap_or_default();
        self.database
            .run(MAPPED_TOKEN, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_mapped_token(
                    session.executor(),
                    jwt_claim_name,
                    jwt_claim_value,
                    issuer,
                )
                .await
            })
            .await
    }

    /// Every claim, active or not, mapped to any of `tokens`, so their cached answers can be evicted.
    pub async fn mapped_claims(
        &self,
        tokens: &[String],
    ) -> Result<Vec<MappedClaim>, litellm_db::Error> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        self.database
            .run(MAPPED_CLAIMS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::mapped_claims_for_tokens(session.executor(), tokens).await
            })
            .await
    }
}
