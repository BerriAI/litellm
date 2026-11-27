//! The JWT key mapping repository: which virtual key a JWT claim maps to, which claims map to given keys, and
//! the insert that maps a claim.

use litellm_db::Call;
use uuid::Uuid;

use super::queries::{self, MappedClaim};
use crate::{Database, Inserted, Side, database::Connections};

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

const AUTO_REGISTER_MAPPING: Call = Call {
    call_type: "auto_register_jwt_mapping",
    table: Some("LiteLLM_JWTKeyMapping"),
    operation: Some("insert"),
};

/// A claim to map to the virtual key hashed as `token`, in the scope of `jwt_issuer`, where `None` is the
/// issuer-agnostic scope. `written_by` is stored as who created and last updated the mapping.
#[derive(Clone, Copy, Debug)]
pub struct NewKeyMapping<'a> {
    pub jwt_claim_name: &'a str,
    pub jwt_claim_value: &'a str,
    pub jwt_issuer: Option<&'a str>,
    pub token: &'a str,
    pub written_by: &'a str,
}

/// Reads and writes `LiteLLM_JWTKeyMapping`.
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

    /// Maps the claim to its key and returns the hashed key now mapped. When the claim is already mapped in that
    /// scope the stored mapping is left alone and its hashed key comes back instead, `None` if it is inactive
    /// or gone.
    pub async fn insert(
        &self,
        mapping: NewKeyMapping<'_>,
    ) -> Result<Inserted<String>, litellm_db::Error> {
        self.database
            .run(AUTO_REGISTER_MAPPING, || async {
                insert_once(&mut &self.database, mapping).await
            })
            .await
    }
}

/// The insert, then on a conflict a read of the mapping that won, in a statement of its own because the insert
/// cannot see a row committed after it began.
pub(super) async fn insert_once(
    connections: &mut impl Connections,
    mapping: NewKeyMapping<'_>,
) -> Result<Inserted<String>, litellm_db::Error> {
    let id = Uuid::new_v4().to_string();
    let jwt_issuer = mapping.jwt_issuer.unwrap_or_default();
    let created = connections
        .with(Side::Writer, async |db| {
            queries::insert_mapping(db, &id, jwt_issuer, mapping).await
        })
        .await?;
    if let Some(token) = created {
        return Ok(Inserted::Created(token));
    }
    connections
        .with(Side::Writer, async |db| {
            queries::find_mapped_token(
                db,
                mapping.jwt_claim_name,
                mapping.jwt_claim_value,
                jwt_issuer,
            )
            .await
        })
        .await
        .map(Inserted::Conflict)
}
