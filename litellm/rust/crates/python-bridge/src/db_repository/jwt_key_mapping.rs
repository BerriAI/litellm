use std::sync::Arc;

use litellm_repositories::{Inserted, JwtKeyMappingRepository, NewKeyMapping};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The JWT key mapping repository.
#[pyclass(frozen)]
pub(crate) struct RustJwtKeyMappingRepository(Arc<JwtKeyMappingRepository>);

#[pymethods]
impl RustJwtKeyMappingRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(JwtKeyMappingRepository::new(
            database.repository_database()?,
        ))))
    }

    /// The hashed key mapped to the claim in exactly the scope of `jwt_issuer`, or `None`.
    #[pyo3(signature = (jwt_claim_name, jwt_claim_value, jwt_issuer=None))]
    fn mapped_token<'py>(
        &self,
        py: Python<'py>,
        jwt_claim_name: String,
        jwt_claim_value: String,
        jwt_issuer: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository
                    .mapped_token(&jwt_claim_name, &jwt_claim_value, jwt_issuer.as_deref())
                    .await
                    .map_err(map_error)
            })?,
        )
    }

    /// `(jwt_issuer, jwt_claim_name, jwt_claim_value)` of every mapping to any of `tokens`.
    fn mapped_claims<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let claims = repository.mapped_claims(&tokens).await.map_err(map_error)?;
                Ok(claims
                    .into_iter()
                    .map(|claim| {
                        (
                            claim.jwt_issuer,
                            claim.jwt_claim_name,
                            claim.jwt_claim_value,
                        )
                    })
                    .collect::<Vec<_>>())
            })?,
        )
    }

    /// `(created, token)`: whether this call mapped the claim, then the hashed key now mapped to it, `None`
    /// when the mapping that holds the claim is inactive or gone.
    #[pyo3(signature = (jwt_claim_name, jwt_claim_value, token, *, jwt_issuer=None, written_by))]
    fn insert<'py>(
        &self,
        py: Python<'py>,
        jwt_claim_name: String,
        jwt_claim_value: String,
        token: String,
        jwt_issuer: Option<String>,
        written_by: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let mapping = NewKeyMapping {
                    jwt_claim_name: &jwt_claim_name,
                    jwt_claim_value: &jwt_claim_value,
                    jwt_issuer: jwt_issuer.as_deref(),
                    token: &token,
                    written_by: &written_by,
                };
                match repository.insert(mapping).await.map_err(map_error)? {
                    Inserted::Created(token) => Ok((true, Some(token))),
                    Inserted::Conflict(token) => Ok((false, token)),
                }
            })?,
        )
    }
}
