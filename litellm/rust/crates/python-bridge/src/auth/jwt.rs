use litellm_gateway_auth_core::{
    authn::jwt::{self as authn_jwt, Claims, JwtSettings},
    identity::jwt,
};
use litellm_host_python::{from_py, to_py};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use super::settings::jwt_auth;

/// The claim readers of `jwt::claims` for one `LiteLLM_JWTAuth`, plus the role it maps to.
#[pyclass(
    frozen,
    get_all,
    name = "RustMappedClaims",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct MappedClaimsView {
    /// `get_user_id`, falling back to the email.
    user_id: Option<String>,
    /// `get_user_id` with no fallback.
    claim_user_id: Option<String>,
    user_email: Option<String>,
    valid_user_email: Option<bool>,
    team_id: Option<String>,
    team_ids: Vec<String>,
    all_team_ids: Vec<String>,
    team_alias: Option<String>,
    org_id: Option<String>,
    org_alias: Option<String>,
    end_user_id: Option<String>,
    object_id: Option<String>,
    agent_claim: Option<String>,
    litellm_role: Option<String>,
}

fn settings(litellm_jwtauth: &Bound<'_, PyAny>) -> PyResult<JwtSettings> {
    Ok(JwtSettings {
        config: jwt_auth(litellm_jwtauth)?,
        ..JwtSettings::default()
    })
}

#[pyfunction]
pub(crate) fn jwt_is_jwt(token: Option<&str>) -> bool {
    token.is_some_and(authn_jwt::is_jwt)
}

#[pyfunction]
pub(crate) fn jwt_unverified_claims(py: Python<'_>, token: &str) -> PyResult<Option<Py<PyAny>>> {
    authn_jwt::unverified_claims(token)
        .map(|claims| to_py(py, &claims.0))
        .transpose()
}

#[pyfunction]
pub(crate) fn jwt_presents_claims(token: &str, claims: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(authn_jwt::presents_claims(
        token,
        &Claims(from_py::<Map<String, Value>>(claims)?),
    ))
}

#[pyfunction]
pub(crate) fn jwt_claims(
    claims: &Bound<'_, PyAny>,
    litellm_jwtauth: &Bound<'_, PyAny>,
) -> PyResult<MappedClaimsView> {
    let claims = Claims(from_py::<Map<String, Value>>(claims)?);
    let settings = settings(litellm_jwtauth)?;
    let mapped = jwt::claims::claims(&claims, &settings);
    Ok(MappedClaimsView {
        user_id: mapped.user.user_id,
        claim_user_id: jwt::user_id(&claims, &settings, None),
        user_email: mapped.user.user_email,
        valid_user_email: mapped.user.valid_user_email,
        team_id: mapped.team_id,
        team_ids: mapped.team_ids,
        all_team_ids: mapped.all_team_ids,
        team_alias: mapped.team_alias,
        org_id: mapped.org_id,
        org_alias: mapped.org_alias,
        end_user_id: mapped.end_user_id,
        object_id: mapped.object_id,
        agent_claim: mapped.agent_claim,
        litellm_role: jwt::litellm_role(&claims, &settings)
            .map(|role| <&str>::from(role).to_owned()),
    })
}

#[pyfunction]
pub(crate) fn jwt_is_allowed_domain(
    user_email: &str,
    litellm_jwtauth: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(jwt::is_allowed_domain(
        user_email,
        &settings(litellm_jwtauth)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (claim_field, claim_value, issuer=None))]
pub(crate) fn jwt_key_mapping_cache_key(
    claim_field: &str,
    claim_value: &str,
    issuer: Option<&str>,
) -> String {
    authn_jwt::mapping_cache_key(claim_field, claim_value, issuer)
}
