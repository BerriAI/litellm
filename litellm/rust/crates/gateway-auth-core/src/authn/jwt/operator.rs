use serde_json::{Map, Value};

use super::{Claims, JwtSettings};
use crate::{AuthHost, Error, JwtDenial, OperatorHooks};

/// Whether the operator's `custom_validate` accepts `claims`, true when none is configured. A host
/// with no operator hooks cannot say, so the request is refused.
pub(super) async fn operator_accepts(
    host: &impl AuthHost,
    claims: &Map<String, Value>,
    settings: &JwtSettings,
) -> Result<bool, Error> {
    if settings.config.custom_validate.is_none() {
        return Ok(true);
    }
    let hooks = host
        .operator_hooks()
        .ok_or(JwtDenial::CustomValidateUnavailable)?;
    hooks.custom_validate(&Value::Object(claims.clone())).await
}

/// `custom_validate` on verified claims: a `false` refuses the token.
pub(crate) async fn custom_validate(
    host: &impl AuthHost,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<(), Error> {
    if !operator_accepts(host, &claims.0, settings).await? {
        return Err(JwtDenial::CustomValidate.into());
    }
    Ok(())
}
