use super::{Org, claims};
use crate::{
    AuthHost, Error,
    authn::jwt::{Claims, JwtSettings},
    host::OrgRow,
};

/// The organization the claims name by id, else by alias.
pub(crate) async fn read<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<Option<Org<OrgRow<H>>>, Error> {
    match (
        claims::org_id(claims, settings).filter(|text| !text.is_empty()),
        claims::org_alias(claims, settings).filter(|text| !text.is_empty()),
    ) {
        (Some(org_id), _) => host.org(&org_id).await,
        (None, Some(alias)) => host.org_by_alias(&alias).await,
        (None, None) => Ok(None),
    }
}
