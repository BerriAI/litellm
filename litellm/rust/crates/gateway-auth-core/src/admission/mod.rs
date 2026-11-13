//! Checks that need reads and decide whether an authenticated, authorized caller may spend.
//! They run last, so nothing is read or written here for a request a cheaper check refuses.

mod budgets;
mod models;

use crate::{
    AuthHost,
    authn::key::{KeyRequest, KeySettings},
    identity::key::KeyPrincipal,
    outcome::Refused,
};

/// The key's temporary budget, the model it resolves to, its budgets and the proxy budget alert.
pub(crate) async fn key<H: AuthHost>(
    host: &H,
    principal: KeyPrincipal<H::Rows>,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
) -> Result<KeyPrincipal<H::Rows>, Refused<H::Rows>> {
    let key = host.with_temp_budget(principal.key).await?;
    models::normalize(host, &key.row).await?;
    budgets::key(host, &key.row, principal.owner.as_ref(), request)
        .await
        .map_err(|error| Refused::new(error, Some(&key.row)))?;
    if settings.proxy_budget && settings.has_db {
        host.proxy_budget_alert(&key.row).await?;
    }
    Ok(KeyPrincipal { key, ..principal })
}
