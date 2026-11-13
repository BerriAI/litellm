use crate::{
    AuthHost, Error,
    authn::key::KeyRequest,
    host::{KeyRow, UserRow},
};

/// The budgets main runs inside authentication, in main's order. A free request skips them.
pub(super) async fn key<H: AuthHost>(
    host: &H,
    key: &KeyRow<H>,
    owner: Option<&UserRow<H>>,
    request: &KeyRequest<'_>,
) -> Result<(), Error> {
    if host.key_skips_budget_checks(key).await? {
        return Ok(());
    }
    host.team_member_budget(key, owner).await?;
    host.key_budgets(key, owner, request.is_llm_api_route).await
}
