//! The writes a request earns, applied only once every check has allowed it. Identity collects
//! them as [`Grants`], and [`apply`] is the only caller of the host's grant writes.

use crate::{
    AuthHost, Error, JwtDenial, Rows,
    authn::jwt::{KeyMappingWrite, mapping::Registration},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Grant<R: Rows> {
    /// `AgentIdentityStore.record_authentication` for a managed agent.
    RecordAgentAuthentication(R::Agent),
    /// `team_member_add` for a provisioned user who is not yet in the team the JWT selected.
    AddTeamMember { user_id: String, team_id: String },
    /// A JWT's virtual key mapping under `auto_register` or `link_only`.
    CreateKeyMapping(Registration),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Grants<R: Rows>(Vec<Grant<R>>);

impl<R: Rows> Default for Grants<R> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<R: Rows> Grants<R> {
    pub(crate) fn with(self, grant: Grant<R>) -> Self {
        Self(self.0.into_iter().chain([grant]).collect())
    }
}

/// What applying the grants wrote that the caller still reads.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Applied {
    pub(crate) key_mapping: Option<KeyMappingWrite>,
}

/// Writes every grant in the order identity collected them.
pub(crate) async fn apply<H: AuthHost>(
    host: &H,
    grants: Grants<H::Rows>,
) -> Result<Applied, Error> {
    let mut applied = Applied::default();
    for grant in grants.0 {
        match grant {
            Grant::RecordAgentAuthentication(agent) => {
                if let Some(failure) = host.record_agent_authentication(&agent).await? {
                    return Err(JwtDenial::AgentIdentity(failure).into());
                }
            }
            Grant::AddTeamMember { user_id, team_id } => {
                host.add_team_member(&user_id, &team_id).await?;
            }
            Grant::CreateKeyMapping(registration) => {
                applied.key_mapping = Some(host.create_key_mapping(&registration.row()).await?);
            }
        }
    }
    Ok(applied)
}
