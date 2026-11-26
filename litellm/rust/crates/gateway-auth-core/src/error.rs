pub(crate) mod body;
mod jwt;
mod key;

pub(crate) use jwt::InvalidCredentials;
pub use jwt::{
    AgentIdentityFailure, AssertionRejection, ClaimDenial, JwtDenial, KeyError, KeyFailure,
    RbacDenial, Searched, TeamDenial, TokenError, VerifyDenial,
};
pub use key::KeyDenial;

use crate::routes::RouteNotExposed;

const NOT_PREMIUM_USER: &str = "You must be a LiteLLM Enterprise user to use this feature. If you have a license please \
set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. \n\
Pricing: https://www.litellm.ai/#pricing";

/// Why a request does not authenticate: a denial the core renders, or a host that could not answer.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    PremiumRequired(#[from] PremiumRequired),
    #[error(transparent)]
    Key(#[from] KeyDenial),
    #[error(transparent)]
    Jwt(#[from] JwtDenial),
    #[error(transparent)]
    BudgetExceeded(#[from] BudgetExceeded),
    /// The end user's budget, which the key path reads before it reads the key.
    #[error(transparent)]
    EndUserBudgetExceeded(BudgetExceeded),
    /// A host step raised. The host keeps that failure and surfaces it, so the core never renders this.
    #[error("the auth host failed a step")]
    HostFailed,
    /// The host holds what a step asked for and does not serve it, so its caller answers the
    /// request another way. No read treats it as absent.
    #[error("the auth host does not serve this request")]
    HostDeclined,
}

impl Error {
    /// A read whose failure the core treats as nothing found. A host that declines found
    /// something, so that ends the call.
    pub(crate) fn absent_unless_declined<T>(
        read: Result<Option<T>, Self>,
    ) -> Result<Option<T>, Self> {
        match read {
            Err(Self::HostDeclined) => Err(Self::HostDeclined),
            read => Ok(read.ok().flatten()),
        }
    }

    /// The host could not answer, so the caller surfaces the host's own failure instead of a denial.
    pub fn is_host_failure(&self) -> bool {
        match self {
            Self::HostFailed | Self::HostDeclined => true,
            Self::PremiumRequired(_)
            | Self::Key(_)
            | Self::Jwt(_)
            | Self::BudgetExceeded(_)
            | Self::EndUserBudgetExceeded(_) => false,
        }
    }

    /// The request was refused before its credential authenticated, so it leaves no spend log row.
    pub fn failed_authentication(&self) -> bool {
        match self {
            Self::PremiumRequired(_) | Self::EndUserBudgetExceeded(_) => true,
            Self::Key(denial) => denial.failed_authentication(),
            Self::Jwt(denial) => denial.failed_authentication(),
            Self::BudgetExceeded(_) | Self::HostFailed | Self::HostDeclined => false,
        }
    }

    /// The end user's budget refusal, told apart from the key's budgets that run after the key authenticated.
    pub(crate) fn into_end_user_refusal(self) -> Self {
        match self {
            Self::BudgetExceeded(exceeded) => Self::EndUserBudgetExceeded(exceeded),
            error => error,
        }
    }
}

impl From<VerifyDenial> for Error {
    fn from(denial: VerifyDenial) -> Self {
        Self::Jwt(JwtDenial::Verify(denial))
    }
}

impl From<ClaimDenial> for Error {
    fn from(denial: ClaimDenial) -> Self {
        Self::Jwt(JwtDenial::Claim(denial))
    }
}

impl From<RbacDenial> for Error {
    fn from(denial: RbacDenial) -> Self {
        Self::Jwt(JwtDenial::Rbac(denial))
    }
}

impl From<TeamDenial> for Error {
    fn from(denial: TeamDenial) -> Self {
        Self::Jwt(JwtDenial::Team(denial))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PremiumRequired {
    #[error("Oauth2 token validation is only available for premium users. {NOT_PREMIUM_USER}")]
    OAuth2,
    #[error("JWT Auth is an enterprise only feature. {NOT_PREMIUM_USER}")]
    Jwt,
}

/// What the client reads when a model is refused, which never names the models that are allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "The requested model '{model}' is not available for this API key, or the model name is invalid. Check the models \
     available to you and try again."
)]
pub struct ModelAccessDenied<'a> {
    pub model: &'a str,
}

/// Whose model list refused the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeniedList {
    Key,
}

/// A request `authz` refuses for a caller the core authenticated, whichever credential it presented.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthzDenial {
    #[error("{}", ModelAccessDenied { model })]
    ModelAccess { list: DeniedList, model: String },
}

/// A request the proxy refuses before it reads a credential.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RequestDenial {
    #[error(transparent)]
    Address(#[from] AddressNotAllowed),
    #[error(transparent)]
    Route(#[from] RouteNotExposed),
}

/// An address the proxy's allowlist leaves out.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("Access forbidden: IP address {address} not allowed.")]
pub struct AddressNotAllowed {
    pub address: String,
}

/// `litellm.BudgetExceededError`, from the core's own comparison or as a host-owned budget step reported it.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct BudgetExceeded {
    pub current_cost: f64,
    pub max_budget: f64,
    pub message: String,
    /// What a core-decided budget names, such as `Key=alias`, for a host that renders the amounts
    /// its own way. Unset when the host wrote `message`.
    pub subject: Option<String>,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use jiff::Timestamp;
    use litellm_config::UnregisteredJwtClientBehavior;
    use rstest::rstest;

    use super::*;
    use crate::Token;

    fn over_budget() -> BudgetExceeded {
        BudgetExceeded {
            current_cost: 2.0,
            max_budget: 1.0,
            message: "Budget has been exceeded!".to_owned(),
            subject: None,
            entity_type: None,
            entity_id: None,
        }
    }

    #[rstest]
    #[case::premium_required(PremiumRequired::Jwt.into())]
    #[case::end_user_budget(Error::EndUserBudgetExceeded(over_budget()))]
    #[case::missing_key(KeyDenial::MissingKey.into())]
    #[case::malformed_key(KeyDenial::MalformedKey.into())]
    #[case::user_auth_disabled(KeyDenial::UserAuthDisabled.into())]
    #[case::expired_key(KeyDenial::Expired { expires: Timestamp::UNIX_EPOCH, now: Timestamp::UNIX_EPOCH, key: "sk-...1".to_owned() }.into())]
    #[case::master_key_only_route(KeyDenial::MasterKeyOnlyRoute { route: "/global/spend/reset".to_owned() }.into())]
    #[case::no_db(KeyDenial::NoDb.into())]
    #[case::no_db_configured(KeyDenial::NoDbConfigured.into())]
    #[case::not_virtual_key(KeyDenial::NotVirtualKey { received: "abc".to_owned(), jwt_hint: false }.into())]
    #[case::key_not_found(KeyDenial::NotFound { hash: Token::new("sk-1").hash() }.into())]
    #[case::blocked_key(KeyDenial::Blocked.into())]
    #[case::key_owner_deactivated(KeyDenial::UserDeactivated { user_id: "u1".to_owned() }.into())]
    #[case::key_without_token(KeyDenial::NoToken.into())]
    #[case::jwt_fails_verification(VerifyDenial::Expired.into())]
    #[case::custom_validate_rejects(JwtDenial::CustomValidate.into())]
    #[case::custom_validate_unavailable(JwtDenial::CustomValidateUnavailable.into())]
    #[case::managed_issuer_untrusted(JwtDenial::ManagedIssuerUntrusted.into())]
    #[case::jwt_user_deactivated(JwtDenial::UserDeactivated { user_id: None }.into())]
    #[case::managed_agent_mapped(JwtDenial::ManagedAgentNeedsDirectJwt.into())]
    #[case::mapping_claim_missing(JwtDenial::MappingClaimMissing { claim_field: "client_id".to_owned() }.into())]
    #[case::no_mapping(JwtDenial::NoMapping { claim_field: "client_id".to_owned(), claim_value: "app".to_owned() }.into())]
    #[case::mapping_needs_db(JwtDenial::MappingNeedsDb { behavior: UnregisteredJwtClientBehavior::AutoRegister }.into())]
    #[case::no_eligible_key(JwtDenial::NoEligibleKey.into())]
    #[case::mapping_race_vanished(JwtDenial::MappingRaceVanished { behavior: UnregisteredJwtClientBehavior::AutoRegister }.into())]
    fn refusals_before_the_credential_authenticates_failed_authentication(#[case] error: Error) {
        assert!(error.failed_authentication(), "{error:?}");
    }

    #[rstest]
    #[case::key_budget(Error::BudgetExceeded(over_budget()))]
    #[case::host_failed(Error::HostFailed)]
    #[case::host_declined(Error::HostDeclined)]
    #[case::scope_claim_type(ClaimDenial::UnmappedScopeType { kind: "number" }.into())]
    #[case::rbac(RbacDenial::UnmatchedRole.into())]
    #[case::team(TeamDenial::NoTeamsInToken.into())]
    #[case::managed_route(JwtDenial::ManagedRouteNotAllowed.into())]
    #[case::agent_identity(JwtDenial::AgentIdentity(AgentIdentityFailure { policy_unavailable: false, message: "retired".to_owned() }).into())]
    #[case::unregistered_agent(JwtDenial::UnregisteredAgent { agent_id_jwt_field: None, claim: "a1".to_owned() }.into())]
    #[case::granting_team_route(JwtDenial::GrantingTeamRouteNotAllowed.into())]
    #[case::email_domain(JwtDenial::EmailDomainNotAllowed { user_email: None, allowed_domain: None }.into())]
    #[case::jwt_user_not_found(JwtDenial::UserNotFound { user_id: "ghost".to_owned() }.into())]
    #[case::no_user_or_team_id(JwtDenial::NoObjectId.into())]
    fn refusals_of_an_authenticated_caller_keep_their_row(#[case] error: Error) {
        assert!(!error.failed_authentication(), "{error:?}");
    }

    #[rstest]
    #[case::budget_becomes_the_end_users(
        Error::BudgetExceeded(over_budget()),
        Error::EndUserBudgetExceeded(over_budget())
    )]
    #[case::other_errors_pass_through(Error::HostFailed, Error::HostFailed)]
    fn the_end_users_refusal_is_its_own_error(#[case] error: Error, #[case] expected: Error) {
        assert_eq!(error.into_end_user_refusal(), expected);
    }
}
