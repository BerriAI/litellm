mod jwt;
mod key;

pub use jwt::{
    AgentIdentityFailure, AssertionRejection, ClaimDenial, JwtDenial, KeyError, KeyFailure,
    ModelAccessDenied, RbacDenial, Searched, TeamDenial, TokenError, VerifyDenial,
};
pub use key::KeyDenial;

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
    /// A host step raised. The host keeps that failure and surfaces it, so the core never renders this.
    #[error("the auth host failed a step")]
    HostFailed,
}

impl Error {
    /// The host could not answer, so the caller surfaces the host's own failure instead of a denial.
    pub fn is_host_failure(&self) -> bool {
        match self {
            Self::HostFailed => true,
            Self::PremiumRequired(_) | Self::Key(_) | Self::Jwt(_) | Self::BudgetExceeded(_) => {
                false
            }
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
