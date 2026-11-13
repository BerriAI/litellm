const NOT_PREMIUM_USER: &str = "You must be a LiteLLM Enterprise user to use this feature. If you have a license please \
set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. \n\
Pricing: https://www.litellm.ai/#pricing";

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PremiumRequired {
    #[error("Oauth2 token validation is only available for premium users. {NOT_PREMIUM_USER}")]
    OAuth2,
    #[error("JWT Auth is an enterprise only feature. {NOT_PREMIUM_USER}")]
    Jwt,
}
