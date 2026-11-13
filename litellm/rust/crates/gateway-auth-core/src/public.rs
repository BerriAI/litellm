use crate::PremiumRequired;

/// `ProxyErrorTypes` values a denial carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum ErrorType {
    #[strum(serialize = "auth_error")]
    Auth,
}

/// A denial as the caller sees it: the exception the Python proxy raised for it, with its status
/// and message. Each host builds its error from this alone.
#[derive(Clone, Debug, PartialEq)]
pub enum PublicDenial {
    /// `ProxyException(message, type, param, code)`
    Proxy {
        status: u16,
        message: String,
        error_type: ErrorType,
        param: Option<String>,
    },
}

fn proxy(
    status: u16,
    denial: &impl ToString,
    error_type: ErrorType,
    param: Option<&str>,
) -> PublicDenial {
    PublicDenial::Proxy {
        status,
        message: denial.to_string(),
        error_type,
        param: param.map(str::to_owned),
    }
}

impl PremiumRequired {
    pub fn public(&self) -> PublicDenial {
        proxy(403, self, ErrorType::Auth, Some("premium_user"))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::jwt(PremiumRequired::Jwt)]
    #[case::oauth2(PremiumRequired::OAuth2)]
    fn premium_required_is_a_403_naming_the_license(#[case] premium: PremiumRequired) {
        assert_eq!(
            premium.public(),
            PublicDenial::Proxy {
                status: 403,
                message: premium.to_string(),
                error_type: ErrorType::Auth,
                param: Some("premium_user".to_owned()),
            }
        );
    }
}
