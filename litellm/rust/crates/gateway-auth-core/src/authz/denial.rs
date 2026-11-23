//! Why authz refuses a request, with what the Python proxy answers for it: the exception class,
//! the status, the `type` and `param`, and the message, whose opening sentence clients match on.

/// The exception class the Python proxy raises a denial as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum Raised {
    #[strum(serialize = "ModelAccessDeniedProxyException")]
    ModelAccessDeniedProxy,
}

/// Whose model list refused the model. The proxy answers each with a `type` of its own, and a
/// team member's with the team's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum DeniedList {
    #[strum(serialize = "key_model_access_denied")]
    Key,
    #[strum(serialize = "team_model_access_denied")]
    Team,
    #[strum(serialize = "team_model_access_denied")]
    Member,
    #[strum(serialize = "user_model_access_denied")]
    User,
    /// A user whose list is `no-default-models`, which the proxy answers with the key's type.
    #[strum(serialize = "key_model_access_denied")]
    UserWithoutDefaults,
    #[strum(serialize = "project_model_access_denied")]
    Project,
    #[strum(serialize = "org_model_access_denied")]
    Org,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthzDenial {
    #[error(
        "The requested model '{model}' is not available for this API key, or the model name is invalid. Check the models available to you and try again."
    )]
    ModelAccess { list: DeniedList, model: String },
    /// The end user's list refused the model, which the proxy words for the customer.
    #[error(
        "The requested model '{model}' is not in the allowed models for this customer. Check the models this customer can use and try again."
    )]
    CustomerModelAccess { model: String },
}

impl AuthzDenial {
    pub fn raised(&self) -> Raised {
        match self {
            Self::ModelAccess { .. } | Self::CustomerModelAccess { .. } => {
                Raised::ModelAccessDeniedProxy
            }
        }
    }

    pub fn status(&self) -> u16 {
        match self {
            Self::ModelAccess { .. } | Self::CustomerModelAccess { .. } => 403,
        }
    }

    /// The `ProxyErrorTypes` value the proxy answers with.
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::ModelAccess { list, .. } => (*list).into(),
            Self::CustomerModelAccess { .. } => "customer_model_access_denied",
        }
    }

    pub fn param(&self) -> Option<&'static str> {
        match self {
            Self::ModelAccess { .. } | Self::CustomerModelAccess { .. } => Some("model"),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn refused(list: DeniedList) -> AuthzDenial {
        AuthzDenial::ModelAccess {
            list,
            model: "gpt".to_owned(),
        }
    }

    #[rstest]
    #[case::key(refused(DeniedList::Key), "key_model_access_denied")]
    #[case::team(refused(DeniedList::Team), "team_model_access_denied")]
    #[case::member(refused(DeniedList::Member), "team_model_access_denied")]
    #[case::user(refused(DeniedList::User), "user_model_access_denied")]
    #[case::user_without_defaults(
        refused(DeniedList::UserWithoutDefaults),
        "key_model_access_denied"
    )]
    #[case::project(refused(DeniedList::Project), "project_model_access_denied")]
    #[case::org(refused(DeniedList::Org), "org_model_access_denied")]
    #[case::end_user(
        AuthzDenial::CustomerModelAccess { model: "gpt".to_owned() },
        "customer_model_access_denied"
    )]
    fn a_model_denial_is_a_403_on_the_model_param_with_its_lists_type(
        #[case] denial: AuthzDenial,
        #[case] error_type: &str,
    ) {
        assert_eq!(
            (
                <&str>::from(denial.raised()),
                denial.status(),
                denial.error_type(),
                denial.param()
            ),
            (
                "ModelAccessDeniedProxyException",
                403,
                error_type,
                Some("model")
            )
        );
    }

    #[rstest]
    #[case::a_list(
        refused(DeniedList::Team),
        "The requested model 'gpt' is not available for this API key, or the model name is invalid."
    )]
    #[case::the_end_users(
        AuthzDenial::CustomerModelAccess { model: "gpt".to_owned() },
        "The requested model 'gpt' is not in the allowed models for this customer."
    )]
    fn a_model_denial_opens_with_the_sentence_clients_match_on(
        #[case] denial: AuthzDenial,
        #[case] opening: &str,
    ) {
        assert!(denial.to_string().starts_with(opening), "{denial}");
    }
}
