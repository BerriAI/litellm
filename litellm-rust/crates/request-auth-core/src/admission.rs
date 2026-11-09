use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum AdmissionFacts {
    CommonGate {
        force_virtual_key_checks: bool,
        public_route: bool,
        additional_public_route: bool,
        unauthenticated_passthrough: bool,
        master_key_present: bool,
        jwt_enabled: bool,
        oauth_enabled: bool,
        oauth_proxy_enabled: bool,
        custom_auth: bool,
        custom_auth_common_checks: bool,
    },
    Reservation {
        skip_budget_checks: bool,
        disabled: bool,
        apply_user_budget_to_team_keys: bool,
        fail_closed: bool,
        billing_agent: bool,
    },
    Ownership {
        virtual_key: bool,
        custom_auth: bool,
        jwt: bool,
        oauth: bool,
    },
}

#[derive(Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum CommonSkip {
    PublicRoute,
    UnauthenticatedPassthrough,
    NoAuthDevelopment,
    CustomAuth,
}

#[derive(Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum Owner {
    Python,
}

#[derive(Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum Candidate {
    VirtualKey,
    CustomAuth,
    Validator,
    Other,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum AdmissionPlan {
    Common {
        skip: Option<CommonSkip>,
    },
    Reserve {
        reserve: bool,
        apply_user_budget_to_team_keys: bool,
        fail_closed: bool,
    },
    Observe {
        owner: Owner,
        candidate: Candidate,
    },
}

pub fn plan(facts: AdmissionFacts) -> AdmissionPlan {
    match facts {
        AdmissionFacts::CommonGate {
            force_virtual_key_checks,
            public_route,
            additional_public_route,
            unauthenticated_passthrough,
            master_key_present,
            jwt_enabled,
            oauth_enabled,
            oauth_proxy_enabled,
            custom_auth,
            custom_auth_common_checks,
        } => {
            let skip = if force_virtual_key_checks {
                None
            } else if public_route || additional_public_route {
                Some(CommonSkip::PublicRoute)
            } else if unauthenticated_passthrough {
                Some(CommonSkip::UnauthenticatedPassthrough)
            } else if !master_key_present && !jwt_enabled && !oauth_enabled && !oauth_proxy_enabled
            {
                Some(CommonSkip::NoAuthDevelopment)
            } else if custom_auth && !custom_auth_common_checks {
                Some(CommonSkip::CustomAuth)
            } else {
                None
            };
            AdmissionPlan::Common { skip }
        }
        AdmissionFacts::Reservation {
            skip_budget_checks,
            disabled,
            apply_user_budget_to_team_keys,
            fail_closed,
            billing_agent,
        } => AdmissionPlan::Reserve {
            reserve: !skip_budget_checks && !disabled,
            apply_user_budget_to_team_keys,
            fail_closed: fail_closed || billing_agent,
        },
        AdmissionFacts::Ownership {
            virtual_key,
            custom_auth,
            jwt,
            oauth,
        } => {
            let candidate = if custom_auth {
                Candidate::CustomAuth
            } else if jwt || oauth {
                Candidate::Validator
            } else if virtual_key {
                Candidate::VirtualKey
            } else {
                Candidate::Other
            };
            AdmissionPlan::Observe {
                owner: Owner::Python,
                candidate,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::normal(false, false, false, true, false)]
    #[case::free(true, false, false, false, false)]
    #[case::disabled(false, true, false, false, false)]
    #[case::managed(false, false, true, true, true)]
    fn reservation(
        #[case] skip: bool,
        #[case] disabled: bool,
        #[case] billing: bool,
        #[case] reserve: bool,
        #[case] closed: bool,
    ) {
        assert_eq!(
            plan(AdmissionFacts::Reservation {
                skip_budget_checks: skip,
                disabled,
                billing_agent: billing,
                fail_closed: false,
                apply_user_budget_to_team_keys: true
            }),
            AdmissionPlan::Reserve {
                reserve,
                apply_user_budget_to_team_keys: true,
                fail_closed: closed
            }
        );
    }

    #[rstest]
    fn ownership_never_transfers_request() {
        assert_eq!(
            plan(AdmissionFacts::Ownership {
                virtual_key: true,
                custom_auth: false,
                jwt: false,
                oauth: false
            }),
            AdmissionPlan::Observe {
                owner: Owner::Python,
                candidate: Candidate::VirtualKey
            }
        );
    }
}
