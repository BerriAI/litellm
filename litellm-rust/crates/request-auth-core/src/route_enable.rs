use crate::{
    Error, patterns,
    route_access::{self, Classification, RouteContext},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum Feature {
    Admin,
    Llm,
}

#[derive(Clone, Deserialize)]
pub struct Setting {
    pub environment_present: bool,
    pub premium: bool,
    pub secret: Option<bool>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub context: RouteContext,
    pub imported: bool,
    pub failed: Option<Failure>,
    pub admin: Option<Setting>,
    pub llm: Option<Setting>,
    pub exemptions: Option<Vec<String>>,
    pub eu_jev_configured: Option<bool>,
}

#[derive(Clone, Copy, Deserialize, strum::VariantArray)]
pub enum Failure {
    Http,
    Other,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Reason {
    License,
    Disabled,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    PropagateHttp,
    ReadImport,
    ReadExemptions,
    ReadEuJev,
    ReadSettings { feature: Feature },
    ReadSecret { feature: Feature },
    Deny { feature: Feature, reason: Reason },
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum LlmDecision {
    Allow,
    Deny,
    ReadEuJev,
}

pub fn llm_route(context: &RouteContext, configured: Option<bool>) -> Result<LlmDecision, Error> {
    let static_context = RouteContext {
        configured_eu_jev: true,
        ..context.clone()
    };
    if route_access::decide(route_access::Request::Classify {
        context: static_context,
        kind: Classification::Llm,
    })?
    .allowed
    {
        return Ok(LlmDecision::Allow);
    }
    if context.route.trim_end_matches('/') != "/eu_jev/v1/systemone" {
        return Ok(LlmDecision::Deny);
    }
    Ok(match configured {
        None => LlmDecision::ReadEuJev,
        Some(true) => LlmDecision::Deny,
        Some(false) => LlmDecision::Allow,
    })
}

fn setting(setting: Option<&Setting>, feature: Feature) -> Option<Decision> {
    let Some(setting) = setting else {
        return Some(Decision::ReadSettings { feature });
    };
    if setting.environment_present && !setting.premium {
        return Some(Decision::Deny {
            feature,
            reason: Reason::License,
        });
    }
    match setting.secret {
        None => Some(Decision::ReadSecret { feature }),
        Some(true) => Some(Decision::Deny {
            feature,
            reason: Reason::Disabled,
        }),
        Some(false) => None,
    }
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    if let Some(failed) = facts.failed {
        return Ok(match failed {
            Failure::Http => Decision::PropagateHttp,
            Failure::Other => Decision::Allow,
        });
    }
    if !facts.imported {
        return Ok(Decision::ReadImport);
    }
    let management = facts
        .context
        .catalog
        .groups
        .get("management_routes")
        .map(Vec::as_slice)
        .unwrap_or_default();
    if patterns::route_access(&facts.context.route, management)?
        && let Some(decision) = setting(facts.admin.as_ref(), Feature::Admin)
    {
        return Ok(decision);
    }
    match llm_route(&facts.context, facts.eu_jev_configured)? {
        LlmDecision::Deny => return Ok(Decision::Allow),
        LlmDecision::ReadEuJev => return Ok(Decision::ReadEuJev),
        LlmDecision::Allow => (),
    }
    let Some(exemptions) = &facts.exemptions else {
        return Ok(Decision::ReadExemptions);
    };
    if exemptions.contains(&facts.context.route) {
        return Ok(Decision::Allow);
    }
    Ok(setting(facts.llm.as_ref(), Feature::Llm).unwrap_or(Decision::Allow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing(None, LlmDecision::ReadEuJev)]
    #[case::builtin(Some(false), LlmDecision::Allow)]
    #[case::configured(Some(true), LlmDecision::Deny)]
    fn builtin_configuration(#[case] configured: Option<bool>, #[case] expected: LlmDecision) {
        let context = RouteContext {
            route: "/eu_jev/v1/systemone/".into(),
            ..RouteContext::default()
        };
        assert_eq!(llm_route(&context, configured).unwrap(), expected);
    }

    #[rstest]
    #[case::license(true, false, Decision::Deny { feature: Feature::Admin, reason: Reason::License })]
    #[case::absent(false, false, Decision::ReadSecret { feature: Feature::Admin })]
    #[case::premium(true, true, Decision::ReadSecret { feature: Feature::Admin })]
    fn license_precedes_secret(
        #[case] environment_present: bool,
        #[case] premium: bool,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            setting(
                Some(&Setting {
                    environment_present,
                    premium,
                    secret: None
                }),
                Feature::Admin
            ),
            Some(expected)
        );
    }
}
