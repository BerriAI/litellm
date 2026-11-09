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
    ReadSettings { feature: Feature },
    ReadSecret { feature: Feature },
    Deny { feature: Feature, reason: Reason },
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
    if !route_access::decide(route_access::Request::Classify {
        context: facts.context.clone(),
        kind: Classification::Llm,
    })?
    .allowed
    {
        return Ok(Decision::Allow);
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
