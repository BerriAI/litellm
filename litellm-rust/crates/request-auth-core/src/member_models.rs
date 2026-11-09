use crate::{Error, model_access};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub user_present: bool,
    pub team_present: bool,
    pub membership_loaded: bool,
    pub grants: Option<Vec<String>>,
    pub scope: Option<model_access::Facts>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny {
        denial: model_access::Decision,
        member_models: Option<Vec<String>>,
    },
    ReadMembership,
    ReadGrants,
    ReadScope,
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    if !facts.user_present || !facts.team_present {
        return Ok(Decision::Allow);
    }
    if !facts.membership_loaded {
        return Ok(Decision::ReadMembership);
    }
    let Some(grants) = &facts.grants else {
        return Ok(Decision::ReadGrants);
    };
    if grants.is_empty() {
        return Ok(Decision::Allow);
    }
    let Some(scope) = &facts.scope else {
        return Ok(Decision::ReadScope);
    };
    let denial = model_access::check(scope)?;
    if denial.allowed() {
        return Ok(Decision::Allow);
    }
    let member_models =
        matches!(denial, model_access::Decision::Deny { .. }).then(|| grants.clone());
    Ok(Decision::Deny {
        denial,
        member_models,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing(false, false, Decision::Allow)]
    #[case::lookup(true, false, Decision::ReadMembership)]
    #[case::preloaded(true, true, Decision::ReadGrants)]
    fn membership_read(
        #[case] user_present: bool,
        #[case] membership_loaded: bool,
        #[case] expected: Decision,
    ) {
        let facts = Facts {
            user_present,
            team_present: true,
            membership_loaded,
            grants: None,
            scope: None,
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
