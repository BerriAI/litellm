use crate::{
    Error,
    fallback_access::{self, GroupedScope},
    model_access,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Scope {
    pub user_present: bool,
    pub team_id_present: bool,
    pub team: Option<GroupedScope>,
    pub member_grants: Option<Vec<String>>,
    pub member: Option<model_access::Facts>,
    pub io_failed: bool,
}

#[derive(Deserialize)]
pub enum Request {
    Scope(Box<Scope>),
    Loaded {
        present: Vec<bool>,
        scopes: Vec<Scope>,
    },
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { allowed: bool },
    ReadTeam,
    ReadGroups,
    ReadMembership,
    ReadMember,
    ReadScopes { indices: Vec<usize> },
}

fn scope(facts: Scope) -> Result<Decision, Error> {
    if facts.io_failed {
        return Ok(Decision::Complete { allowed: false });
    }
    let Some(team) = facts.team else {
        return Ok(Decision::ReadTeam);
    };
    let team = fallback_access::grouped(team, true)?;
    if matches!(team, model_access::Decision::ReadGroups) {
        return Ok(Decision::ReadGroups);
    }
    if !team.allowed() {
        return Ok(Decision::Complete { allowed: false });
    }
    if !facts.user_present || !facts.team_id_present {
        return Ok(Decision::Complete { allowed: true });
    }
    let Some(grants) = facts.member_grants else {
        return Ok(Decision::ReadMembership);
    };
    if grants.is_empty() {
        return Ok(Decision::Complete { allowed: true });
    }
    let Some(member) = facts.member else {
        return Ok(Decision::ReadMember);
    };
    Ok(Decision::Complete {
        allowed: model_access::check(&member)?.allowed(),
    })
}

pub fn decide(request: Request) -> Result<Decision, Error> {
    match request {
        Request::Scope(facts) => scope(*facts),
        Request::Loaded { present, scopes } => {
            let indices: Vec<_> = present
                .iter()
                .enumerate()
                .filter_map(|(index, present)| present.then_some(index))
                .collect();
            if scopes.is_empty() && !indices.is_empty() {
                return Ok(Decision::ReadScopes { indices });
            }
            if scopes.len() != indices.len() {
                return Err(Error::InvalidAnswer);
            }
            let results: Result<Vec<_>, _> = scopes
                .into_iter()
                .map(|facts| match scope(facts)? {
                    Decision::Complete { allowed } => Ok(allowed),
                    _ => Err(Error::InvalidAnswer),
                })
                .collect();
            Ok(Decision::Complete {
                allowed: results?.into_iter().any(|allowed| allowed),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::none(vec![], Decision::Complete { allowed: false })]
    #[case::missing(vec![false], Decision::Complete { allowed: false })]
    #[case::read_all(vec![true, false, true], Decision::ReadScopes { indices: vec![0, 2] })]
    fn all_team_reads(#[case] present: Vec<bool>, #[case] expected: Decision) {
        assert_eq!(
            decide(Request::Loaded {
                present,
                scopes: vec![]
            })
            .unwrap(),
            expected
        );
    }
}
