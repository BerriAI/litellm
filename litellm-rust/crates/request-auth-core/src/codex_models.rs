use crate::{Error, model_helpers};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Kind {
    Routes,
    NativeRoutes,
    Chatgpt,
    Grants,
}

#[derive(Clone, Default, Deserialize)]
pub struct Candidate {
    pub provider_loaded: bool,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub name: Option<String>,
    pub public_loaded: bool,
    pub public_name: Option<String>,
}

#[derive(Clone, Deserialize)]
pub struct Grants {
    pub key: Vec<String>,
    pub team: Vec<String>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub kind: Kind,
    pub model: Option<String>,
    pub router_present: bool,
    pub team_id: Option<String>,
    pub bundled: Option<Vec<String>>,
    pub priced: Option<bool>,
    pub candidates: Option<Vec<Candidate>>,
    pub alias_loaded: bool,
    pub alias: Option<String>,
    pub grants: Option<Grants>,
    pub prefixed_grants: Option<Grants>,
    pub bare: Vec<model_helpers::Facts>,
    pub prefixed: Vec<model_helpers::Facts>,
    pub route: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum Field {
    Provider,
    Model,
    Name,
    PublicName,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete {
        allowed: bool,
        target: Option<String>,
    },
    ReadBundled,
    ReadCost {
        model: String,
    },
    ReadCandidates {
        model: String,
    },
    ReadAlias {
        model: String,
    },
    ReadCandidate {
        index: usize,
        field: Field,
    },
    ReadGrants {
        prefixed: bool,
    },
    ReadHelper {
        index: usize,
        model: String,
        prefixed: bool,
    },
}

fn complete(allowed: bool) -> Decision {
    Decision::Complete {
        allowed,
        target: None,
    }
}

fn grants(facts: &Facts, model: &str, prefixed: bool) -> Result<Result<bool, Decision>, Error> {
    let selected = if prefixed {
        &facts.prefixed_grants
    } else {
        &facts.grants
    };
    let Some(grants) = selected else {
        return Ok(Err(Decision::ReadGrants { prefixed }));
    };
    let scopes = if prefixed {
        &facts.prefixed
    } else {
        &facts.bare
    };
    for (index, expected) in [&grants.key, &grants.team].into_iter().enumerate() {
        let Some(scope) = scopes.get(index) else {
            return Ok(Err(Decision::ReadHelper {
                index,
                model: model.to_owned(),
                prefixed,
            }));
        };
        if scope.model != model
            || scope.grants.as_ref() != Some(expected)
            || scope.kind != model_helpers::Kind::Helper
        {
            return Err(Error::InvalidAnswer);
        }
        match model_helpers::decide(scope)? {
            model_helpers::Decision::Complete { allowed: false } => return Ok(Ok(false)),
            model_helpers::Decision::Complete { allowed: true } => (),
            _ => return Err(Error::InvalidAnswer),
        }
    }
    Ok(Ok(true))
}

fn candidate(
    candidate: &Candidate,
    index: usize,
    facts: &Facts,
    model: &str,
) -> Result<bool, Decision> {
    let read = |field| Decision::ReadCandidate { index, field };
    if !candidate.provider_loaded {
        return Err(read(Field::Provider));
    }
    let chatgpt = if candidate.provider.as_deref() == Some("chatgpt") {
        true
    } else {
        let backend = candidate
            .model
            .as_deref()
            .ok_or_else(|| read(Field::Model))?;
        backend.split('/').next() == Some("chatgpt")
    };
    if facts.kind == Kind::Chatgpt {
        return Ok(chatgpt);
    }
    if chatgpt {
        return Ok(false);
    }
    let alias = facts
        .alias
        .as_deref()
        .filter(|alias| !alias.is_empty())
        .unwrap_or(model);
    let name = candidate.name.as_deref().ok_or_else(|| read(Field::Name))?;
    if name == model || name == alias {
        return Ok(true);
    }
    if !candidate.public_loaded {
        return Err(read(Field::PublicName));
    }
    Ok(candidate
        .public_name
        .as_deref()
        .is_some_and(|name| name == model || name == alias))
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    if facts.kind == Kind::NativeRoutes
        && facts.route.as_deref() != Some("/backend-api/codex/responses")
    {
        return Ok(complete(false));
    }
    let Some(model) = facts.model.as_deref().filter(|_| facts.router_present) else {
        return Ok(complete(false));
    };
    if facts.kind == Kind::Grants {
        return Ok(match grants(facts, model, false)? {
            Ok(allowed) => complete(allowed),
            Err(read) => read,
        });
    }
    if facts.kind == Kind::Chatgpt {
        if model.starts_with("chatgpt/") {
            return Ok(complete(false));
        }
        match grants(facts, model, false)? {
            Ok(true) => return Ok(complete(false)),
            Err(read) => return Ok(read),
            Ok(false) => (),
        }
    } else {
        let Some(bundled) = &facts.bundled else {
            return Ok(Decision::ReadBundled);
        };
        if bundled.iter().any(|name| name == model) {
            return Ok(complete(false));
        }
        match facts.priced {
            None => {
                return Ok(Decision::ReadCost {
                    model: format!("chatgpt/{model}"),
                });
            }
            Some(true) => return Ok(complete(false)),
            Some(false) => (),
        }
    }
    let target = if facts.kind == Kind::Chatgpt {
        format!("chatgpt/{model}")
    } else {
        model.to_owned()
    };
    let Some(candidates) = &facts.candidates else {
        return Ok(Decision::ReadCandidates { model: target });
    };
    if matches!(facts.kind, Kind::Routes | Kind::NativeRoutes) && !facts.alias_loaded {
        return Ok(Decision::ReadAlias {
            model: model.to_owned(),
        });
    }
    if candidates.is_empty() {
        return Ok(complete(false));
    }
    for (index, value) in candidates.iter().enumerate() {
        match candidate(value, index, facts, model) {
            Ok(false) => return Ok(complete(false)),
            Err(read) => return Ok(read),
            Ok(true) => (),
        }
    }
    if matches!(facts.kind, Kind::Routes | Kind::NativeRoutes) {
        return Ok(complete(true));
    }
    Ok(match grants(facts, &target, true)? {
        Ok(true) => Decision::Complete {
            allowed: true,
            target: Some(target),
        },
        Ok(false) => complete(false),
        Err(read) => read,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn base() -> Facts {
        Facts {
            kind: Kind::Routes,
            model: Some("model".into()),
            router_present: true,
            team_id: None,
            bundled: None,
            priced: None,
            candidates: None,
            alias_loaded: false,
            alias: None,
            grants: None,
            prefixed_grants: None,
            bare: vec![],
            prefixed: vec![],
            route: None,
        }
    }

    #[rstest]
    #[case::no_model(None, true, false)]
    #[case::no_router(Some("model"), false, false)]
    #[case::needs_bundled_catalog(Some("model"), true, true)]
    fn routing_starts_lazily(
        base: Facts,
        #[case] model: Option<&str>,
        #[case] router: bool,
        #[case] read: bool,
    ) {
        let facts = Facts {
            model: model.map(str::to_owned),
            router_present: router,
            ..base
        };
        let expected = if read {
            Decision::ReadBundled
        } else {
            Decision::Complete {
                allowed: false,
                target: None,
            }
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }

    #[rstest]
    #[case::bundled("model", false)]
    #[case::other("other", true)]
    fn bundled_route_stops_before_cost(base: Facts, #[case] bundled: &str, #[case] read: bool) {
        let facts = Facts {
            bundled: Some(vec![bundled.into()]),
            ..base
        };
        let expected = if read {
            Decision::ReadCost {
                model: "chatgpt/model".into(),
            }
        } else {
            Decision::Complete {
                allowed: false,
                target: None,
            }
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }

    #[rstest]
    #[case::alias_pending(false, true)]
    #[case::empty_candidates(true, false)]
    fn empty_candidates_still_read_alias(
        base: Facts,
        #[case] alias_loaded: bool,
        #[case] read: bool,
    ) {
        let facts = Facts {
            bundled: Some(vec![]),
            priced: Some(false),
            candidates: Some(vec![]),
            alias_loaded,
            ..base
        };
        let expected = if read {
            Decision::ReadAlias {
                model: "model".into(),
            }
        } else {
            Decision::Complete {
                allowed: false,
                target: None,
            }
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
