use crate::{Error, codex_models, codex_turn, route_access::RouteContext, route_enable};
use serde::{Deserialize, Serialize};

const NATIVE_ROUTE: &str = "/backend-api/codex/responses";

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Kind {
    Alias,
    Prefix,
}

#[derive(Deserialize)]
pub struct Facts {
    pub kind: Kind,
    pub route: String,
    pub request_present: bool,
    pub router_present: Option<bool>,
    pub context: Option<RouteContext>,
    pub eu_jev: Option<bool>,
    pub resolved: Option<bool>,
    pub marked: bool,
    pub model_loaded: bool,
    pub model: Option<String>,
    pub explicit: Option<bool>,
    pub provider: Option<bool>,
    pub routing: Option<codex_models::Facts>,
    pub body_present: Option<bool>,
    pub settings_loaded: bool,
    pub target: Option<String>,
    pub provider_after: Option<bool>,
    pub turn: Option<codex_turn::Facts>,
    pub prefix: Option<codex_models::Facts>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete,
    ReadRouter,
    ReadContext,
    ReadEuJev,
    ReadResolved,
    MarkResolved,
    ReadModel,
    ReadExplicit,
    ReadProvider,
    ReadRouting,
    ReadBody,
    ReadSettings,
    ReadProviderAfter,
    ReadTurn,
    ReadPrefix,
    Rewrite { target: String },
}

fn routed(facts: &codex_models::Facts) -> Result<bool, Error> {
    match codex_models::decide(facts)? {
        codex_models::Decision::Complete { allowed, .. } => Ok(allowed),
        _ => Err(Error::InvalidAnswer),
    }
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    if !facts.request_present {
        return Ok(Decision::Complete);
    }
    if facts.kind == Kind::Prefix && (facts.route != NATIVE_ROUTE || facts.model.is_none()) {
        return Ok(Decision::Complete);
    }
    match facts.router_present {
        None => return Ok(Decision::ReadRouter),
        Some(false) => return Ok(Decision::Complete),
        Some(true) => (),
    }
    if facts.kind == Kind::Prefix {
        let Some(prefix) = &facts.prefix else {
            return Ok(Decision::ReadPrefix);
        };
        return match codex_models::decide(prefix)? {
            codex_models::Decision::Complete {
                target: Some(target),
                ..
            } => Ok(Decision::Rewrite { target }),
            codex_models::Decision::Complete { .. } => Ok(Decision::Complete),
            _ => Err(Error::InvalidAnswer),
        };
    }
    let Some(context) = &facts.context else {
        return Ok(Decision::ReadContext);
    };
    match route_enable::llm_route(context, facts.eu_jev)? {
        route_enable::LlmDecision::Deny => return Ok(Decision::Complete),
        route_enable::LlmDecision::ReadEuJev => return Ok(Decision::ReadEuJev),
        route_enable::LlmDecision::Allow => (),
    }
    match facts.resolved {
        None => return Ok(Decision::ReadResolved),
        Some(true) => return Ok(Decision::Complete),
        Some(false) => (),
    }
    if !facts.marked {
        return Ok(Decision::MarkResolved);
    }
    if !facts.model_loaded {
        return Ok(Decision::ReadModel);
    }
    match facts.explicit {
        None => return Ok(Decision::ReadExplicit),
        Some(true) => return Ok(Decision::Complete),
        Some(false) => (),
    }
    let Some(provider) = facts.provider else {
        return Ok(Decision::ReadProvider);
    };
    if provider {
        if facts.route != NATIVE_ROUTE {
            return Ok(Decision::Complete);
        }
        let Some(routing) = &facts.routing else {
            return Ok(Decision::ReadRouting);
        };
        if !routed(routing)? {
            return Ok(Decision::Complete);
        }
    }
    let Some(model) = &facts.model else {
        return Ok(Decision::Complete);
    };
    match facts.body_present {
        None => return Ok(Decision::ReadBody),
        Some(false) => return Ok(Decision::Complete),
        Some(true) => (),
    }
    if !facts.settings_loaded {
        return Ok(Decision::ReadSettings);
    }
    let Some(target) = facts
        .target
        .as_ref()
        .filter(|target| !target.is_empty() && *target != model)
    else {
        return Ok(Decision::Complete);
    };
    let Some(provider_after) = facts.provider_after else {
        return Ok(Decision::ReadProviderAfter);
    };
    if provider_after {
        let Some(turn) = &facts.turn else {
            return Ok(Decision::ReadTurn);
        };
        match codex_turn::turn(turn)? {
            codex_turn::Decision::Complete { allowed: true } => (),
            codex_turn::Decision::Complete { allowed: false } => return Ok(Decision::Complete),
            _ => return Err(Error::InvalidAnswer),
        }
    }
    Ok(Decision::Rewrite {
        target: target.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing_request(false, NATIVE_ROUTE, Some("model"), Decision::Complete)]
    #[case::wrong_route(true, "/v1/responses", Some("model"), Decision::Complete)]
    #[case::non_string(true, NATIVE_ROUTE, None, Decision::Complete)]
    #[case::native(true, NATIVE_ROUTE, Some("model"), Decision::ReadRouter)]
    fn prefix_does_not_read_router_on_early_exit(
        #[case] request: bool,
        #[case] route: &str,
        #[case] model: Option<&str>,
        #[case] expected: Decision,
    ) {
        let facts = Facts {
            kind: Kind::Prefix,
            route: route.into(),
            request_present: request,
            router_present: None,
            context: None,
            eu_jev: None,
            resolved: None,
            marked: false,
            model_loaded: true,
            model: model.map(str::to_owned),
            explicit: None,
            provider: None,
            routing: None,
            body_present: None,
            settings_loaded: false,
            target: None,
            provider_after: None,
            turn: None,
            prefix: None,
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
