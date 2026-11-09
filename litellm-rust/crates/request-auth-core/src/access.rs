use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum AccessFacts {
    ConcurrentFailures {
        failures: Vec<bool>,
        terminal: Vec<bool>,
    },
    UserParamRequest {
        enabled: bool,
    },
    ClientTagRequest {
        enabled: bool,
        llm_route: Option<bool>,
    },
    GuardrailRequest {
        requested_keys: Vec<String>,
    },
    TraceRequest {
        required: bool,
    },
    Search {
        name: String,
        allowed: Vec<String>,
        strict: bool,
    },
    TraceGate {
        required: bool,
        method: String,
        llm_route: bool,
        passthrough: bool,
        telemetry: bool,
    },
    TraceStep {
        source: TraceSource,
        value: Option<String>,
        present: bool,
        headers_only: bool,
    },
    Blocked {
        blocked: bool,
    },
    UserParam {
        enabled: bool,
        method: Option<String>,
        llm_route: bool,
        mcp_route: bool,
        has_user: bool,
    },
    ClientTags {
        enabled: bool,
        llm_route: bool,
        has_tags: bool,
    },
    Guardrails {
        requested_keys: Vec<String>,
        modify_guardrails: Option<bool>,
    },
    AgentTrace {
        required: bool,
        header: Option<String>,
    },
    TeamTrace {
        required: bool,
        method: String,
        llm_route: bool,
        passthrough: bool,
        telemetry: bool,
        headers_only: bool,
        header: Option<String>,
        body_trace: Option<String>,
        body_trace_present: bool,
        traceparent_id: Option<String>,
        selected_trace: Option<String>,
        selected_trace_present: bool,
        other_trace: Option<String>,
    },
    ToolRequest {
        token_present: bool,
        call_types: Option<Vec<String>>,
        capable_types: Vec<String>,
    },
    ToolAllowlist {
        key_nonempty_list: bool,
        team_nonempty_list: Option<bool>,
        requested_count: Option<usize>,
    },
    Tools {
        requested: Vec<String>,
        key_allowed: Option<Vec<String>>,
        team_allowed: Option<Vec<String>>,
    },
    GrantLayers {
        deny_by_default: bool,
        identity_present: bool,
        master_key: bool,
        dashboard: bool,
        virtual_key: bool,
        session_token: bool,
        team_present: bool,
    },
    VectorRequest {
        client_present: bool,
        strict: bool,
        ids_read: bool,
        unreadable: bool,
        grants: Option<Vec<Option<Vec<String>>>>,
        requested_count: Option<usize>,
    },
    VectorStores {
        requested: Vec<String>,
        allowed: Option<Vec<String>>,
        strict: bool,
    },
    TeamLookup {
        dashboard: bool,
        missing: bool,
        recorded_models: Vec<String>,
        allow_unavailable: Option<bool>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum Denial {
    Search,
    Blocked,
    UserParam,
    ClientTags,
    Guardrails,
    AgentTrace,
    TeamTrace,
    Tools,
    VectorStores,
    MissingVectorGrant,
    MalformedVectorIds,
    TeamLookup,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, strum::VariantArray)]
pub enum TraceSource {
    Gate,
    Header,
    Body,
    Traceparent,
    Selected,
    Other,
}

#[derive(Debug, Default, PartialEq, Serialize)]
pub struct AccessDecision {
    pub read: Option<AccessRead>,
    pub denial: Option<Denial>,
    pub rejected: Vec<String>,
    pub layers: Vec<String>,
    pub rejected_index: Option<usize>,
    pub next_trace: Option<TraceSource>,
    pub failure_index: Option<usize>,
}

#[derive(Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum AccessRead {
    UserParam,
    ClientTagRoute,
    ClientTags,
    GuardrailPermission,
    UnavailablePolicy,
    ToolRoute,
    ToolAllowlist,
    TeamToolList,
    ToolNames,
    KeyToolGrant,
    TeamToolGrant,
    StrictVectorIds,
    LegacyVectorIds,
    LegacyVectorGrants,
    RagVectorIds,
    VectorGrants,
}

fn modifies_guardrails(keys: &[String]) -> bool {
    keys.iter().any(|key| {
        matches!(
            key.as_str(),
            "guardrails"
                | "disable_global_guardrails"
                | "disable_global_guardrail"
                | "opted_out_global_guardrails"
        )
    })
}

fn deny_if(deny: bool, denial: Denial) -> AccessDecision {
    AccessDecision {
        denial: deny.then_some(denial),
        ..AccessDecision::default()
    }
}

fn nonempty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|value| !value.is_empty())
}

pub fn decide(facts: AccessFacts) -> AccessDecision {
    match facts {
        AccessFacts::VectorRequest {
            client_present,
            strict,
            ids_read,
            unreadable,
            grants,
            requested_count,
        } => {
            if !client_present {
                return AccessDecision::default();
            }
            let read = if !ids_read {
                Some(if strict {
                    AccessRead::StrictVectorIds
                } else {
                    AccessRead::LegacyVectorIds
                })
            } else if unreadable {
                let Some(grants) = grants else {
                    return AccessDecision {
                        read: Some(AccessRead::LegacyVectorGrants),
                        ..AccessDecision::default()
                    };
                };
                return AccessDecision {
                    denial: grants
                        .iter()
                        .flatten()
                        .any(|grant| !grant.is_empty())
                        .then_some(Denial::MalformedVectorIds),
                    ..AccessDecision::default()
                };
            } else {
                match requested_count {
                    None => Some(AccessRead::RagVectorIds),
                    Some(0) => None,
                    Some(_) => Some(AccessRead::VectorGrants),
                }
            };
            AccessDecision {
                read,
                ..AccessDecision::default()
            }
        }
        AccessFacts::ToolRequest {
            token_present,
            call_types,
            capable_types,
        } => AccessDecision {
            read: if !token_present {
                None
            } else {
                match call_types {
                    None => Some(AccessRead::ToolRoute),
                    Some(types) => types
                        .iter()
                        .any(|kind| capable_types.contains(kind))
                        .then_some(AccessRead::ToolAllowlist),
                }
            },
            ..AccessDecision::default()
        },
        AccessFacts::ToolAllowlist {
            key_nonempty_list,
            team_nonempty_list,
            requested_count,
        } => AccessDecision {
            read: if !key_nonempty_list && team_nonempty_list.is_none() {
                Some(AccessRead::TeamToolList)
            } else if !key_nonempty_list && team_nonempty_list != Some(true) {
                None
            } else {
                match requested_count {
                    None => Some(AccessRead::ToolNames),
                    Some(0) => None,
                    Some(_) if key_nonempty_list => Some(AccessRead::KeyToolGrant),
                    Some(_) => Some(AccessRead::TeamToolGrant),
                }
            },
            ..AccessDecision::default()
        },
        AccessFacts::ConcurrentFailures { failures, terminal } => AccessDecision {
            failure_index: terminal
                .iter()
                .position(|failed| *failed)
                .or_else(|| failures.iter().position(|failed| *failed)),
            ..AccessDecision::default()
        },
        AccessFacts::UserParamRequest { enabled } => AccessDecision {
            read: enabled.then_some(AccessRead::UserParam),
            ..AccessDecision::default()
        },
        AccessFacts::ClientTagRequest { enabled, llm_route } => AccessDecision {
            read: match (enabled, llm_route) {
                (false, _) | (_, Some(false)) => None,
                (_, None) => Some(AccessRead::ClientTagRoute),
                (_, Some(true)) => Some(AccessRead::ClientTags),
            },
            ..AccessDecision::default()
        },
        AccessFacts::GuardrailRequest { requested_keys } => AccessDecision {
            read: modifies_guardrails(&requested_keys).then_some(AccessRead::GuardrailPermission),
            ..AccessDecision::default()
        },
        AccessFacts::TraceRequest { required } => AccessDecision {
            next_trace: required.then_some(TraceSource::Gate),
            ..AccessDecision::default()
        },
        AccessFacts::Search {
            name,
            allowed,
            strict,
        } => deny_if(
            (strict || !allowed.is_empty()) && !allowed.contains(&name),
            Denial::Search,
        ),
        AccessFacts::TraceGate {
            required,
            method,
            llm_route,
            passthrough,
            telemetry,
        } => AccessDecision {
            next_trace: (required
                && !matches!(method.as_str(), "GET" | "HEAD" | "OPTIONS")
                && (llm_route || passthrough)
                && !telemetry)
                .then_some(TraceSource::Header),
            ..AccessDecision::default()
        },
        AccessFacts::TraceStep {
            source,
            value,
            present,
            headers_only,
        } => {
            if nonempty(&value) {
                return AccessDecision::default();
            }
            let next_trace = match source {
                TraceSource::Gate => Some(TraceSource::Header),
                TraceSource::Header if headers_only => None,
                TraceSource::Header => Some(TraceSource::Body),
                TraceSource::Body if !present => Some(TraceSource::Traceparent),
                TraceSource::Body | TraceSource::Traceparent => Some(TraceSource::Selected),
                TraceSource::Selected if !present => Some(TraceSource::Other),
                TraceSource::Selected | TraceSource::Other => None,
            };
            AccessDecision {
                denial: next_trace.is_none().then_some(Denial::TeamTrace),
                next_trace,
                ..AccessDecision::default()
            }
        }
        AccessFacts::Blocked { blocked } => deny_if(blocked, Denial::Blocked),
        AccessFacts::UserParam {
            enabled,
            method,
            llm_route,
            mcp_route,
            has_user,
        } => deny_if(
            enabled
                && method.is_some_and(|value| value.to_uppercase() == "POST")
                && llm_route
                && !mcp_route
                && !has_user,
            Denial::UserParam,
        ),
        AccessFacts::ClientTags {
            enabled,
            llm_route,
            has_tags,
        } => deny_if(enabled && llm_route && has_tags, Denial::ClientTags),
        AccessFacts::Guardrails {
            requested_keys,
            modify_guardrails,
        } => deny_if(
            modify_guardrails == Some(false)
                && requested_keys.iter().any(|key| {
                    matches!(
                        key.as_str(),
                        "guardrails"
                            | "disable_global_guardrails"
                            | "disable_global_guardrail"
                            | "opted_out_global_guardrails"
                    )
                }),
            Denial::Guardrails,
        ),
        AccessFacts::AgentTrace { required, header } => {
            deny_if(required && !nonempty(&header), Denial::AgentTrace)
        }
        AccessFacts::TeamTrace {
            required,
            method,
            llm_route,
            passthrough,
            telemetry,
            headers_only,
            header,
            body_trace,
            body_trace_present,
            traceparent_id,
            selected_trace,
            selected_trace_present,
            other_trace,
        } => {
            let gate = required
                && !matches!(method.as_str(), "GET" | "HEAD" | "OPTIONS")
                && (llm_route || passthrough)
                && !telemetry;
            let body = nonempty(&body_trace)
                || (!body_trace_present && nonempty(&traceparent_id))
                || if selected_trace_present {
                    nonempty(&selected_trace)
                } else {
                    nonempty(&other_trace)
                };
            deny_if(
                gate && !nonempty(&header) && (headers_only || !body),
                Denial::TeamTrace,
            )
        }
        AccessFacts::Tools {
            requested,
            key_allowed,
            team_allowed,
        } => {
            let allowed = key_allowed
                .filter(|allowed| !allowed.is_empty())
                .or(team_allowed);
            let rejected: Vec<_> = match allowed {
                Some(allowed) if !allowed.is_empty() => requested
                    .into_iter()
                    .filter(|name| !allowed.contains(name))
                    .collect(),
                _ => Vec::new(),
            };
            AccessDecision {
                denial: (!rejected.is_empty()).then_some(Denial::Tools),
                rejected,
                ..AccessDecision::default()
            }
        }
        AccessFacts::GrantLayers {
            deny_by_default,
            identity_present,
            master_key,
            dashboard,
            virtual_key,
            session_token,
            team_present,
        } => {
            let layers = if !deny_by_default || !identity_present || master_key || dashboard {
                Vec::new()
            } else if !(virtual_key && !session_token) && !team_present {
                vec!["user".into()]
            } else {
                [
                    ("key", virtual_key && !session_token),
                    ("team", team_present),
                ]
                .into_iter()
                .filter_map(|(name, required)| required.then_some(name.into()))
                .collect()
            };
            AccessDecision {
                layers,
                ..AccessDecision::default()
            }
        }
        AccessFacts::VectorStores {
            requested,
            allowed,
            strict,
        } => match allowed {
            None => deny_if(strict, Denial::MissingVectorGrant),
            Some(allowed) if allowed.is_empty() => deny_if(strict, Denial::MissingVectorGrant),
            Some(allowed) => {
                let rejected_index = requested.iter().position(|name| !allowed.contains(name));
                AccessDecision {
                    denial: rejected_index.map(|_| Denial::VectorStores),
                    rejected_index,
                    ..AccessDecision::default()
                }
            }
        },
        AccessFacts::TeamLookup {
            dashboard,
            missing,
            recorded_models,
            allow_unavailable,
        } => {
            if !dashboard && !missing && recorded_models.is_empty() && allow_unavailable.is_none() {
                return AccessDecision {
                    read: Some(AccessRead::UnavailablePolicy),
                    ..AccessDecision::default()
                };
            }
            deny_if(
                !dashboard
                    && (missing || (recorded_models.is_empty() && allow_unavailable != Some(true))),
                Denial::TeamLookup,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::no_database(false, true, false, false, None, None, (None, None))]
    #[case::strict_ids(
        true,
        true,
        false,
        false,
        None,
        None,
        (Some(AccessRead::StrictVectorIds), None)
    )]
    #[case::legacy_ids(
        true,
        false,
        false,
        false,
        None,
        None,
        (Some(AccessRead::LegacyVectorIds), None)
    )]
    #[case::unreadable_grants(
        true,
        false,
        true,
        true,
        None,
        None,
        (Some(AccessRead::LegacyVectorGrants), None)
    )]
    #[case::unrestricted_unreadable(true, false, true, true, Some(vec![None, Some(vec![])]), None, (None, None))]
    #[case::restricted_unreadable(true, false, true, true, Some(vec![Some(vec!["a".into()])]), None, (None, Some(Denial::MalformedVectorIds)))]
    #[case::rag_ids(
        true,
        true,
        true,
        false,
        None,
        None,
        (Some(AccessRead::RagVectorIds), None)
    )]
    #[case::no_ids(true, true, true, false, None, Some(0), (None, None))]
    #[case::identity_grants(
        true,
        true,
        true,
        false,
        None,
        Some(2),
        (Some(AccessRead::VectorGrants), None)
    )]
    fn vector_read_order(
        #[case] client_present: bool,
        #[case] strict: bool,
        #[case] ids_read: bool,
        #[case] unreadable: bool,
        #[case] grants: Option<Vec<Option<Vec<String>>>>,
        #[case] requested_count: Option<usize>,
        #[case] expected: (Option<AccessRead>, Option<Denial>),
    ) {
        assert_eq!(
            decide(AccessFacts::VectorRequest {
                client_present,
                strict,
                ids_read,
                unreadable,
                grants,
                requested_count,
            }),
            AccessDecision {
                read: expected.0,
                denial: expected.1,
                ..AccessDecision::default()
            }
        );
    }

    #[rstest]
    #[case::no_token(false, None, None)]
    #[case::route(true, None, Some(AccessRead::ToolRoute))]
    #[case::no_call_type(true, Some(vec![]), None)]
    #[case::non_tool_route(true, Some(vec!["embedding".into()]), None)]
    #[case::tool_route(true, Some(vec!["embedding".into(), "completion".into()]), Some(AccessRead::ToolAllowlist))]
    fn tool_route_reads(
        #[case] token_present: bool,
        #[case] call_types: Option<Vec<String>>,
        #[case] read: Option<AccessRead>,
    ) {
        assert_eq!(
            decide(AccessFacts::ToolRequest {
                token_present,
                call_types,
                capable_types: vec!["completion".into()],
            })
            .read,
            read
        );
    }

    #[rstest]
    #[case::key_override(true, None, None, Some(AccessRead::ToolNames))]
    #[case::team_fallback(false, None, None, Some(AccessRead::TeamToolList))]
    #[case::no_allowlist(false, Some(false), None, None)]
    #[case::team_enabled(false, Some(true), None, Some(AccessRead::ToolNames))]
    #[case::no_tools(true, None, Some(0), None)]
    #[case::key_grant(true, None, Some(1), Some(AccessRead::KeyToolGrant))]
    #[case::team_grant(false, Some(true), Some(1), Some(AccessRead::TeamToolGrant))]
    fn tool_grant_reads(
        #[case] key_nonempty_list: bool,
        #[case] team_nonempty_list: Option<bool>,
        #[case] requested_count: Option<usize>,
        #[case] read: Option<AccessRead>,
    ) {
        assert_eq!(
            decide(AccessFacts::ToolAllowlist {
                key_nonempty_list,
                team_nonempty_list,
                requested_count,
            })
            .read,
            read
        );
    }

    #[rstest]
    #[case::disabled(false, None, None)]
    #[case::read_route(true, None, Some(AccessRead::ClientTagRoute))]
    #[case::non_llm(true, Some(false), None)]
    #[case::read_tags(true, Some(true), Some(AccessRead::ClientTags))]
    fn client_tag_reads(
        #[case] enabled: bool,
        #[case] llm_route: Option<bool>,
        #[case] read: Option<AccessRead>,
    ) {
        assert_eq!(
            decide(AccessFacts::ClientTagRequest { enabled, llm_route }).read,
            read
        );
    }

    #[rstest]
    #[case::allow(vec![false, false], vec![false, false], None)]
    #[case::scope_priority(vec![true, true], vec![false, false], Some(0))]
    #[case::second_scope(vec![false, true], vec![false, false], Some(1))]
    #[case::terminal_fault(vec![true, true], vec![false, true], Some(1))]
    fn gathered_failures_preserve_scope_order_unless_native_failed(
        #[case] failures: Vec<bool>,
        #[case] terminal: Vec<bool>,
        #[case] expected: Option<usize>,
    ) {
        assert_eq!(
            decide(AccessFacts::ConcurrentFailures { failures, terminal }).failure_index,
            expected
        );
    }

    #[rstest]
    #[case::key_overrides_team(Some(vec!["key".into()]), Some(vec!["team".into()]), vec!["team".into()])]
    #[case::empty_key_inherits(Some(vec![]), Some(vec!["team".into()]), vec!["key".into()])]
    #[case::unrestricted(None, Some(vec![]), vec![])]
    fn tools(
        #[case] key: Option<Vec<String>>,
        #[case] team: Option<Vec<String>>,
        #[case] rejected: Vec<String>,
    ) {
        let result = decide(AccessFacts::Tools {
            requested: vec!["key".into(), "team".into()],
            key_allowed: key,
            team_allowed: team,
        });
        assert_eq!(
            result.denial,
            (!rejected.is_empty()).then_some(Denial::Tools)
        );
        assert_eq!(result.rejected, rejected);
    }

    #[rstest]
    #[case::missing_strict(None, true, Some(Denial::MissingVectorGrant))]
    #[case::empty_legacy(Some(vec![]), false, None)]
    #[case::empty_strict(Some(vec![]), true, Some(Denial::MissingVectorGrant))]
    #[case::denied(Some(vec!["other".into()]), false, Some(Denial::VectorStores))]
    #[case::granted(Some(vec!["store".into()]), true, None)]
    fn vectors(
        #[case] allowed: Option<Vec<String>>,
        #[case] strict: bool,
        #[case] denial: Option<Denial>,
    ) {
        let result = decide(AccessFacts::VectorStores {
            requested: vec!["store".into()],
            allowed,
            strict,
        });
        assert_eq!(result.denial, denial);
    }
}
