use litellm_traces::QueryScope;
use serde::Serialize;

use crate::TraceTable;

macro_rules! owned_by {
    (otel_traces) => {
        "({access_all:UInt8} = 1 OR ({access_user:String} != '' AND UserId = {access_user:String}) OR has({access_teams:Array(String)}, TeamId))"
    };
    (agent_traces_by_key) => {
        "({access_all:UInt8} = 1 OR ({access_user:String} != '' AND UserIds = [{access_user:String}]) OR has({access_teams:Array(String)}, TeamId))"
    };
    (spend_logs) => {
        "({access_all:UInt8} = 1 OR ({access_user:String} != '' AND user = {access_user:String}) OR has({access_teams:Array(String)}, team_id))"
    };
}

/// Prefixes a read with the rows its caller may see: `owned_spans`, `owned_runs` and
/// `owned_calls`. Trusted SQL reads only these, never the tables.
macro_rules! owned {
    ($($sql:expr),+ $(,)?) => {
        concat!(
            "WITH owned_spans AS (SELECT * FROM otel_traces WHERE ",
            $crate::access::owned_by!(otel_traces),
            "),\nowned_runs AS (SELECT * FROM agent_traces_by_key WHERE ",
            $crate::access::owned_by!(agent_traces_by_key),
            "),\nowned_calls AS (SELECT * FROM spend_logs FINAL WHERE ",
            $crate::access::owned_by!(spend_logs),
            ")",
            $($sql),+
        )
    };
}

pub(crate) use owned;
pub(crate) use owned_by;

#[derive(Debug, Serialize)]
pub(crate) struct AccessParams {
    access_all: u8,
    access_user: String,
    access_teams: Vec<String>,
}

impl From<&QueryScope> for AccessParams {
    fn from(scope: &QueryScope) -> Self {
        match scope {
            QueryScope::All => Self {
                access_all: 1,
                access_user: String::new(),
                access_teams: Vec::new(),
            },
            QueryScope::Owned { user_id, team_ids } => Self {
                access_all: 0,
                access_user: user_id.clone(),
                access_teams: team_ids.clone(),
            },
        }
    }
}

/// The ownership rule of `table` with `scope` written in, for a row policy.
pub(crate) fn predicate(scope: &QueryScope, table: TraceTable) -> String {
    if *scope == QueryScope::All {
        return "1".to_owned();
    }
    let template = match table {
        TraceTable::OtelTraces => owned_by!(otel_traces),
        TraceTable::AgentTracesByKey => owned_by!(agent_traces_by_key),
        TraceTable::SpendLogs => owned_by!(spend_logs),
    };
    let AccessParams {
        access_all,
        access_user,
        access_teams,
    } = scope.into();
    let teams = access_teams
        .iter()
        .map(|team| literal(team))
        .collect::<Vec<_>>()
        .join(", ");
    template
        .replace("{access_all:UInt8}", &access_all.to_string())
        .replace("{access_user:String}", &literal(&access_user))
        .replace(
            "{access_teams:Array(String)}",
            &format!("CAST([{teams}], 'Array(String)')"),
        )
}

fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}
