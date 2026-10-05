use litellm_traces::QueryScope;
use serde::Serialize;

use crate::TraceTable;

macro_rules! owned_by {
    (otel_traces) => {
        "({access_all:UInt8} = 1 OR ({access_user:String} != '' AND UserId = {access_user:String}) OR has({access_teams:Array(String)}, TeamId))"
    };
    (spans_core) => {
        "({access_all:UInt8} = 1 OR ({access_user:String} != '' AND UserId = {access_user:String}) OR has({access_teams:Array(String)}, TeamId))"
    };
    (spend_logs) => {
        "({access_all:UInt8} = 1 OR ({access_user:String} != '' AND user = {access_user:String}) OR has({access_teams:Array(String)}, team_id))"
    };
}

/// A user-owned span can belong to a run that other users also wrote to. Trusted reads see a run
/// only if every span they summarize allows it; a row policy cannot express this.
macro_rules! owns_run {
    () => {
        "({access_all:UInt8} = 1 OR has({access_teams:Array(String)}, TeamId) OR ({access_user:String} != '' AND groupUniqArray(UserId) = [{access_user:String}]))"
    };
}

/// Prefixes a read with the rows its caller may see: `owned_spans`, `owned_core` and
/// `owned_calls`. `owned_core` holds the narrow span rows that can belong to a visible run; a read
/// that summarizes runs must read all their rows and keep a run only if it satisfies `owns_run!`.
macro_rules! owned {
    ($($sql:expr),+ $(,)?) => {
        concat!(
            "WITH owned_spans AS (SELECT * FROM otel_traces WHERE ",
            $crate::access::owned_by!(otel_traces),
            "),\nowned_core AS (SELECT * FROM spans_core WHERE ",
            $crate::access::owned_by!(spans_core),
            "),\nowned_calls AS (SELECT * FROM spend_logs FINAL WHERE ",
            $crate::access::owned_by!(spend_logs),
            ")",
            $($sql),+
        )
    };
}

pub(crate) use owned;
pub(crate) use owned_by;
pub(crate) use owns_run;

#[derive(Clone, Debug, Serialize)]
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
        TraceTable::SpansCore => owned_by!(spans_core),
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
