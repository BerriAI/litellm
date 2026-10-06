import { getProxyBaseUrl } from "@/components/networking";
import type { TimeWindow } from "@/components/shared/timeline/Timeline";
import type { TraceQueryBody } from "../../types";

import { isNegatedOp, valueMatcher } from "@/components/shared/search/language";
import type { SearchFilter, SearchQuery } from "@/components/shared/search/searchQuery";
import type { RunField } from "./runQuery";

const RUN_ROWS = `SELECT TraceId AS trace_id, any(RootName) AS name, any(RootInput) AS input, sum(ErrorCount) AS errors,
       if(ifNull(any(RootStatus), '') IN ('STATUS_CODE_ERROR', 'error'), 'error', 'ok') AS status,
       groupUniqArrayArray(AgentNames) AS agents, groupUniqArrayArray(Models) AS models
FROM agent_traces_by_key
GROUP BY TraceId`;

const sqlString = (value: string): string => `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;

/** LIKE's own metacharacters match literally. */
const likeLiteral = (value: string): string => value.replace(/[\\%_]/g, (char) => `\\${char}`);
/** `*` is the only glob, as in the list filter. */
const likePattern = (value: string): string => likeLiteral(value).replaceAll("*", "%");

const matches = (column: string, pattern: string): string => `${column} ILIKE ${sqlString(pattern)}`;
const anyMatches = (column: string, pattern: string): string => `arrayExists(x -> ${matches("x", pattern)}, ${column})`;

const STATUS_PREDICATES = { error: "status = 'error'", ok: "status = 'ok'" } as const;

/** Status has two values, so a glob over it resolves statically to one, both or neither predicate. */
function statusPredicate(value: string): string {
  const hits = (["error", "ok"] as const).filter(valueMatcher(value)).map((status) => STATUS_PREDICATES[status]);
  if (hits.length === 2) return "true";
  return hits[0] ?? "false";
}

const FIELD_PREDICATES: Record<RunField, (value: string) => string> = {
  name: (value) => matches("name", likePattern(value)),
  agent: (value) => anyMatches("agents", likePattern(value)),
  status: statusPredicate,
  model: (value) => anyMatches("models", likePattern(value)),
  input: (value) => matches("input", likePattern(value)),
  trace_id: (value) => matches("trace_id", likePattern(value)),
};

const FREE_TEXT_COLUMNS = ["trace_id", "input", "name"] as const;

const textPredicate = (term: string): string => {
  const contains = `%${likeLiteral(term)}%`;
  return `(${FREE_TEXT_COLUMNS.map((column) => matches(column, contains)).join(" OR ")})`;
};

function filterPredicate(filter: SearchFilter<RunField>): string {
  const predicate = FIELD_PREDICATES[filter.field](filter.value);
  return isNegatedOp(filter.op) ? `NOT (${predicate})` : predicate;
}

const timeBound = (range: TimeWindow | undefined): string =>
  range
    ? `min(StartTs) >= fromUnixTimestamp64Milli(${range.startMs}) AND min(StartTs) < fromUnixTimestamp64Milli(${range.endMs})`
    : "min(StartTs) >= now() - INTERVAL 1 DAY";

export const runPredicates = (query: SearchQuery<RunField>): string[] => [
  ...query.text.map(textPredicate),
  ...query.filters.map(filterPredicate),
];

/** The runs list as a trace query: one row per trace from the per-key rollup, filtered like the list. */
export function runQuerySql(query: SearchQuery<RunField>, range?: TimeWindow): string {
  const having = [timeBound(range), ...runPredicates(query)].join("\n   AND ");
  return `${RUN_ROWS}\nHAVING ${having}\nORDER BY min(StartTs) DESC\nLIMIT 100`;
}

/** Runs `sql` through the trace query API; the quoted heredoc keeps the SQL's own quotes intact. */
export const traceQueryCommand = (sql: string): string =>
  [
    `curl -s "${getProxyBaseUrl().replace(/\/$/, "")}/v1/traces/query" \\`,
    `  -H "Authorization: Bearer $LITELLM_API_KEY" -H "Content-Type: application/json" -d @- <<'EOF'`,
    JSON.stringify({ sql } satisfies TraceQueryBody),
    "EOF",
  ].join("\n");

export const runQueryCommand =
  (range?: TimeWindow) =>
  (query: SearchQuery<RunField>): string =>
    traceQueryCommand(runQuerySql(query, range));
