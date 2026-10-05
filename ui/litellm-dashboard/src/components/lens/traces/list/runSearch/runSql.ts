import { getProxyBaseUrl } from "@/components/networking";
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import { isNegatedOp } from "@/components/shared/search/language";
import type { SearchFilter, SearchQuery } from "@/components/shared/search/searchQuery";
import { NEWEST, type RunOrder, type RunSortKey } from "../runOrder";
import type { RunField } from "./runQuery";

const RUN_ROWS = `SELECT id, trace_id, name, input_preview, root_status, has_error,
       agent_names, models, span_count, error_count, start_time, duration_ms
FROM traces`;

const ORDER_COLUMNS: Record<RunSortKey, string> = {
  start_ms: "start_time",
  duration_ms: "duration_ms",
  span_count: "span_count",
  error_count: "error_count",
};

/** The list's order with the same tie-break the server pages by. */
const orderBy = (order: RunOrder): string => {
  const direction = order.descending ? "DESC" : "ASC";
  return `ORDER BY ${ORDER_COLUMNS[order.key]} ${direction}, id ${direction}`;
};

const sqlString = (value: string): string => `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;

/** LIKE's own metacharacters match literally. */
const likeLiteral = (value: string): string => value.replace(/[\\%_]/g, (char) => `\\${char}`);
/** `*` is the only glob, as in the list filter. */
const likePattern = (value: string): string => likeLiteral(value).replaceAll("*", "%");

const matches = (column: string, pattern: string): string => `${column} ILIKE ${sqlString(pattern)}`;
const anyMatches = (column: string, pattern: string): string => `arrayExists(x -> ${matches("x", pattern)}, ${column})`;

const FIELD_PREDICATES: Record<RunField, (value: string) => string> = {
  name: (value) => matches("name", likePattern(value)),
  agent: (value) => anyMatches("agent_names", likePattern(value)),
  root_status: (value) => matches("root_status", likePattern(value)),
  has_error: (value) => `has_error = ${value.toLowerCase()}`,
  service: (value) => matches("service", likePattern(value)),
  model: (value) => anyMatches("models", likePattern(value)),
  input: (value) => matches("input_preview", likePattern(value)),
  trace_id: (value) => matches("trace_id", likePattern(value)),
};

const FREE_TEXT_COLUMNS = ["trace_id", "input_preview", "name"] as const;

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
    ? `start_time >= fromUnixTimestamp64Milli(${range.startMs}) AND start_time < fromUnixTimestamp64Milli(${range.endMs})`
    : "start_time >= now() - INTERVAL 1 DAY";

export const runPredicates = (query: SearchQuery<RunField>): string[] => [
  ...query.text.map(textPredicate),
  ...query.filters.map(filterPredicate),
];

export type RunSQL =
  | { readonly kind: "sql"; readonly sql: string }
  | { readonly kind: "unsupported"; readonly reason: string };

const unsupported = (query: SearchQuery<RunField>): string | undefined => {
  if (query.text.some((term) => term.includes(":")))
    return "Use the traces API for attribute or unsupported field filters";
  if (
    query.filters.some(
      (filter) => filter.field === "root_status" && !["ok", "error", "unset"].includes(filter.value.toLowerCase()),
    )
  )
    return "Choose root_status:ok, root_status:error, or root_status:unset";
  if (
    query.filters.some(
      (filter) => filter.field === "has_error" && !["true", "false"].includes(filter.value.toLowerCase()),
    )
  )
    return "Choose has_error:true or has_error:false";
  return undefined;
};

export function runQuerySql(query: SearchQuery<RunField>, range?: TimeWindow, order: RunOrder = NEWEST): RunSQL {
  const reason = unsupported(query);
  if (reason) return { kind: "unsupported", reason };
  const where = [timeBound(range), ...runPredicates(query)].join("\n   AND ");
  return { kind: "sql", sql: `${RUN_ROWS}\nWHERE ${where}\n${orderBy(order)}\nLIMIT 100` };
}

/** Runs `sql` through the trace query API; the quoted heredoc keeps the SQL's own quotes intact. */
export const traceQueryCommand = (sql: string): string =>
  [
    `curl -s "${getProxyBaseUrl().replace(/\/$/, "")}/v1/traces/query" \\`,
    `  -H "Authorization: Bearer $LITELLM_API_KEY" -H "Content-Type: application/json" -d @- <<'EOF'`,
    JSON.stringify({ sql }),
    "EOF",
  ].join("\n");

export const runQueryCommand =
  (range?: TimeWindow, order: RunOrder = NEWEST) =>
  (query: SearchQuery<RunField>): string => {
    const result = runQuerySql(query, range, order);
    return result.kind === "sql" ? traceQueryCommand(result.sql) : result.reason;
  };
