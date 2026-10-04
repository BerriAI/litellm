import { describe, expect, it } from "vitest";

import { parseQuery } from "@/components/shared/search/language";
import { toSearchQuery } from "@/components/shared/search/searchQuery";
import { RUN_QUERY } from "./runQuery";
import { runPredicates, runQueryCommand, runQuerySql, traceQueryCommand } from "./runSql";

const query = (text: string) => toSearchQuery(parseQuery(RUN_QUERY, text));
const predicates = (text: string) => runPredicates(query(text));
const RANGE = { startMs: 1_700_000_000_000, endMs: 1_700_003_600_000 };

describe("runPredicates", () => {
  it("turns each run field into a predicate over the per-trace rollup columns", () => {
    expect(predicates("agent:researcher status:error model:gpt-5 name:support trace_id:aaa111")).toEqual([
      "arrayExists(x -> x ILIKE 'researcher', agents)",
      "errors > 0",
      "arrayExists(x -> x ILIKE 'gpt-5', models)",
      "name ILIKE 'support'",
      "trace_id ILIKE 'aaa111'",
    ]);
  });

  it("negates and globs like the list filter", () => {
    expect(predicates("-model:gpt* input:*vector*")).toEqual([
      "NOT (arrayExists(x -> x ILIKE 'gpt%', models))",
      "input ILIKE '%vector%'",
    ]);
  });

  it("keeps LIKE metacharacters and quotes literal", () => {
    expect(predicates(String.raw`name:"50%_off's"`)).toEqual([String.raw`name ILIKE '50\\%\\_off\'s'`]);
    expect(predicates(String.raw`input:back\slash`)).toEqual([String.raw`input ILIKE 'back\\\\slash'`]);
  });

  it("resolves a status glob statically, since status has two values", () => {
    expect(predicates("status:ok")).toEqual(["errors = 0"]);
    expect(predicates("status:*")).toEqual(["true"]);
    expect(predicates("-status:pending")).toEqual(["NOT (false)"]);
  });

  it("searches free text across trace id, input and name, and skips a key without a value", () => {
    expect(predicates("agent: refund")).toEqual([
      "(trace_id ILIKE '%refund%' OR input ILIKE '%refund%' OR name ILIKE '%refund%')",
    ]);
    expect(predicates("a*b")).toEqual(["(trace_id ILIKE '%a*b%' OR input ILIKE '%a*b%' OR name ILIKE '%a*b%')"]);
  });
});

describe("runQuerySql", () => {
  it("groups the rollup by trace, bounds the window the list shows and applies the filters", () => {
    const sql = runQuerySql(query("agent:researcher status:error"), RANGE);
    expect(sql).toBe(
      [
        "SELECT TraceId AS trace_id, any(RootName) AS name, any(RootInput) AS input, sum(ErrorCount) AS errors,",
        "       groupUniqArrayArray(AgentNames) AS agents, groupUniqArrayArray(Models) AS models",
        "FROM agent_traces_by_key",
        "GROUP BY TraceId",
        `HAVING min(StartTs) >= fromUnixTimestamp64Milli(${RANGE.startMs}) AND min(StartTs) < fromUnixTimestamp64Milli(${RANGE.endMs})`,
        "   AND arrayExists(x -> x ILIKE 'researcher', agents)",
        "   AND errors > 0",
        "ORDER BY min(StartTs) DESC",
        "LIMIT 100",
      ].join("\n"),
    );
  });

  it("falls back to the last day without a range", () => {
    expect(runQuerySql(query(""))).toContain("HAVING min(StartTs) >= now() - INTERVAL 1 DAY\nORDER BY");
  });
});

describe("traceQueryCommand", () => {
  it("posts the SQL as JSON to the trace query endpoint through a quoted heredoc", () => {
    const sql = `SELECT 1 WHERE name ILIKE 'o\\'reilly' AND note = "q"`;
    const [curl, headers, body, terminator] = traceQueryCommand(sql).split("\n");
    expect(curl).toMatch(/^curl -s ".*\/v1\/traces\/query" \\$/);
    expect(headers).toBe(
      `  -H "Authorization: Bearer $LITELLM_API_KEY" -H "Content-Type: application/json" -d @- <<'EOF'`,
    );
    expect(JSON.parse(body)).toEqual({ sql });
    expect(terminator).toBe("EOF");
  });
});

describe("runQueryCommand", () => {
  it("wraps the bounded, filtered query in the trace query call", () => {
    const command = runQueryCommand(RANGE)(query("agent:researcher status:error"));
    expect(command).toBe(traceQueryCommand(runQuerySql(query("agent:researcher status:error"), RANGE)));
    expect(command).toContain("arrayExists(x -> x ILIKE 'researcher', agents)");
    expect(command).toContain("errors > 0");
  });
});
