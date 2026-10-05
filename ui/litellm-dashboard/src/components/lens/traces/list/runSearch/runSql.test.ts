import { describe, expect, it } from "vitest";

import { parseQuery } from "@/components/shared/search/language";
import { toSearchQuery } from "@/components/shared/search/searchQuery";
import { RUN_QUERY } from "./runQuery";
import { runPredicates, runQueryCommand, runQuerySql, traceQueryCommand } from "./runSql";
import type { RunOrder } from "../runOrder";

const query = (text: string) => toSearchQuery(parseQuery(RUN_QUERY, text));
const predicates = (text: string) => runPredicates(query(text));
const sqlFor = (text: string, range = RANGE, order?: RunOrder): string => {
  const result = runQuerySql(query(text), range, order);
  expect(result.kind).toBe("sql");
  return result.kind === "sql" ? result.sql : result.reason;
};
const RANGE = { startMs: 1_700_000_000_000, endMs: 1_700_003_600_000 };

describe("runPredicates", () => {
  it("turns each run field into a predicate over the logical trace columns", () => {
    expect(predicates("agent:researcher has_error:true model:gpt-5 name:support trace_id:aaa111")).toEqual([
      "arrayExists(x -> x ILIKE 'researcher', agent_names)",
      "has_error = true",
      "arrayExists(x -> x ILIKE 'gpt-5', models)",
      "name ILIKE 'support'",
      "trace_id ILIKE 'aaa111'",
    ]);
  });

  it("negates and globs like the list filter", () => {
    expect(predicates("-model:gpt* input:*vector*")).toEqual([
      "NOT (arrayExists(x -> x ILIKE 'gpt%', models))",
      "input_preview ILIKE '%vector%'",
    ]);
  });

  it("keeps LIKE metacharacters and quotes literal", () => {
    expect(predicates(String.raw`name:"50%_off's"`)).toEqual([String.raw`name ILIKE '50\\%\\_off\'s'`]);
    expect(predicates(String.raw`input:back\slash`)).toEqual([String.raw`input_preview ILIKE 'back\\\\slash'`]);
  });

  it("distinguishes root status from the presence of any error", () => {
    expect(predicates("root_status:ok has_error:true")).toEqual(["root_status ILIKE 'ok'", "has_error = true"]);
    expect(predicates("-has_error:true root_status:unset")).toEqual([
      "NOT (has_error = true)",
      "root_status ILIKE 'unset'",
    ]);
  });

  it("searches free text across trace id, input and name, and skips a key without a value", () => {
    expect(predicates("agent: refund")).toEqual([
      "(trace_id ILIKE '%refund%' OR input_preview ILIKE '%refund%' OR name ILIKE '%refund%')",
    ]);
    expect(predicates("a*b")).toEqual([
      "(trace_id ILIKE '%a*b%' OR input_preview ILIKE '%a*b%' OR name ILIKE '%a*b%')",
    ]);
  });
});

describe("runQuerySql", () => {
  it("queries canonical logical traces within the displayed time window", () => {
    const sql = sqlFor("agent:researcher has_error:true");
    expect(sql).toBe(
      [
        "SELECT id, trace_id, name, input_preview, root_status, has_error,",
        "       agent_names, models, span_count, error_count, start_time, duration_ms",
        "FROM traces",
        `WHERE start_time >= fromUnixTimestamp64Milli(${RANGE.startMs}) AND start_time < fromUnixTimestamp64Milli(${RANGE.endMs})`,
        "   AND arrayExists(x -> x ILIKE 'researcher', agent_names)",
        "   AND has_error = true",
        "ORDER BY start_time DESC, id DESC",
        "LIMIT 100",
      ].join("\n"),
    );
  });

  it.each(["attr.tenant:demo", "unknown:value", "root_status:pending", "root_status:*", "has_error:error"])(
    "does not copy unsupported query %s as SQL",
    (text) => {
      const result = runQuerySql(query(text), RANGE);
      expect(result.kind).toBe("unsupported");
      expect(result.kind === "unsupported" && result.reason).toBeTruthy();
    },
  );

  it("falls back to the last day without a range", () => {
    const result = runQuerySql(query(""));
    expect(result.kind === "sql" && result.sql).toContain("WHERE start_time >= now() - INTERVAL 1 DAY\nORDER BY");
  });

  it.each<{ order: RunOrder; clause: string }>([
    { order: { key: "duration_ms", descending: false }, clause: "ORDER BY duration_ms ASC, id ASC" },
    { order: { key: "span_count", descending: true }, clause: "ORDER BY span_count DESC, id DESC" },
    { order: { key: "error_count", descending: true }, clause: "ORDER BY error_count DESC, id DESC" },
    { order: { key: "start_ms", descending: false }, clause: "ORDER BY start_time ASC, id ASC" },
  ])("orders the copied rows like the list, $clause", ({ order, clause }) => {
    expect(sqlFor("", RANGE, order)).toContain(`\n${clause}\nLIMIT 100`);
    expect(runQueryCommand(RANGE, order)(query(""))).toContain(clause);
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
    const command = runQueryCommand(RANGE)(query("agent:researcher has_error:true"));
    expect(command).toBe(traceQueryCommand(sqlFor("agent:researcher has_error:true")));
    expect(command).toContain("arrayExists(x -> x ILIKE 'researcher', agent_names)");
    expect(command).toContain("has_error = true");
  });
});
