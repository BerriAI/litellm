import { describe, expect, it, vi } from "vitest";
import { createLiteAdminOperations, type AriaAccess, type OperationContext } from "./operations";

const accessModes: AriaAccess[] = [
  { mode: "admin" },
  { mode: "usage", scope: { type: "global" } },
  { mode: "usage", scope: { type: "user", userId: "user" } },
  { mode: "usage", scope: { type: "user_and_teams", userId: "user" } },
];

const operations = (access: AriaAccess) => {
  const context: OperationContext = {
    access,
    accessToken: "test-session",
    signal: new AbortController().signal,
    assertCurrent: vi.fn(),
    beforeTool: vi.fn(),
    confirm: vi.fn(async () => false),
    onResult: vi.fn(),
    onChart: vi.fn(),
  };
  return createLiteAdminOperations(context);
};

function formats(value: unknown): unknown[] {
  if (Array.isArray(value)) return value.flatMap(formats);
  if (value === null || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, child]) => (key === "format" ? [child] : formats(child)));
}

const datedArguments: Record<string, Record<string, unknown>> = {
  spend_report: { group_by: "team", api_key: null, team_id: null, internal_user_id: null, customer_id: null },
  team_spend_report: { team_id: "team" },
  key_spend_report: { api_key: "a".repeat(64) },
  get_usage_data: { user_id: null },
  get_team_usage_data: { team_ids: null },
  get_tag_usage_data: { tags: null },
  request_logs: {
    page: 1,
    page_size: 20,
    request_id: null,
    team_id: null,
    user_id: null,
    model: null,
    status_filter: null,
  },
};

describe.each(accessModes)("Aria tool schemas for %j", (access) => {
  const tools = operations(access);

  it("serializes only provider-compatible formats across every tool", () => {
    for (const tool of tools) {
      const parameters: unknown = JSON.parse(JSON.stringify(tool.function.parameters));
      for (const format of formats(parameters)) {
        // DeepSeek /beta/chat/completions error, observed 2026-11-17: "Invalid tool parameters schema : unknown variant `date`, expected one of `email`, `hostname`, `ipv4`, `ipv6`, `uuid`" (https://github.com/BerriAI/litellm/issues/52184).
        expect(["email", "hostname", "ipv4", "ipv6", "uuid"], tool.function.name).toContain(format);
      }
    }
  });

  const datedTools = tools.filter((tool) => tool.function.name in datedArguments);
  it.each(datedTools)("validates calendar dates through $function.name's parser", (tool) => {
    const args =
      access.mode === "usage" && access.scope.type !== "global" && tool.function.name === "get_usage_data"
        ? {}
        : datedArguments[tool.function.name];
    const parse = (start_date: string, end_date: string) =>
      tool.$parseRaw(JSON.stringify({ ...args, start_date, end_date }));

    expect(parse("2026-02-01", "2026-02-28")).toMatchObject({
      start_date: "2026-02-01",
      end_date: "2026-02-28",
    });
    expect(parse("2024-02-29", "2024-02-29")).toMatchObject({ start_date: "2024-02-29" });
    for (const invalid of ["2026-02-30", "2026-13-01", "next week", ""]) {
      expect(() => parse(invalid, "2026-03-01")).toThrow();
      expect(() => parse("2026-02-01", invalid)).toThrow();
    }
    expect(() => parse("2026-02-28", "2026-02-01")).toThrow();
    expect(() => parse("2024-01-01", "2026-01-01")).toThrow();
  });
});
