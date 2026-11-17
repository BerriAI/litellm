import { describe, expect, it } from "vitest";
import { createLiteAdminOperations, type OperationContext } from "./operations";

const makeContext = (): OperationContext => ({
  access: { mode: "admin" },
  accessToken: "test-token",
  signal: new AbortController().signal,
  assertCurrent: () => {},
  beforeTool: () => {},
  confirm: () => Promise.resolve(true),
  onResult: () => {},
  onChart: () => {},
});

describe("createLiteAdminOperations tool parameter schemas", () => {
  const tools = createLiteAdminOperations(makeContext());

  it("exposes tools", () => {
    expect(tools.length).toBeGreaterThan(10);
  });

  it("does not emit a `format: date` tag, which strict providers reject", () => {
    for (const tool of tools) {
      // zodFunction() serializes parameters into a plain JSON Schema object
      expect(JSON.stringify(tool.function.parameters), `tool ${tool.function.name}`).not.toContain('"format":"date"');
    }
  });

  it("keeps date bounds readable as a regex pattern on dated tools", () => {
    const tool = tools.find((t) => t.function.name === "team_spend_report");
    expect(tool).toBeDefined();
    const schema = tool!.function.parameters as {
      properties: Record<string, { pattern?: string }>;
    };
    expect(schema.properties.start_date?.pattern).toMatch(/d\{4\}/);
  });
});
