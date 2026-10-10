import { describe, expect, it } from "vitest";
import { buildEditServerPayload } from "./editServerPayload";
import { baseUi } from "./editServerPayload.differential.cases";

const savedEnv = { EXISTING: "saved-value" };
const ui = { ...baseUi, mcpServer: { ...baseUi.mcpServer, transport: "stdio", env: savedEnv } };

describe("stdio edit environment", () => {
  it.each([
    { label: "omitted", envJson: undefined, expected: savedEnv },
    { label: "cleared", envJson: "", expected: {} },
    { label: "empty object", envJson: "{}", expected: {} },
    { label: "replaced", envJson: '{"NEW":"replacement"}', expected: { NEW: "replacement" } },
  ])("should preserve the intended $label environment", ({ envJson, expected }) => {
    const result = buildEditServerPayload({ transport: "stdio", command: "python3", env_json: envJson }, ui);
    expect(result.kind).toBe("ok");
    if (result.kind === "ok") expect(result.payload.env).toEqual(expected);
  });
});
