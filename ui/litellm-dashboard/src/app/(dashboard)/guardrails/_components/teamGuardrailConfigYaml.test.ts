import { describe, expect, it } from "vitest";

import { buildEquivalentConfigYaml, defaultUnreachableFallback, type TeamGuardrail } from "./teamGuardrailConfigYaml";

function guardrail(overrides: Partial<TeamGuardrail>): TeamGuardrail {
  return {
    id: "g-1",
    team: "Security",
    name: "agent365-mcp",
    endpoint: "",
    status: "active",
    model: "—",
    forwardKey: false,
    description: "",
    method: "POST",
    customHeaders: [],
    extraHeaders: [],
    submittedAt: "2026-01-01",
    submittedBy: "ops@example.com",
    ...overrides,
  };
}

function fallbackLine(yaml: string): string {
  const line = yaml.split("\n").find((l) => l.includes("unreachable_fallback:"));
  if (line === undefined) throw new Error(`no unreachable_fallback line in:\n${yaml}`);
  return line.trim().split("  #")[0];
}

describe("defaultUnreachableFallback", () => {
  it("fails open for agent_365 and typesafe, closed for everything else including unknown", () => {
    expect(defaultUnreachableFallback("agent_365")).toBe("fail_open");
    expect(defaultUnreachableFallback("typesafe")).toBe("fail_open");
    expect(defaultUnreachableFallback("generic_guardrail_api")).toBe("fail_closed");
    expect(defaultUnreachableFallback("akto")).toBe("fail_closed");
    expect(defaultUnreachableFallback(undefined)).toBe("fail_closed");
  });
});

describe("buildEquivalentConfigYaml unreachable_fallback line", () => {
  it("shows fail_open for an agent_365 guardrail with no explicit fallback", () => {
    const yaml = buildEquivalentConfigYaml(guardrail({ guardrailType: "agent_365" }));
    expect(fallbackLine(yaml)).toBe("unreachable_fallback: fail_open");
    expect(yaml).toContain("        guardrail: agent_365");
  });

  it("shows fail_open for a typesafe guardrail with no explicit fallback", () => {
    expect(fallbackLine(buildEquivalentConfigYaml(guardrail({ guardrailType: "typesafe" })))).toBe(
      "unreachable_fallback: fail_open",
    );
  });

  it("shows fail_closed for a generic guardrail and when the type is unknown", () => {
    expect(fallbackLine(buildEquivalentConfigYaml(guardrail({ guardrailType: "generic_guardrail_api" })))).toBe(
      "unreachable_fallback: fail_closed",
    );
    const untyped = buildEquivalentConfigYaml(guardrail({}));
    expect(fallbackLine(untyped)).toBe("unreachable_fallback: fail_closed");
    expect(untyped).toContain("        guardrail: generic_guardrail_api");
  });

  it("keeps an explicit override over the per-guardrail default", () => {
    expect(
      fallbackLine(
        buildEquivalentConfigYaml(guardrail({ guardrailType: "agent_365", unreachable_fallback: "fail_closed" })),
      ),
    ).toBe("unreachable_fallback: fail_closed");
    expect(
      fallbackLine(buildEquivalentConfigYaml(guardrail({ guardrailType: "akto", unreachable_fallback: "fail_open" }))),
    ).toBe("unreachable_fallback: fail_open");
  });

  it("explains the shown value is the guardrail's default only when nothing was set explicitly", () => {
    const rawLine = (g: TeamGuardrail) =>
      buildEquivalentConfigYaml(g)
        .split("\n")
        .find((l) => l.includes("unreachable_fallback:"));
    expect(rawLine(guardrail({ guardrailType: "agent_365" }))).toBe(
      "        unreachable_fallback: fail_open  # fail_closed blocks, fail_open proceeds when the guardrail endpoint is unreachable. Shown value is this guardrail's default.",
    );
    expect(rawLine(guardrail({ guardrailType: "agent_365", unreachable_fallback: "fail_closed" }))).toBe(
      "        unreachable_fallback: fail_closed  # fail_closed blocks, fail_open proceeds when the guardrail endpoint is unreachable",
    );
  });
});
