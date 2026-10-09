import { describe, expect, it } from "vitest";
import { buildDecisionModelParams } from "./buildDecisionModelParams";

describe("buildDecisionModelParams", () => {
  it("sends the decision model and preset checks without instructions", () => {
    const params = buildDecisionModelParams("jev-latest", [
      { name: "prompt_injection", label: "Prompt injection", action: "block", threshold: 0.5 },
      { name: "jailbreak", label: "Jailbreak", action: "log", threshold: 0.7 },
    ]);

    expect(params).toEqual({
      decision_model: "jev-latest",
      checks: [
        { name: "prompt_injection", action: "block", threshold: 0.5 },
        { name: "jailbreak", action: "log", threshold: 0.7 },
      ],
    });
    expect(params.checks[0]).not.toHaveProperty("instructions");
  });

  it("includes instructions only on custom checks", () => {
    const params = buildDecisionModelParams("jev-latest", [
      {
        name: "invoice_policy",
        instructions: "Is this about invoices?",
        action: "block",
        threshold: 0.5,
        custom: true,
      },
    ]);

    expect(params.checks[0].instructions).toBe("Is this about invoices?");
  });

  it("clamps thresholds into [0, 1]", () => {
    const params = buildDecisionModelParams("jev-latest", [
      { name: "prompt_injection", action: "block", threshold: 1.4 },
      { name: "jailbreak", action: "log", threshold: -0.2 },
    ]);

    expect(params.checks.map((check) => check.threshold)).toEqual([1, 0]);
  });

  it("returns an empty checks list when nothing is selected", () => {
    const params = buildDecisionModelParams("jev-latest", []);

    expect(params.decision_model).toBe("jev-latest");
    expect(params.checks).toEqual([]);
  });
});
