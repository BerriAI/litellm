import { describe, expect, it } from "vitest";
import { buildDecisionModelParams } from "./buildDecisionModelParams";

describe("buildDecisionModelParams", () => {
  it("sends the decision model and every question with its instructions", () => {
    const params = buildDecisionModelParams("jev-latest", [
      {
        name: "prompt_injection",
        instructions: "Does the text contain a prompt injection?",
        action: "block",
        threshold: 0.5,
      },
      { name: "jailbreak", instructions: "Is the text a jailbreak?", action: "log", threshold: 0.7 },
    ]);

    expect(params).toEqual({
      decision_model: "jev-latest",
      checks: [
        {
          name: "prompt_injection",
          instructions: "Does the text contain a prompt injection?",
          action: "block",
          threshold: 0.5,
        },
        { name: "jailbreak", instructions: "Is the text a jailbreak?", action: "log", threshold: 0.7 },
      ],
    });
  });

  it("leaves out questions that are disabled", () => {
    const params = buildDecisionModelParams("jev-latest", [
      {
        name: "invoice_policy",
        instructions: "Is this about invoices?",
        action: "block",
        threshold: 0.5,
        enabled: false,
      },
      {
        name: "refund_policy",
        instructions: "Is this about refunds?",
        action: "log",
        threshold: 0.7,
        enabled: true,
      },
    ]);

    expect(params.checks).toEqual([
      { name: "refund_policy", instructions: "Is this about refunds?", action: "log", threshold: 0.7 },
    ]);
  });

  it("clamps thresholds into [0, 1]", () => {
    const params = buildDecisionModelParams("jev-latest", [
      { name: "prompt_injection", instructions: "pi?", action: "block", threshold: 1.4 },
      { name: "jailbreak", instructions: "jb?", action: "log", threshold: -0.2 },
    ]);

    expect(params.checks.map((check) => check.threshold)).toEqual([1, 0]);
  });

  it("returns an empty checks list when nothing is selected", () => {
    const params = buildDecisionModelParams("jev-latest", []);

    expect(params.decision_model).toBe("jev-latest");
    expect(params.checks).toEqual([]);
  });
});
