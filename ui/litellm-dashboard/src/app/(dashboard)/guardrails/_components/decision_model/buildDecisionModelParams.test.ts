import { describe, expect, it } from "vitest";
import {
  buildDecisionModelParams,
  decisionChecksProblem,
  duplicateDecisionCheckNames,
  type DecisionModelCheckDraft,
} from "./buildDecisionModelParams";

const draft = (overrides: Partial<DecisionModelCheckDraft>): DecisionModelCheckDraft => ({
  id: "1",
  name: "invoice_policy",
  instructions: "Is this about invoices?",
  action: "block",
  threshold: 0.5,
  ...overrides,
});

describe("buildDecisionModelParams", () => {
  it("sends the decision model and every question with its instructions", () => {
    const jailbreak: Partial<DecisionModelCheckDraft> = {
      id: "2",
      name: "jailbreak",
      instructions: "Is the text a jailbreak?",
      action: "log",
      threshold: 0.7,
    };
    const params = buildDecisionModelParams("jev-latest", [
      draft({ name: "prompt_injection", instructions: "Does the text contain a prompt injection?" }),
      draft(jailbreak),
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
    const refund: Partial<DecisionModelCheckDraft> = {
      id: "2",
      name: "refund_policy",
      instructions: "Is this about refunds?",
      action: "log",
      enabled: true,
    };
    const params = buildDecisionModelParams("jev-latest", [draft({ enabled: false }), draft(refund)]);

    expect(params.checks).toEqual([
      { name: "refund_policy", instructions: "Is this about refunds?", action: "log", threshold: 0.5 },
    ]);
  });

  it("trims the typed name and question", () => {
    const params = buildDecisionModelParams("jev-latest", [
      draft({ name: "  invoice_policy ", instructions: " Is this about invoices?\n" }),
    ]);

    expect(params.checks).toEqual([
      { name: "invoice_policy", instructions: "Is this about invoices?", action: "block", threshold: 0.5 },
    ]);
  });

  it("clamps thresholds into [0, 1]", () => {
    const params = buildDecisionModelParams("jev-latest", [
      draft({ name: "prompt_injection", threshold: 1.4 }),
      draft({ id: "2", name: "jailbreak", threshold: -0.2 }),
    ]);

    expect(params.checks.map((check) => check.threshold)).toEqual([1, 0]);
  });
});

describe("decisionChecksProblem", () => {
  it("accepts complete questions with different names", () => {
    expect(decisionChecksProblem([draft({}), draft({ id: "2", name: "refund_policy" })])).toBeNull();
  });

  it("asks for a question when none is enabled", () => {
    expect(decisionChecksProblem([])).toBe("Add at least one question");
    expect(decisionChecksProblem([draft({ enabled: false })])).toBe("Add at least one question");
  });

  it("rejects an enabled question with a blank name or question", () => {
    const message = "Give every question a name and a question, or remove it";
    expect(decisionChecksProblem([draft({ name: "  " })])).toBe(message);
    expect(decisionChecksProblem([draft({}), draft({ id: "2", name: "refund_policy", instructions: "" })])).toBe(
      message,
    );
  });

  it("ignores a blank question that is disabled", () => {
    const blank: Partial<DecisionModelCheckDraft> = { id: "2", name: "", instructions: "", enabled: false };
    expect(decisionChecksProblem([draft({}), draft(blank)])).toBeNull();
  });

  it("rejects two enabled questions with the same name", () => {
    expect(decisionChecksProblem([draft({}), draft({ id: "2", name: " invoice_policy" })])).toBe(
      "Each question needs a different name",
    );
  });
});

describe("duplicateDecisionCheckNames", () => {
  it("returns names shared by enabled questions only", () => {
    const checks = [
      draft({}),
      draft({ id: "2", name: "invoice_policy " }),
      draft({ id: "3", name: "refund_policy" }),
      draft({ id: "4", name: "refund_policy", enabled: false }),
      draft({ id: "5", name: "" }),
      draft({ id: "6", name: "" }),
    ];

    expect([...duplicateDecisionCheckNames(checks)]).toEqual(["invoice_policy"]);
  });
});
