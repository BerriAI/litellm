import { describe, expect, it } from "vitest";
import {
  buildDecisionTestBody,
  decisionModelsForProvider,
  decisionProvidersForGroups,
  decisionTestOutcome,
  parseDecisionTestResponse,
} from "./decisionModelQuestion";

const groups = [
  { model_group: "jev-latest", providers: ["typesafe"] },
  { model_group: "clef", providers: ["cloudflare"] },
  { model_group: "gpt-5", providers: ["openai"] },
  { model_group: "no-provider" },
];

describe("decisionProvidersForGroups", () => {
  it("keeps only decisions providers that appear on model groups", () => {
    const providers = decisionProvidersForGroups(["typesafe", "openrouter", "cloudflare"], groups);

    expect(providers).toEqual(["typesafe", "cloudflare"]);
  });
});

describe("decisionModelsForProvider", () => {
  it("lists only the model groups on the selected provider, sorted", () => {
    expect(decisionModelsForProvider(groups, "typesafe")).toEqual(["jev-latest"]);
    expect(decisionModelsForProvider(groups, "openrouter")).toEqual([]);
    expect(decisionModelsForProvider(groups, null)).toEqual([]);
  });
});

describe("buildDecisionTestBody", () => {
  it("builds a single-predicate decisions request", () => {
    expect(buildDecisionTestBody("jev-latest", "hello", "invoice_policy", "Is this about invoices?")).toEqual({
      model: "jev-latest",
      input: "hello",
      questions: [{ type: "predicate", name: "invoice_policy", instructions: "Is this about invoices?" }],
    });
  });
});

describe("parseDecisionTestResponse", () => {
  it("reads the probability of the named answer", () => {
    const verdict = parseDecisionTestResponse(
      {
        model: "jev-latest",
        answers: [{ type: "predicate", name: "invoice_policy", probability: 0.87 }],
      },
      "invoice_policy",
    );

    expect(verdict).toEqual({ kind: "probability", probability: 0.87 });
  });

  it("reports a refusal answer", () => {
    const verdict = parseDecisionTestResponse(
      { answers: [{ type: "refusal", name: "invoice_policy" }] },
      "invoice_policy",
    );

    expect(verdict).toEqual({ kind: "refused" });
  });

  it("turns an error body into an error verdict", () => {
    const verdict = parseDecisionTestResponse({ error: { message: "model not found" } }, "invoice_policy");

    expect(verdict).toEqual({ kind: "error", message: "model not found" });
  });
});

describe("decisionTestOutcome", () => {
  it("maps probability and action to the visible outcome", () => {
    const verdict = { kind: "probability" as const, probability: 0.9 };

    expect(decisionTestOutcome(verdict, "block", 0.5)).toBe("would_block");
    expect(decisionTestOutcome(verdict, "log", 0.5)).toBe("would_log");
    expect(decisionTestOutcome({ kind: "probability", probability: 0.3 }, "block", 0.5)).toBe("pass");
    expect(decisionTestOutcome({ kind: "refused" }, "block", 0.5)).toBe("refused");
    expect(decisionTestOutcome({ kind: "error", message: "x" }, "block", 0.5)).toBe("error");
  });
});
