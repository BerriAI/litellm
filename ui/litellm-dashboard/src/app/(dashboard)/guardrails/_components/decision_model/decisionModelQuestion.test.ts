import { describe, expect, it } from "vitest";
import type { DecisionModelCheckDraft } from "./buildDecisionModelParams";
import {
  buildDecisionTestBody,
  DECISION_TEST_HISTORY_CAP,
  decisionModelsForProvider,
  decisionProvidersForGroups,
  decisionTestChip,
  decisionTestOverall,
  parseDecisionTestResponse,
  prependTestRun,
  runTestShortcutLabel,
  visibleTestResults,
  type DecisionTestRun,
} from "./decisionModelQuestion";

const groups = [
  { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
  { model_group: "clef", providers: ["cloudflare"], mode: "evaluation" },
  { model_group: "gpt-5", providers: ["openai"], mode: "chat" },
  { model_group: "no-provider", mode: "evaluation" },
];

const draft = (overrides: Partial<DecisionModelCheckDraft>): DecisionModelCheckDraft => ({
  name: "invoice_policy",
  instructions: "Is this about invoices?",
  action: "block",
  threshold: 0.5,
  enabled: true,
  ...overrides,
});

describe("decisionProvidersForGroups", () => {
  it("keeps only decisions providers that appear on model groups", () => {
    const providers = decisionProvidersForGroups(["typesafe", "openrouter", "cloudflare"], groups);

    expect(providers).toEqual(["typesafe", "cloudflare"]);
  });

  it("omits a provider whose only groups are chat models", () => {
    const chatOnly = [
      { model_group: "gpt-4o-mini", providers: ["openai"], mode: "chat" },
      { model_group: "gpt-6-luna", providers: ["openai"], mode: "chat" },
    ];

    expect(decisionProvidersForGroups(["openai"], chatOnly)).toEqual([]);
    expect(decisionModelsForProvider(chatOnly, "openai")).toEqual([]);
  });

  it("excludes groups with a null or missing mode", () => {
    const noMode = [
      { model_group: "legacy-eval", providers: ["typesafe"] },
      { model_group: "unset-eval", providers: ["typesafe"], mode: null },
    ];

    expect(decisionProvidersForGroups(["typesafe"], noMode)).toEqual([]);
    expect(decisionModelsForProvider(noMode, "typesafe")).toEqual([]);
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
  it("sends every enabled question with instructions, skipping disabled ones", () => {
    const body = buildDecisionTestBody("jev-latest", "hello", [
      draft({ name: "invoice_policy" }),
      draft({ name: "refund_policy", instructions: "Is this about refunds?", enabled: false }),
      draft({ name: "jailbreak", instructions: "Is this a jailbreak?" }),
    ]);

    expect(body).toEqual({
      model: "jev-latest",
      input: "hello",
      questions: [
        { type: "predicate", name: "invoice_policy", instructions: "Is this about invoices?" },
        { type: "predicate", name: "jailbreak", instructions: "Is this a jailbreak?" },
      ],
    });
  });
});

describe("parseDecisionTestResponse", () => {
  it("maps each requested question to a probability, refusal or missing result", () => {
    const results = parseDecisionTestResponse(
      {
        answers: [
          { type: "predicate", name: "invoice_policy", probability: 0.87 },
          { type: "refusal", name: "jailbreak" },
        ],
      },
      ["invoice_policy", "jailbreak", "other"],
    );

    expect(results).toEqual({
      invoice_policy: { kind: "probability", probability: 0.87 },
      jailbreak: { kind: "refused" },
      other: { kind: "missing" },
    });
  });
});

describe("decisionTestChip", () => {
  it("reads the current action and threshold", () => {
    const result = { kind: "probability" as const, probability: 0.6 };

    expect(decisionTestChip(result, draft({ action: "block", threshold: 0.5 }))).toBe("block");
    expect(decisionTestChip(result, draft({ action: "log", threshold: 0.5 }))).toBe("logged");
    expect(decisionTestChip(result, draft({ action: "block", threshold: 0.9 }))).toBe("pass");
    expect(decisionTestChip({ kind: "refused" }, draft({}))).toBe("no_answer");
    expect(decisionTestChip({ kind: "missing" }, draft({}))).toBe("no_answer");
  });
});

describe("decisionTestOverall", () => {
  it("is block when any result would block under current thresholds", () => {
    const checks = [draft({ name: "invoice_policy", threshold: 0.9 }), draft({ name: "jailbreak", threshold: 0.5 })];

    expect(
      decisionTestOverall(
        {
          invoice_policy: { kind: "probability", probability: 0.6 },
          jailbreak: { kind: "probability", probability: 0.8 },
        },
        checks,
      ),
    ).toBe("block");
    expect(
      decisionTestOverall(
        {
          invoice_policy: { kind: "probability", probability: 0.6 },
          jailbreak: { kind: "probability", probability: 0.8 },
        },
        checks.map((check) => (check.name === "jailbreak" ? { ...check, threshold: 0.9 } : check)),
      ),
    ).toBe("pass");
  });
});

describe("visibleTestResults", () => {
  it("drops results for questions deleted since the run", () => {
    const results = {
      invoice_policy: { kind: "probability" as const, probability: 0.9 },
      deleted_question: { kind: "probability" as const, probability: 0.1 },
    };

    const visible = visibleTestResults(results, [draft({ name: "invoice_policy" })]);

    expect(visible).toEqual([
      { check: draft({ name: "invoice_policy" }), result: { kind: "probability", probability: 0.9 } },
    ]);
  });
});

describe("runTestShortcutLabel", () => {
  it("shows the command key on Mac and Ctrl elsewhere", () => {
    expect(runTestShortcutLabel("MacIntel Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)")).toBe("⌘+Enter");
    expect(runTestShortcutLabel("Win32 Mozilla/5.0 (Windows NT 10.0)")).toBe("Ctrl+Enter");
  });
});

describe("prependTestRun", () => {
  it("prepends the latest run and caps history at 20", () => {
    const run: DecisionTestRun = { id: 1, input: "new", results: {} };
    const old = Array.from({ length: DECISION_TEST_HISTORY_CAP }, (_, i) => ({
      id: i + 2,
      input: `old ${i}`,
      results: {},
    }));

    const runs = prependTestRun(old, run);

    expect(runs[0]).toBe(run);
    expect(runs).toHaveLength(DECISION_TEST_HISTORY_CAP);
    expect(runs.at(-1)).toEqual(old[DECISION_TEST_HISTORY_CAP - 2]);
  });
});
