import { describe, expect, it } from "vitest";
import type { DecisionModelCheckDraft } from "./buildDecisionModelParams";
import {
  buildDecisionTestBody,
  DECISION_TEST_CHIP_TONE,
  DECISION_TEST_HISTORY_CAP,
  decisionModelsForProvider,
  decisionProvidersForGroups,
  decisionTestChip,
  decisionTestOverall,
  newDecisionCheckDraft,
  parseDecisionModelGroups,
  parseDecisionTestResponse,
  prependTestRun,
  runTestShortcutLabel,
  visibleTestResults,
  type DecisionTestRun,
  type DecisionTestScores,
} from "./decisionModelQuestion";

const groups = [
  { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
  { model_group: "clef", providers: ["cloudflare"], mode: "evaluation" },
  { model_group: "gpt-5", providers: ["openai"], mode: "chat" },
  { model_group: "no-provider", mode: "evaluation" },
];

const draft = (overrides: Partial<DecisionModelCheckDraft>): DecisionModelCheckDraft => ({
  id: "1",
  name: "invoice_policy",
  instructions: "Is this about invoices?",
  action: "block",
  threshold: 0.5,
  enabled: true,
  ...overrides,
});

describe("parseDecisionModelGroups", () => {
  it("returns no groups when the response data is not a list", () => {
    expect(parseDecisionModelGroups({})).toEqual([]);
    expect(parseDecisionModelGroups(undefined)).toEqual([]);
  });

  it("skips entries without a model group name and normalizes providers and mode", () => {
    expect(
      parseDecisionModelGroups([
        null,
        { providers: ["typesafe"] },
        { model_group: "jev-latest", providers: ["typesafe", 7], mode: "evaluation" },
        { model_group: "bare", providers: "typesafe", mode: 3 },
      ]),
    ).toEqual([
      { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
      { model_group: "bare", providers: [], mode: null },
    ]);
  });
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

  it("skips a question still missing its name or text, and trims the rest", () => {
    const body = buildDecisionTestBody("jev-latest", "hello", [
      draft({ name: " invoice_policy ", instructions: " Is this about invoices? " }),
      draft({ name: "", instructions: "No name yet" }),
      draft({ name: "refund_policy", instructions: "  " }),
    ]);

    expect(body.questions).toEqual([
      { type: "predicate", name: "invoice_policy", instructions: "Is this about invoices?" },
    ]);
  });
});

describe("newDecisionCheckDraft", () => {
  it("starts a blank enabled Block question at the default threshold", () => {
    const blank: DecisionModelCheckDraft = {
      id: "1",
      name: "",
      instructions: "",
      action: "block",
      threshold: 0.7,
      enabled: true,
    };
    expect(newDecisionCheckDraft([])).toEqual(blank);
  });

  it("gives each new question an id no existing question uses", () => {
    expect(newDecisionCheckDraft([draft({ id: "3" }), draft({ id: "1" })]).id).toBe("4");
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
  });

  it("blocks an unanswered block question, as the saved fail-closed guardrail does", () => {
    expect(decisionTestChip({ kind: "refused" }, draft({ action: "block" }))).toBe("block");
    expect(decisionTestChip({ kind: "missing" }, draft({ action: "block" }))).toBe("block");
    expect(decisionTestChip({ kind: "refused" }, draft({ action: "log" }))).toBe("no_answer");
    expect(decisionTestChip({ kind: "missing" }, draft({ action: "log" }))).toBe("no_answer");
  });
});

describe("DECISION_TEST_CHIP_TONE", () => {
  it("shows Pass in the success tone and Block in the error tone", () => {
    expect(DECISION_TEST_CHIP_TONE.pass).toBe("success");
    expect(DECISION_TEST_CHIP_TONE.block).toBe("error");
    expect(DECISION_TEST_CHIP_TONE.logged).toBe("warning");
    expect(DECISION_TEST_CHIP_TONE.no_answer).toBe("neutral");
  });
});

const scores = (results: DecisionTestScores["results"], checks: DecisionModelCheckDraft[]): DecisionTestScores => ({
  model: "jev-latest",
  asked: Object.fromEntries(checks.map((check) => [check.name, check.instructions])),
  results,
});

describe("decisionTestOverall", () => {
  it("is block when any result would block under current thresholds", () => {
    const checks = [draft({ name: "invoice_policy", threshold: 0.9 }), draft({ name: "jailbreak", threshold: 0.5 })];
    const run = scores(
      {
        invoice_policy: { kind: "probability", probability: 0.6 },
        jailbreak: { kind: "probability", probability: 0.8 },
      },
      checks,
    );
    const raised = checks.map((check) => (check.name === "jailbreak" ? { ...check, threshold: 0.9 } : check));

    expect(decisionTestOverall(visibleTestResults(run, checks, "jev-latest"))).toBe("block");
    expect(decisionTestOverall(visibleTestResults(run, raised, "jev-latest"))).toBe("pass");
  });

  it("is block when a block question got no answer", () => {
    const checks = [draft({ name: "jailbreak" })];
    const run = scores({ jailbreak: { kind: "missing" } }, checks);

    expect(decisionTestOverall(visibleTestResults(run, checks, "jev-latest"))).toBe("block");
  });

  it("ignores a question disabled after the run", () => {
    const checks = [draft({ name: "jailbreak" })];
    const run = scores({ jailbreak: { kind: "probability", probability: 0.99 } }, checks);
    const disabled = [draft({ name: "jailbreak", enabled: false })];

    expect(decisionTestOverall(visibleTestResults(run, disabled, "jev-latest"))).toBe("pass");
  });
});

describe("visibleTestResults", () => {
  it("drops results for questions deleted since the run", () => {
    const checks = [draft({ name: "invoice_policy" }), draft({ name: "deleted_question" })];
    const run = scores(
      {
        invoice_policy: { kind: "probability", probability: 0.9 },
        deleted_question: { kind: "probability", probability: 0.1 },
      },
      checks,
    );

    expect(visibleTestResults(run, [draft({ name: "invoice_policy" })], "jev-latest")).toEqual([
      { check: draft({ name: "invoice_policy" }), result: { kind: "probability", probability: 0.9 } },
    ]);
  });

  it("drops a score whose question text changed since the run, even under the same name", () => {
    const run = scores({ invoice_policy: { kind: "probability", probability: 0.9 } }, [draft({})]);

    expect(visibleTestResults(run, [draft({ instructions: "Does the text ask for a refund?" })], "jev-latest")).toEqual(
      [],
    );
  });

  it("drops every score when the decision model changed since the run", () => {
    const run = scores({ invoice_policy: { kind: "probability", probability: 0.9 } }, [draft({})]);

    expect(visibleTestResults(run, [draft({})], "pplx-decider-v1-27b")).toEqual([]);
  });

  it("keeps a score when only the threshold or action changed", () => {
    const run = scores({ invoice_policy: { kind: "probability", probability: 0.9 } }, [draft({})]);
    const retuned = draft({ threshold: 0.95, action: "log" });

    expect(visibleTestResults(run, [retuned], "jev-latest")).toEqual([
      { check: retuned, result: { kind: "probability", probability: 0.9 } },
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
    const run: DecisionTestRun = { id: 1, input: "new", model: "jev-latest", asked: {}, results: {} };
    const old = Array.from({ length: DECISION_TEST_HISTORY_CAP }, (_, i) => ({
      id: i + 2,
      input: `old ${i}`,
      model: "jev-latest",
      asked: {},
      results: {},
    }));

    const runs = prependTestRun(old, run);

    expect(runs[0]).toBe(run);
    expect(runs).toHaveLength(DECISION_TEST_HISTORY_CAP);
    expect(runs.at(-1)).toEqual(old[DECISION_TEST_HISTORY_CAP - 2]);
  });
});
