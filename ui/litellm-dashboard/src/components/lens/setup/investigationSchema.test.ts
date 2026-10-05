import { describe, expect, it } from "vitest";
import {
  investigationDefaults,
  investigationSchema,
  investigationSettings,
  investigationStepFields,
  SETUP_STEPS,
  type InvestigationInput,
} from "./investigationSchema";
import { watchChecks } from "../model/watches";

const defaults = investigationDefaults(undefined, "new", "traces");

function parse(
  patch: Partial<Omit<InvestigationInput, "selection">> & { selection?: Partial<InvestigationInput["selection"]> } = {},
) {
  return investigationSchema.safeParse({
    ...defaults,
    ...patch,
    selection: { ...defaults.selection, ...patch.selection },
  });
}

function expectIssue(
  patch: Partial<Omit<InvestigationInput, "selection">> & { selection?: Partial<InvestigationInput["selection"]> },
  path: PropertyKey[],
  message: string,
) {
  const result = parse(patch);
  if (result.success) throw new Error("Expected the draft to be invalid");
  expect(result.error.issues).toContainEqual(expect.objectContaining({ path, message }));
}

describe("investigation validation", () => {
  it("attaches incomplete metadata errors to the missing filter field", () => {
    expectIssue(
      { selection: { filters: [{ key: "swarm", value: " " }] } },
      ["selection", "filters", 0, "value"],
      "Choose a key and value for every condition, or remove it",
    );
  });

  it.each([0, 0.5, NaN, Infinity])("rejects an invalid history window: %s", (lookback_hours) => {
    expectIssue(
      { selection: { lookback_hours } },
      ["selection", "lookback_hours"],
      "Choose a time range of at least 1 hour",
    );
  });

  it.each([0, -1, 101, NaN, Infinity])("rejects an invalid sample percentage: %s", (sample_percent) => {
    expectIssue(
      { selection: { sample_percent } },
      ["selection", "sample_percent"],
      "Choose a sampling percentage greater than 0 and up to 100",
    );
  });

  it.each([0, -1, 1.5, NaN])("rejects an invalid maximum: %s", (sample_size) => {
    expectIssue(
      { selection: { sample_size } },
      ["selection", "sample_size"],
      "Choose a positive maximum or leave it blank for no limit",
    );
  });

  it.each([1, 8760, 8761, 100000])(
    "accepts the history window %s with fractional sampling and no maximum",
    (lookback_hours) => {
      const result = parse({ selection: { lookback_hours, sample_percent: 0.01, sample_size: null } });
      expect(result.success).toBe(true);
    },
  );

  it("requires individual runs only on the Run step", () => {
    expectIssue(
      { manualSelection: true },
      ["selection", "execution_ids"],
      "Choose at least one run or turn off individual selection",
    );
    expect(parse({ selection: { execution_ids: ["run"] }, manualSelection: true }).success).toBe(true);
  });

  it("requires expectations from context, presets, or a filled custom check", () => {
    expectIssue({ watching: [] }, ["context"], "Describe the expected behavior or pick something to watch for");
    expect(parse({ watching: [], context: "Expected behavior" }).success).toBe(true);
    const questions = [{ id: "custom", instruction: "Investigate retries", enabled: false }];
    expect(parse({ watching: [], questions }).success).toBe(true);
    expectIssue(
      { questions: [{ ...questions[0], instruction: " x " }] },
      ["questions", 0, "instruction"],
      "Use at least three characters for each check",
    );
  });

  it("validates budget and repeat interval with field-specific paths", () => {
    expectIssue({ budget: Infinity }, ["budget"], "Choose a monthly limit greater than zero");
    expectIssue({ repeat: true, interval: 1.5 }, ["interval"], "Choose a repeat interval of at least 1 minute");
    expect(parse({ repeat: false, interval: 0 }).success).toBe(true);
    expect(parse({ repeat: true, interval: 10080 }).success).toBe(true);
  });

  it("omits blank checks and trims payload text without changing check identity or enabled state", () => {
    const input = {
      ...defaults,
      name: " ",
      context: " Expected behavior \n Keep sources ",
      watching: ["watch_invented"],
      questions: [
        { id: "custom", instruction: "  Find repetitive searches\nInclude retries  ", enabled: false },
        { id: "blank", instruction: " ", enabled: true },
      ],
      selection: { ...defaults.selection, filters: [{ key: " swarm ", value: " research=v2 " }] },
    };
    const draft = investigationSchema.parse(input);
    const saved = investigationSettings(draft, undefined, "analysis");
    const expectedSaved = {
      name: "Find repetitive searches\nInclude retries",
      context: "Expected behavior \n Keep sources",
      model: "analysis",
      concurrency: 8,
      filters: [{ key: "swarm", value: "research=v2" }],
      checks: [
        ...watchChecks(new Set(["watch_invented"])),
        { id: "custom", instruction: "Find repetitive searches\nInclude retries", enabled: false },
      ],
    };
    expect(saved).toMatchObject(expectedSaved);
    expect(input.questions[0].instruction).toBe("  Find repetitive searches\nInclude retries  ");
  });

  it("preserves saved configuration and uses the same edit and duplicate defaults", () => {
    const saved = investigationSettings(investigationSchema.parse(defaults), undefined, "analysis");
    const initial = { ...saved, enabled: true, concurrency: 3, interval_minutes: 15 };
    expect(investigationDefaults(initial, "edit", "requests")).toMatchObject({ repeat: true, interval: 15 });
    const duplicate = investigationDefaults(initial, "duplicate", "requests");
    expect(duplicate.repeat).toBe(false);
    expect(investigationSettings(investigationSchema.parse(duplicate), initial, "analysis")).toMatchObject({
      concurrency: 3,
      interval_minutes: 15,
      enabled: false,
    });
  });
});

describe("investigationStepFields", () => {
  it("assigns every form field to exactly one step", () => {
    const { selection, ...rest } = defaults;
    const formFields = [...Object.keys(rest), ...Object.keys(selection).map((key) => `selection.${key}`)].sort();
    const stepFields = SETUP_STEPS.flatMap((step) => investigationStepFields[step]).sort();
    expect(stepFields).toEqual(formFields);
  });
});
