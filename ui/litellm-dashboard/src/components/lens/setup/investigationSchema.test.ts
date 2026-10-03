import { describe, expect, it } from "vitest";
import { investigationDefaults, investigationSchema, investigationSettings } from "./investigationSchema";
import { watchChecks } from "./watches";

const defaults = investigationDefaults(undefined, "new", "traces");

function validationMessage(patch: Partial<typeof defaults>, step = 2) {
  const result = investigationSchema(step).safeParse({ ...defaults, ...patch });
  return result.success ? null : result.error.issues[0].message;
}

describe("investigation validation", () => {
  it("rejects incomplete metadata before changing steps", () => {
    const selection = { ...defaults.selection, filters: [{ key: "swarm", value: " " }] };
    expect(validationMessage({ selection }, 0)).toBe("Choose a key and value for every condition, or remove it");
  });

  it.each([0, 0.5, 8761, NaN, Infinity])("rejects an invalid history window: %s", (lookback_hours) => {
    const selection = { ...defaults.selection, lookback_hours };
    expect(validationMessage({ selection }, 0)).toBe("Choose a time range between 1 hour and 365 days");
  });

  it.each([0, -1, 101, NaN, Infinity])("rejects an invalid sample percentage: %s", (sample_percent) => {
    const selection = { ...defaults.selection, sample_percent };
    expect(validationMessage({ selection })).toBe("Choose a sampling percentage greater than 0 and up to 100");
  });

  it.each([0, -1, 1.5, NaN])("rejects an invalid maximum: %s", (sample_size) => {
    const selection = { ...defaults.selection, sample_size };
    expect(validationMessage({ selection })).toBe("Choose a positive maximum or leave it blank for no limit");
  });

  it.each([1, 8760])("accepts the history boundary %s with fractional sampling and no maximum", (lookback_hours) => {
    const selection = { ...defaults.selection, lookback_hours, sample_percent: 0.01, sample_size: null };
    expect(validationMessage({ selection })).toBeNull();
  });

  it("requires individual runs only on the Run step", () => {
    expect(validationMessage({ manualSelection: true }, 1)).toBeNull();
    expect(validationMessage({ manualSelection: true })).toBe(
      "Choose at least one run or turn off individual selection",
    );
    const selection = { ...defaults.selection, execution_ids: ["run"] };
    expect(validationMessage({ selection, manualSelection: true })).toBeNull();
  });

  it("requires expectations from context, presets, or a filled custom check", () => {
    expect(validationMessage({ watching: [] }, 0)).toBeNull();
    expect(validationMessage({ watching: [] }, 1)).toBe(
      "Describe the expected behavior or pick something to watch for",
    );
    expect(validationMessage({ watching: [], context: "Expected behavior" })).toBeNull();
    const questions = [{ id: "custom", instruction: "Investigate retries", enabled: false }];
    expect(validationMessage({ watching: [], questions })).toBeNull();
    expect(validationMessage({ questions: [{ ...questions[0], instruction: " x " }] }, 0)).toBe(
      "Use at least three characters for each check",
    );
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
    const draft = investigationSchema(2).parse(input);
    const saved = investigationSettings(draft, undefined, "analysis");
    expect(saved).toMatchObject({
      name: "Find repetitive searches\nInclude retries",
      context: "Expected behavior \n Keep sources",
      model: "analysis",
      concurrency: 8,
      filters: [{ key: "swarm", value: "research=v2" }],
      checks: [
        ...watchChecks(new Set(["watch_invented"])),
        { id: "custom", instruction: "Find repetitive searches\nInclude retries", enabled: false },
      ],
    });
    expect(input.questions[0].instruction).toBe("  Find repetitive searches\nInclude retries  ");
  });

  it("preserves saved configuration and uses the same edit and duplicate defaults", () => {
    const saved = investigationSettings(investigationSchema(2).parse(defaults), undefined, "analysis");
    const initial = { ...saved, enabled: true, concurrency: 3, interval_minutes: 15 };
    expect(investigationDefaults(initial, "edit", "requests")).toMatchObject({ repeat: true, interval: 15 });
    const duplicate = investigationDefaults(initial, "duplicate", "requests");
    expect(duplicate.repeat).toBe(false);
    expect(investigationSettings(investigationSchema(2).parse(duplicate), initial, "analysis")).toMatchObject({
      concurrency: 3,
      interval_minutes: 15,
      enabled: false,
    });
  });
});
