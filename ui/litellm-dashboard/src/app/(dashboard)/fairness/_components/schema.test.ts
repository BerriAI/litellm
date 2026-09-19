import { describe, expect, it } from "vitest";

import {
  buildBody,
  EMPTY_WORKLOAD_CLASS,
  fairnessSettingsSchema,
  settingsToForm,
  type FairnessSettings,
  type FairnessSettingsFormValues,
} from "./schema";

const SERVER_SETTINGS: FairnessSettings = {
  enabled: true,
  workload_classes: [
    { name: "production", reserved_share: 0.6, max_queue_wait_seconds: 30, description: "customer traffic" },
    { name: "batch", reserved_share: 0.1, max_queue_wait_seconds: 120, description: null },
  ],
  default_reserved_share: 0.1,
  default_max_queue_wait_seconds: 5,
  saturation_threshold: 0.8,
  saturation_check_cache_ttl: 30,
  max_queue_depth_per_class: 250,
  queue_poll_interval_seconds: 0.25,
};

const VALID_FORM: FairnessSettingsFormValues = settingsToForm(SERVER_SETTINGS);

const issuesAt = (values: FairnessSettingsFormValues): string[] => {
  const result = fairnessSettingsSchema.safeParse(values);
  return result.success ? [] : result.error.issues.map((issue) => issue.path.join("."));
};

describe("fairness settings form schema", () => {
  it("round-trips server settings through the form without losing anything", () => {
    const parsed = fairnessSettingsSchema.parse(VALID_FORM);
    expect(buildBody(parsed)).toEqual(SERVER_SETTINGS);
  });

  it("shows shares as percentages and stores them back as fractions", () => {
    expect(VALID_FORM.workload_classes[0].reserved_share).toBe("60");
    expect(VALID_FORM.saturation_threshold).toBe("80");
    const parsed = fairnessSettingsSchema.parse({ ...VALID_FORM, saturation_threshold: "12.5" });
    expect(buildBody(parsed).saturation_threshold).toBeCloseTo(0.125);
  });

  it("turns a blank description into null", () => {
    const parsed = fairnessSettingsSchema.parse({
      ...VALID_FORM,
      workload_classes: [{ ...VALID_FORM.workload_classes[0], description: "   " }],
    });
    expect(buildBody(parsed).workload_classes[0].description).toBeNull();
  });

  it("rejects the reserved default pool name and invalid characters", () => {
    expect(issuesAt({ ...VALID_FORM, workload_classes: [{ ...EMPTY_WORKLOAD_CLASS, name: "default" }] })).toEqual([
      "workload_classes.0.name",
    ]);
    expect(issuesAt({ ...VALID_FORM, workload_classes: [{ ...EMPTY_WORKLOAD_CLASS, name: "prod team" }] })).toEqual([
      "workload_classes.0.name",
    ]);
  });

  it("flags the second occurrence of a duplicated class name", () => {
    expect(
      issuesAt({
        ...VALID_FORM,
        workload_classes: [
          { ...EMPTY_WORKLOAD_CLASS, name: "batch" },
          { ...EMPTY_WORKLOAD_CLASS, name: "batch" },
        ],
      }),
    ).toEqual(["workload_classes.1.name"]);
  });

  it("rejects reserved shares that add up to more than 100%", () => {
    expect(
      issuesAt({
        ...VALID_FORM,
        default_reserved_share: "50",
        workload_classes: [{ ...EMPTY_WORKLOAD_CLASS, name: "production", reserved_share: "60" }],
      }),
    ).toEqual(["default_reserved_share"]);
    expect(
      issuesAt({
        ...VALID_FORM,
        default_reserved_share: "40",
        workload_classes: [{ ...EMPTY_WORKLOAD_CLASS, name: "production", reserved_share: "60" }],
      }),
    ).toEqual([]);
  });

  it("enforces the numeric bounds the proxy enforces", () => {
    expect(issuesAt({ ...VALID_FORM, saturation_threshold: "101" })).toEqual(["saturation_threshold"]);
    expect(issuesAt({ ...VALID_FORM, default_max_queue_wait_seconds: "601" })).toEqual([
      "default_max_queue_wait_seconds",
    ]);
    expect(issuesAt({ ...VALID_FORM, queue_poll_interval_seconds: "0" })).toEqual(["queue_poll_interval_seconds"]);
    expect(issuesAt({ ...VALID_FORM, max_queue_depth_per_class: "0" })).toEqual(["max_queue_depth_per_class"]);
    expect(issuesAt({ ...VALID_FORM, max_queue_depth_per_class: "1.5" })).toEqual(["max_queue_depth_per_class"]);
    expect(issuesAt({ ...VALID_FORM, saturation_check_cache_ttl: "" })).toEqual(["saturation_check_cache_ttl"]);
  });
});
