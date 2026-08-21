import { describe, it, expect } from "vitest";
import { buildProjectCreateParams, buildProjectUpdateParams } from "./projectFormUtils";
import { ProjectFormValues } from "./ProjectBaseForm";

const baseValues: ProjectFormValues = {
  project_alias: "My Project",
  team_id: "team-1",
  models: [],
  isBlocked: false,
};

describe("buildProjectCreateParams", () => {
  it("should map basic fields to the API shape", () => {
    const result = buildProjectCreateParams(baseValues);
    expect(result.project_alias).toBe("My Project");
    expect(result.blocked).toBe(false);
    expect(result.models).toEqual([]);
  });

  it("should set blocked=true when isBlocked is true", () => {
    const result = buildProjectCreateParams({ ...baseValues, isBlocked: true });
    expect(result.blocked).toBe(true);
  });

  it("should pass through description when provided", () => {
    const result = buildProjectCreateParams({ ...baseValues, description: "A description" });
    expect(result.description).toBe("A description");
  });

  it("should pass through max_budget when provided", () => {
    const result = buildProjectCreateParams({ ...baseValues, max_budget: 50.0 });
    expect(result.max_budget).toBe(50.0);
  });

  it("should build model_rpm_limit from modelLimits entries", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [{ model: "gpt-4", rpm: 100, tpm: 200 }],
    });
    expect(result.model_rpm_limit).toEqual({ "gpt-4": 100 });
  });

  it("should build model_tpm_limit from modelLimits entries", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [{ model: "gpt-4", rpm: 100, tpm: 200 }],
    });
    expect(result.model_tpm_limit).toEqual({ "gpt-4": 200 });
  });

  it("should build model_itpm_limit from modelLimits entries", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [{ model: "gpt-4", itpm: 150 }],
    });
    expect(result.model_itpm_limit).toEqual({ "gpt-4": 150 });
  });

  it("should build model_otpm_limit from modelLimits entries", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [{ model: "gpt-4", otpm: 250 }],
    });
    expect(result.model_otpm_limit).toEqual({ "gpt-4": 250 });
  });

  it("should map input and output-only model limits independently", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [
        { model: "input-model", itpm: 150 },
        { model: "output-model", otpm: 250 },
      ],
    });
    expect(result.model_itpm_limit).toEqual({ "input-model": 150 });
    expect(result.model_otpm_limit).toEqual({ "output-model": 250 });
    expect(result).not.toHaveProperty("model_tpm_limit");
    expect(result).not.toHaveProperty("model_rpm_limit");
  });

  it("should omit model_rpm_limit when no modelLimits are provided", () => {
    const result = buildProjectCreateParams(baseValues);
    expect(result).not.toHaveProperty("model_rpm_limit");
  });

  it("should omit model_tpm_limit when no modelLimits are provided", () => {
    const result = buildProjectCreateParams(baseValues);
    expect(result).not.toHaveProperty("model_tpm_limit");
  });

  it("should omit input and output TPM limits when no modelLimits are provided", () => {
    const result = buildProjectCreateParams(baseValues);
    expect(result).not.toHaveProperty("model_itpm_limit");
    expect(result).not.toHaveProperty("model_otpm_limit");
  });

  it("should skip a modelLimits entry that has no model name", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [{ model: "", rpm: 100 }],
    });
    expect(result).not.toHaveProperty("model_rpm_limit");
  });

  it("should handle multiple model limit entries", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      modelLimits: [
        { model: "gpt-4", rpm: 100 },
        { model: "gpt-3.5-turbo", tpm: 5000 },
      ],
    });
    expect(result.model_rpm_limit).toEqual({ "gpt-4": 100 });
    expect(result.model_tpm_limit).toEqual({ "gpt-3.5-turbo": 5000 });
  });

  it("should build metadata from key-value entries", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      metadata: [{ key: "env", value: "production" }],
    });
    expect(result.metadata).toEqual({ env: "production" });
  });

  it("should omit metadata when no entries are provided", () => {
    const result = buildProjectCreateParams(baseValues);
    expect(result).not.toHaveProperty("metadata");
  });

  it("should skip metadata entries with no key", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      metadata: [{ key: "", value: "something" }],
    });
    expect(result).not.toHaveProperty("metadata");
  });

  it("should include guardrails as a top-level field when provided", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      guardrails: ["pii-check", "content-filter"],
    });
    expect(result.guardrails).toEqual(["pii-check", "content-filter"]);
  });

  it("should omit guardrails when the array is empty", () => {
    const result = buildProjectCreateParams({
      ...baseValues,
      guardrails: [],
    });
    expect(result).not.toHaveProperty("guardrails");
  });
});

describe("buildProjectUpdateParams clearing", () => {
  it("should send empty limit maps when every model limit row has been removed", () => {
    const result = buildProjectUpdateParams({ ...baseValues, modelLimits: [] });
    expect(result.model_rpm_limit).toEqual({});
    expect(result.model_tpm_limit).toEqual({});
    expect(result.model_itpm_limit).toEqual({});
    expect(result.model_otpm_limit).toEqual({});
  });

  it("should send an empty metadata object when every metadata row has been removed", () => {
    const result = buildProjectUpdateParams({ ...baseValues, metadata: [] });
    expect(result.metadata).toEqual({});
  });

  it("should send an empty input TPM map when the field is blanked on a row that keeps its RPM limit", () => {
    const result = buildProjectUpdateParams({ ...baseValues, modelLimits: [{ model: "gpt-4", rpm: 20 }] });
    expect(result.model_itpm_limit).toEqual({});
    expect(result.model_otpm_limit).toEqual({});
    expect(result.model_rpm_limit).toEqual({ "gpt-4": 20 });
  });

  it("should send an empty output TPM map for whichever row is removed, whatever its position", () => {
    const rows = [
      { model: "input-model", itpm: 150 },
      { model: "output-model", otpm: 250 },
    ];
    expect(buildProjectUpdateParams({ ...baseValues, modelLimits: [rows[0]] }).model_otpm_limit).toEqual({});
    expect(buildProjectUpdateParams({ ...baseValues, modelLimits: [rows[1]] }).model_itpm_limit).toEqual({});
  });

  it("should keep omitting every limit map when the advanced section was never touched", () => {
    const result = buildProjectUpdateParams(baseValues);
    expect(result).not.toHaveProperty("model_rpm_limit");
    expect(result).not.toHaveProperty("model_tpm_limit");
    expect(result).not.toHaveProperty("model_itpm_limit");
    expect(result).not.toHaveProperty("model_otpm_limit");
    expect(result).not.toHaveProperty("metadata");
    expect(result).not.toHaveProperty("guardrails");
  });
});

describe("buildProjectCreateParams vs buildProjectUpdateParams", () => {
  const cleared: ProjectFormValues = { ...baseValues, modelLimits: [], metadata: [], guardrails: [] };

  it("should leave a brand new project free of the empty maps that only exist to clear stored values", () => {
    const result = buildProjectCreateParams(cleared);
    expect(result).not.toHaveProperty("model_itpm_limit");
    expect(result).not.toHaveProperty("model_otpm_limit");
    expect(result).not.toHaveProperty("metadata");
    expect(result).not.toHaveProperty("guardrails");
  });

  it("should send those same empty values on an update so the stored ones are cleared", () => {
    const result = buildProjectUpdateParams(cleared);
    expect(result.model_itpm_limit).toEqual({});
    expect(result.model_otpm_limit).toEqual({});
    expect(result.metadata).toEqual({});
    expect(result.guardrails).toEqual([]);
  });
});
