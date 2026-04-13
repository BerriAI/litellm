import { describe, it, expect } from "vitest";
import { buildProjectApiParams } from "./projectFormUtils";
import { ProjectFormValues } from "./ProjectBaseForm";

const baseValues: ProjectFormValues = {
  project_alias: "My Project",
  team_id: "team-1",
  models: [],
  isBlocked: false,
};

describe("buildProjectApiParams", () => {
  it("should map basic fields to the API shape", () => {
    const result = buildProjectApiParams(baseValues);
    expect(result.project_alias).toBe("My Project");
    expect(result.blocked).toBe(false);
    expect(result.models).toEqual([]);
  });

  it("should set blocked=true when isBlocked is true", () => {
    const result = buildProjectApiParams({ ...baseValues, isBlocked: true });
    expect(result.blocked).toBe(true);
  });

  it("should pass through description when provided", () => {
    const result = buildProjectApiParams({ ...baseValues, description: "A description" });
    expect(result.description).toBe("A description");
  });

  it("should pass through max_budget when provided", () => {
    const result = buildProjectApiParams({ ...baseValues, max_budget: 50.0 });
    expect(result.max_budget).toBe(50.0);
  });

  it("should build model_rpm_limit from modelLimits entries", () => {
    const result = buildProjectApiParams({
      ...baseValues,
      modelLimits: [{ model: "gpt-4", rpm: 100, tpm: 200 }],
    });
    expect(result.model_rpm_limit).toEqual({ "gpt-4": 100 });
  });

  it("should build model_tpm_limit from modelLimits entries", () => {
    const result = buildProjectApiParams({
      ...baseValues,
      modelLimits: [{ model: "gpt-4", rpm: 100, tpm: 200 }],
    });
    expect(result.model_tpm_limit).toEqual({ "gpt-4": 200 });
  });

  it("should omit model_rpm_limit when no modelLimits are provided", () => {
    const result = buildProjectApiParams(baseValues);
    expect(result).not.toHaveProperty("model_rpm_limit");
  });

  it("should omit model_tpm_limit when no modelLimits are provided", () => {
    const result = buildProjectApiParams(baseValues);
    expect(result).not.toHaveProperty("model_tpm_limit");
  });

  it("should skip a modelLimits entry that has no model name", () => {
    const result = buildProjectApiParams({
      ...baseValues,
      modelLimits: [{ model: "", rpm: 100 }],
    });
    expect(result).not.toHaveProperty("model_rpm_limit");
  });

  it("should handle multiple model limit entries", () => {
    const result = buildProjectApiParams({
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
    const result = buildProjectApiParams({
      ...baseValues,
      metadata: [{ key: "env", value: "production" }],
    });
    expect(result.metadata).toEqual({ env: "production" });
  });

  it("should omit metadata when no entries are provided", () => {
    const result = buildProjectApiParams(baseValues);
    expect(result).not.toHaveProperty("metadata");
  });

  it("should skip metadata entries with no key", () => {
    const result = buildProjectApiParams({
      ...baseValues,
      metadata: [{ key: "", value: "something" }],
    });
    expect(result).not.toHaveProperty("metadata");
  });
});
