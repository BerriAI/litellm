import { describe, expect, it, vi } from "vitest";
import {
  createDefaultStep,
  insertStep,
  removeStep,
  updateStepAtIndex,
  derivePipelineFromPolicy,
  complianceMatchExpected,
  getPromptsForTestSource,
  TEST_SOURCE_QUICK,
  TEST_SOURCE_ALL,
} from "./pipeline_utils";
import type { PipelineStep, Policy } from "./types";
import * as complianceData from "../../data/compliancePrompts";

const makeStep = (guardrail: string, overrides?: Partial<PipelineStep>): PipelineStep => ({
  guardrail,
  on_pass: "next",
  on_fail: "block",
  pass_data: false,
  modify_response_message: null,
  ...overrides,
});

describe("createDefaultStep", () => {
  it("should return a step with empty guardrail and default actions", () => {
    const step = createDefaultStep();
    expect(step).toEqual({
      guardrail: "",
      on_pass: "next",
      on_fail: "block",
      pass_data: false,
      modify_response_message: null,
    });
  });
});

describe("insertStep", () => {
  it("should insert a default step at the given index", () => {
    const steps = [makeStep("a"), makeStep("b")];
    const result = insertStep(steps, 1);
    expect(result).toHaveLength(3);
    expect(result[1].guardrail).toBe("");
    expect(result[0].guardrail).toBe("a");
    expect(result[2].guardrail).toBe("b");
  });

  it("should insert at the beginning when index is 0", () => {
    const steps = [makeStep("a")];
    const result = insertStep(steps, 0);
    expect(result[0].guardrail).toBe("");
    expect(result[1].guardrail).toBe("a");
  });

  it("should not mutate the original array", () => {
    const steps = [makeStep("a")];
    const result = insertStep(steps, 0);
    expect(steps).toHaveLength(1);
    expect(result).toHaveLength(2);
  });
});

describe("removeStep", () => {
  it("should remove the step at the given index", () => {
    const steps = [makeStep("a"), makeStep("b"), makeStep("c")];
    const result = removeStep(steps, 1);
    expect(result).toHaveLength(2);
    expect(result.map((s) => s.guardrail)).toEqual(["a", "c"]);
  });

  it("should not remove if only one step remains", () => {
    const steps = [makeStep("a")];
    const result = removeStep(steps, 0);
    expect(result).toHaveLength(1);
    expect(result[0].guardrail).toBe("a");
  });

  it("should not mutate the original array", () => {
    const steps = [makeStep("a"), makeStep("b")];
    removeStep(steps, 0);
    expect(steps).toHaveLength(2);
  });
});

describe("updateStepAtIndex", () => {
  it("should update only the step at the target index", () => {
    const steps = [makeStep("a"), makeStep("b")];
    const result = updateStepAtIndex(steps, 1, { guardrail: "updated" });
    expect(result[0].guardrail).toBe("a");
    expect(result[1].guardrail).toBe("updated");
  });

  it("should merge partial updates into the existing step", () => {
    const steps = [makeStep("a", { on_pass: "allow" })];
    const result = updateStepAtIndex(steps, 0, { on_fail: "allow" });
    expect(result[0].on_pass).toBe("allow");
    expect(result[0].on_fail).toBe("allow");
    expect(result[0].guardrail).toBe("a");
  });
});

describe("derivePipelineFromPolicy", () => {
  it("should return a default pipeline when policy is null", () => {
    const pipeline = derivePipelineFromPolicy(null);
    expect(pipeline.mode).toBe("pre_call");
    expect(pipeline.steps).toHaveLength(1);
    expect(pipeline.steps[0].guardrail).toBe("");
  });

  it("should return a default pipeline when policy is undefined", () => {
    const pipeline = derivePipelineFromPolicy(undefined);
    expect(pipeline.mode).toBe("pre_call");
    expect(pipeline.steps).toHaveLength(1);
  });

  it("should use the existing pipeline when present", () => {
    const policy: Policy = {
      policy_id: "p1",
      policy_name: "test",
      inherit: null,
      description: null,
      guardrails_add: [],
      guardrails_remove: [],
      condition: null,
      pipeline: {
        mode: "post_call",
        steps: [makeStep("existing-guardrail")],
      },
    };
    const pipeline = derivePipelineFromPolicy(policy);
    expect(pipeline.mode).toBe("post_call");
    expect(pipeline.steps[0].guardrail).toBe("existing-guardrail");
  });

  it("should convert guardrails_add to pipeline steps when no pipeline exists", () => {
    const policy: Policy = {
      policy_id: "p1",
      policy_name: "test",
      inherit: null,
      description: null,
      guardrails_add: ["gA", "gB"],
      guardrails_remove: [],
      condition: null,
    };
    const pipeline = derivePipelineFromPolicy(policy);
    expect(pipeline.steps).toHaveLength(2);
    expect(pipeline.steps[0].guardrail).toBe("gA");
    expect(pipeline.steps[1].guardrail).toBe("gB");
    expect(pipeline.steps[0].on_pass).toBe("next");
    expect(pipeline.steps[0].on_fail).toBe("block");
  });

  it("should return a default pipeline when policy has no pipeline and no guardrails", () => {
    const policy: Policy = {
      policy_id: "p1",
      policy_name: "test",
      inherit: null,
      description: null,
      guardrails_add: [],
      guardrails_remove: [],
      condition: null,
    };
    const pipeline = derivePipelineFromPolicy(policy);
    expect(pipeline.steps).toHaveLength(1);
    expect(pipeline.steps[0].guardrail).toBe("");
  });
});

describe("complianceMatchExpected", () => {
  it("should match 'pass' expectation with 'allow' action", () => {
    expect(complianceMatchExpected("pass", "allow")).toBe(true);
  });

  it("should match 'pass' expectation with 'modify_response' action", () => {
    expect(complianceMatchExpected("pass", "modify_response")).toBe(true);
  });

  it("should not match 'pass' expectation with 'block' action", () => {
    expect(complianceMatchExpected("pass", "block")).toBe(false);
  });

  it("should match 'fail' expectation with 'block' action", () => {
    expect(complianceMatchExpected("fail", "block")).toBe(true);
  });

  it("should not match 'fail' expectation with 'allow' action", () => {
    expect(complianceMatchExpected("fail", "allow")).toBe(false);
  });
});

describe("getPromptsForTestSource", () => {
  it("should return an empty array for quick_chat source", () => {
    expect(getPromptsForTestSource(TEST_SOURCE_QUICK)).toEqual([]);
  });

  it("should return all compliance prompts for __all__ source", () => {
    const mockPrompts = [
      { id: "p1", framework: "fw", category: "c", categoryIcon: "", categoryDescription: "", prompt: "test", expectedResult: "pass" as const },
    ];
    vi.spyOn(complianceData, "getComplianceDatasetPrompts").mockReturnValue(mockPrompts);
    const result = getPromptsForTestSource(TEST_SOURCE_ALL);
    expect(result).toEqual(mockPrompts);
    vi.restoreAllMocks();
  });

  it("should return prompts for a matching framework name", () => {
    const prompt = { id: "p1", framework: "GDPR", category: "c", categoryIcon: "", categoryDescription: "", prompt: "test", expectedResult: "fail" as const };
    vi.spyOn(complianceData, "getFrameworks").mockReturnValue([
      { name: "GDPR", icon: "", description: "", categories: [{ name: "c", icon: "", description: "", prompts: [prompt] }] },
    ]);
    const result = getPromptsForTestSource("GDPR");
    expect(result).toEqual([prompt]);
    vi.restoreAllMocks();
  });

  it("should return an empty array for an unrecognized source", () => {
    vi.spyOn(complianceData, "getFrameworks").mockReturnValue([]);
    expect(getPromptsForTestSource("nonexistent")).toEqual([]);
    vi.restoreAllMocks();
  });
});
