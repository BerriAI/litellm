import { expect, it } from "vitest";
import { jevClassifierConfigSchema } from "./jev_classifier_config";

it("preserves legacy classifier tuning when the backend returns a null deployment reference", () => {
  const legacyConfig = {
    deployment_name: null,
    model: "english",
    timeout_ms: 4100,
  };
  const parsed = jevClassifierConfigSchema.parse(legacyConfig);
  expect(parsed).toMatchObject({ model: "english", timeout_ms: 4100 });
  expect(parsed.deployment_name).toBeUndefined();
});

it("preserves a deployment alias independently of legacy provider model validation", () => {
  const deploymentConfig = {
    deployment_name: "decider",
    model: "custom-deployed-model",
    timeout_ms: 4100,
  };
  const parsed = jevClassifierConfigSchema.parse(deploymentConfig);
  expect(parsed.deployment_name).toBe("decider");
});
