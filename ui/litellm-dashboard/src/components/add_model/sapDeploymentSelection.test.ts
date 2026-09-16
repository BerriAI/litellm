import { describe, expect, it } from "vitest";
import { buildSapDeploymentSelection, SAP_DEPLOYMENT_MODEL_PREFIX } from "./sapDeploymentSelection";

const DEPLOYMENT = {
  model_name: "anthropic--claude-4.8-opus",
  deployment_url: "https://api.example.com/v2/inference/deployments/d38af17dc133a768",
};

describe("buildSapDeploymentSelection", () => {
  it("prefixes the model name so the proxy routes to the deployment backend", () => {
    const selection = buildSapDeploymentSelection(DEPLOYMENT);
    expect(selection.model).toEqual(["sap/deployment/anthropic--claude-4.8-opus"]);
  });

  it("pins api_base to the discovered deployment url", () => {
    const selection = buildSapDeploymentSelection(DEPLOYMENT);
    expect(selection.api_base).toBe(DEPLOYMENT.deployment_url);
  });

  it("maps the prefixed model to itself so submit sends the deployment form", () => {
    const selection = buildSapDeploymentSelection(DEPLOYMENT);
    expect(selection.model_mappings).toEqual([
      {
        public_name: "sap/deployment/anthropic--claude-4.8-opus",
        litellm_model: "sap/deployment/anthropic--claude-4.8-opus",
      },
    ]);
  });

  it("builds the model string from the exported prefix constant", () => {
    const selection = buildSapDeploymentSelection({ model_name: "gpt-4", deployment_url: "https://x/y" });
    expect(selection.model[0]).toBe(`${SAP_DEPLOYMENT_MODEL_PREFIX}gpt-4`);
  });
});
