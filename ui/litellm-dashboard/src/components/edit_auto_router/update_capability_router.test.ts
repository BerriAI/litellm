import { beforeEach, describe, expect, it, vi } from "vitest";
import { type CapabilityRouterConfigValue } from "../add_model/capability_router_config";

const { modelPatchUpdateCall, validateAutoRouterConfig } = vi.hoisted(() => ({
  modelPatchUpdateCall: vi.fn(),
  validateAutoRouterConfig: vi.fn(),
}));

vi.mock("../networking", () => ({ modelPatchUpdateCall, validateAutoRouterConfig }));

import { updateCapabilityRouter } from "./update_capability_router";

const config: CapabilityRouterConfigValue = {
  candidates: [
    { model: "economy", description: "Bounded tasks" },
    { model: "frontier", description: "Open-ended tasks" },
  ],
  classifier: { model: "classifier", timeout_ms: 3000, max_output_tokens: 1024 },
  probability_threshold: 0.7,
  fallback_model: "frontier",
  estimated_output_tokens: 1000,
  cache_ttl_seconds: 3600,
};

const modelData = {
  model_name: "router",
  litellm_params: { model: "auto_router/capability_router", timeout: 30 },
  model_info: { id: "model-id", team_id: "team-id", access_groups: ["old"] },
};

describe("updateCapabilityRouter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    validateAutoRouterConfig.mockResolvedValue({ valid: true });
    modelPatchUpdateCall.mockResolvedValue(undefined);
  });

  it("validates and patches the complete capability router", async () => {
    const params = {
      accessToken: "token",
      config,
      modelData,
      values: { auto_router_name: "renamed", model_access_group: ["new"] },
    };
    const result = await updateCapabilityRouter(params);

    expect(validateAutoRouterConfig).toHaveBeenCalledWith("token", config, "team-id", "capability");
    expect(modelPatchUpdateCall).toHaveBeenCalledWith(
      "token",
      expect.objectContaining({
        model_name: "renamed",
        litellm_params: { model: "auto_router/capability_router", timeout: 30, capability_router_config: config },
        model_info: { id: "model-id", team_id: "team-id", access_groups: ["new"] },
      }),
      "model-id",
    );
    expect(result).toEqual({
      kind: "success",
      updatedModel: expect.objectContaining({ model_name: "renamed" }),
    });
  });

  it("does not call the proxy when local validation fails", async () => {
    const params = {
      accessToken: "token",
      config: { ...config, fallback_model: "missing" },
      modelData,
      values: { auto_router_name: "router" },
    };
    const result = await updateCapabilityRouter(params);

    expect(result).toEqual({ kind: "error", message: "Select one candidate as the fallback model" });
    expect(validateAutoRouterConfig).not.toHaveBeenCalled();
    expect(modelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("does not patch a config rejected by the proxy", async () => {
    validateAutoRouterConfig.mockResolvedValue({ valid: false, error: "Classifier model is unavailable" });

    const params = {
      accessToken: "token",
      config,
      modelData,
      values: { auto_router_name: "router" },
    };
    const result = await updateCapabilityRouter(params);

    expect(result).toEqual({ kind: "error", message: "Classifier model is unavailable" });
    expect(modelPatchUpdateCall).not.toHaveBeenCalled();
  });
});
