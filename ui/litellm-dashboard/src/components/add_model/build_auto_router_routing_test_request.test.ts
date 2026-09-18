import { describe, expect, it } from "vitest";
import {
  buildAutoRouterRoutingTestRequest,
  buildSavedJevConnectionTestRequest,
  JEV_CONNECTION_TEST_PROMPT,
} from "./build_auto_router_routing_test_request";
import { ComplexityRouterConfigPayload } from "./build_complexity_router_config";

const CONFIG = {
  tiers: { SIMPLE: ["cheap"], MEDIUM: ["mid"], COMPLEX: ["strong"], REASONING: ["o3"] },
  classifier_type: "heuristic",
} as unknown as ComplexityRouterConfigPayload;

const params = {
  prompt: "what is 2+2",
  config: CONFIG,
  defaultModel: "mid",
  routerName: "my-router",
  teamId: "team-1",
};

describe("buildAutoRouterRoutingTestRequest", () => {
  it.each(["object", "json"])("probes saved JEV %s configuration with custom tiers and team context", (format) => {
    const config = {
      classifier_type: "jev",
      jev_classifier_config: { model: "jev-test", timeout_ms: 900 },
      tiers: { QUICK: ["fast"], DEEP: ["strong"] },
      tier_definitions: { QUICK: "Simple questions", DEEP: "Complex questions" },
      fallback_tier: "DEEP",
      classifier_context_window_size: 4,
    };
    expect(
      buildSavedJevConnectionTestRequest(
        format === "json" ? JSON.stringify(config) : config,
        "strong",
        "saved-router",
        "team-1",
      ),
    ).toEqual({
      prompt: JEV_CONNECTION_TEST_PROMPT,
      complexity_router_config: config,
      default_model: "strong",
      router_name: "saved-router",
      team_id: "team-1",
    });
  });
  it.each([undefined, null, "not json", "[]", {}, { classifier_type: "llm", tiers: {} }, { classifier_type: "jev" }])(
    "does not build a JEV probe for invalid or other classifier configurations: %j",
    (config) => {
      expect(buildSavedJevConnectionTestRequest(config)).toBeUndefined();
    },
  );
  it("sends the prompt with the config being edited", () => {
    const request = buildAutoRouterRoutingTestRequest(params);

    expect(request.prompt).toBe("what is 2+2");
    expect(request.complexity_router_config).toBe(CONFIG);
    expect(request.default_model).toBe("mid");
    expect(request.router_name).toBe("my-router");
    expect(request.team_id).toBe("team-1");
  });

  it("trims the router name a caller is midway through typing", () => {
    expect(buildAutoRouterRoutingTestRequest({ ...params, routerName: "  my-router  " }).router_name).toBe("my-router");
  });

  it("omits optional fields a caller has not filled in", () => {
    const blankParams = { ...params, defaultModel: undefined, routerName: "   ", teamId: undefined };

    const request = buildAutoRouterRoutingTestRequest(blankParams);

    expect(request).not.toHaveProperty("default_model");
    expect(request).not.toHaveProperty("router_name");
    expect(request).not.toHaveProperty("team_id");
  });
});
