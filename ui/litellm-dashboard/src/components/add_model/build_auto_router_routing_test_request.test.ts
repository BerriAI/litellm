import { buildAutoRouterRoutingTestRequest } from "./build_auto_router_routing_test_request";
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
