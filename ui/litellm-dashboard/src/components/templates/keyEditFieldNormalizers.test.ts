import { describe, expect, it } from "vitest";
import { keyTypeFromRoutes, normalizeKeyEditRoutePayload } from "./keyEditFieldNormalizers";

describe("key edit route payload normalization", () => {
  it("parses comma-separated custom routes", () => {
    expect(normalizeKeyEditRoutePayload({ allowed_routes: "route1, route2, route3" }, [])).toEqual({
      allowed_routes: ["route1", "route2", "route3"],
    });
  });

  it("submits the default preset without derived routes", () => {
    expect(normalizeKeyEditRoutePayload({ allowed_routes: "" }, ["llm_api_routes"])).toEqual({
      key_type: "default",
    });
  });

  it("submits the AI APIs preset without derived routes", () => {
    expect(normalizeKeyEditRoutePayload({ allowed_routes: "llm_api_routes" }, [])).toEqual({
      key_type: "llm_api",
    });
  });

  it("preserves mixed allowlists without applying a preset", () => {
    expect(normalizeKeyEditRoutePayload({ allowed_routes: "llm_api_routes, /custom" }, [])).toEqual({
      allowed_routes: ["llm_api_routes", "/custom"],
    });
  });

  it.each([
    { original: ["llm_api_routes"], submitted: "llm_api_routes" },
    { original: null, submitted: "" },
    { original: ["beta_routes", "alpha_routes"], submitted: "alpha_routes, beta_routes" },
  ])("omits unchanged routes: $original", ({ original, submitted }) => {
    expect(normalizeKeyEditRoutePayload({ allowed_routes: submitted }, original)).toEqual({});
  });

  it("does not label a mixed allowlist as Full Access", () => {
    expect(keyTypeFromRoutes(["llm_api_routes", "/custom"])).toBeUndefined();
  });
});
