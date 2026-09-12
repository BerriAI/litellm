import {
  buildAutoRouterCompressionParams,
  DEFAULT_AUTO_ROUTER_COMPRESSION,
  hydrateAutoRouterCompression,
  NO_COMPRESSION,
} from "./buildAutoRouterCompression";

describe("buildAutoRouterCompressionParams", () => {
  it("omits both keys when routing was never configured", () => {
    expect(buildAutoRouterCompressionParams(DEFAULT_AUTO_ROUTER_COMPRESSION)).toEqual({});
  });

  it("mirrors routing onto model when same-as-routing is chosen", () => {
    const params = buildAutoRouterCompressionParams({
      routing: "headroom-a",
      sameAsRouting: true,
      model: undefined,
    });
    expect(params).toEqual({
      auto_router_routing_compression: "headroom-a",
      auto_router_model_compression: "headroom-a",
    });
  });

  it("uses the explicit model choice when different is chosen", () => {
    const params = buildAutoRouterCompressionParams({
      routing: "headroom-a",
      sameAsRouting: false,
      model: "headroom-b",
    });
    expect(params).toEqual({
      auto_router_routing_compression: "headroom-a",
      auto_router_model_compression: "headroom-b",
    });
  });

  it("defaults the model side to none when different is chosen but nothing is picked", () => {
    const params = buildAutoRouterCompressionParams({
      routing: "headroom-a",
      sameAsRouting: false,
      model: undefined,
    });
    expect(params).toEqual({
      auto_router_routing_compression: "headroom-a",
      auto_router_model_compression: NO_COMPRESSION,
    });
  });

  it("sends the none sentinel when routing itself is explicitly turned off", () => {
    const params = buildAutoRouterCompressionParams({
      routing: NO_COMPRESSION,
      sameAsRouting: true,
      model: undefined,
    });
    expect(params).toEqual({
      auto_router_routing_compression: NO_COMPRESSION,
      auto_router_model_compression: NO_COMPRESSION,
    });
  });
});

describe("hydrateAutoRouterCompression", () => {
  it("returns the default state when neither key is set", () => {
    expect(hydrateAutoRouterCompression({})).toEqual(DEFAULT_AUTO_ROUTER_COMPRESSION);
  });

  it("is same-as-routing when the model value matches routing", () => {
    const state = hydrateAutoRouterCompression({
      auto_router_routing_compression: "headroom-a",
      auto_router_model_compression: "headroom-a",
    });
    expect(state).toEqual({ routing: "headroom-a", sameAsRouting: true, model: undefined });
  });

  it("is different when the model value diverges from routing", () => {
    const state = hydrateAutoRouterCompression({
      auto_router_routing_compression: "headroom-a",
      auto_router_model_compression: "headroom-b",
    });
    expect(state).toEqual({ routing: "headroom-a", sameAsRouting: false, model: "headroom-b" });
  });

  it("treats a missing model key as no model-hop compression, not same-as-routing", () => {
    const state = hydrateAutoRouterCompression({ auto_router_routing_compression: "headroom-a" });
    expect(state).toEqual({ routing: "headroom-a", sameAsRouting: false, model: "none" });
  });

  it("re-saving a routing-only config leaves the model hop uncompressed", () => {
    // Regression: the backend reads an absent model key as no model-hop compression.
    // Hydrating it as same-as-routing made opening the router and saving any unrelated
    // edit write the routing guardrail onto the model hop, so the model call silently
    // started receiving compressed messages.
    const stored = { auto_router_routing_compression: "headroom-a" };
    const rebuilt = buildAutoRouterCompressionParams(hydrateAutoRouterCompression(stored));
    expect(rebuilt.auto_router_model_compression).toBe("none");
    expect(rebuilt.auto_router_model_compression).not.toBe("headroom-a");
  });

  it("surfaces a stored model-only policy instead of reading as untouched", () => {
    // Regression: the backend treats either key alone as an authoritative policy, so a
    // model-only config that hydrated to the inherit state was invisible in the form,
    // and the next save overwrote the stored model hop with the routing value.
    const state = hydrateAutoRouterCompression({ auto_router_model_compression: "headroom-b" });
    expect(state).toEqual({ routing: "none", sameAsRouting: false, model: "headroom-b" });
  });

  it("round-trips a model-only policy without changing either hop", () => {
    const stored = { auto_router_model_compression: "headroom-b" };
    const rebuilt = buildAutoRouterCompressionParams(hydrateAutoRouterCompression(stored));
    expect(rebuilt.auto_router_model_compression).toBe("headroom-b");
    expect(rebuilt.auto_router_routing_compression).toBe("none");
  });

  it("round-trips through buildAutoRouterCompressionParams", () => {
    const original = { auto_router_routing_compression: "headroom-a", auto_router_model_compression: "none" };
    const rebuilt = buildAutoRouterCompressionParams(hydrateAutoRouterCompression(original));
    expect(rebuilt).toEqual(original);
  });
});
