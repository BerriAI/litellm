import { buildAutoRouterShuntParams, DEFAULT_AUTO_ROUTER_SHUNT, hydrateAutoRouterShunt } from "./buildAutoRouterShunt";

describe("buildAutoRouterShuntParams", () => {
  it("omits all three keys when the threshold was never configured", () => {
    expect(buildAutoRouterShuntParams(DEFAULT_AUTO_ROUTER_SHUNT)).toEqual({});
  });

  it("sends only the threshold when neither worker model is chosen", () => {
    const params = buildAutoRouterShuntParams({ minLines: 350, bulkReadModel: undefined, codeWriteModel: undefined });
    expect(params).toEqual({ auto_router_shunt_min_lines: 350 });
  });

  it("sends both worker models when both are chosen", () => {
    const params = buildAutoRouterShuntParams({
      minLines: 200,
      bulkReadModel: "claude-haiku-4-5",
      codeWriteModel: "gpt-5.6-luna",
    });
    expect(params).toEqual({
      auto_router_shunt_min_lines: 200,
      auto_router_shunt_bulk_read_model: "claude-haiku-4-5",
      auto_router_shunt_code_write_model: "gpt-5.6-luna",
    });
  });

  it("omits a worker model key when only the other is chosen", () => {
    const params = buildAutoRouterShuntParams({
      minLines: 200,
      bulkReadModel: "claude-haiku-4-5",
      codeWriteModel: undefined,
    });
    expect(params).toEqual({ auto_router_shunt_min_lines: 200, auto_router_shunt_bulk_read_model: "claude-haiku-4-5" });
  });

  it("arms at a threshold of 0, since only undefined means untouched", () => {
    const params = buildAutoRouterShuntParams({ minLines: 0, bulkReadModel: undefined, codeWriteModel: undefined });
    expect(params).toEqual({ auto_router_shunt_min_lines: 0 });
  });
});

describe("hydrateAutoRouterShunt", () => {
  it("returns the default (untouched) state when the threshold key is absent", () => {
    expect(hydrateAutoRouterShunt({})).toEqual(DEFAULT_AUTO_ROUTER_SHUNT);
  });

  it("hydrates a stored threshold with no worker models set", () => {
    const state = hydrateAutoRouterShunt({ auto_router_shunt_min_lines: 350 });
    expect(state).toEqual({ minLines: 350, bulkReadModel: undefined, codeWriteModel: undefined });
  });

  it("hydrates a stored threshold with both worker models set", () => {
    const state = hydrateAutoRouterShunt({
      auto_router_shunt_min_lines: 200,
      auto_router_shunt_bulk_read_model: "claude-haiku-4-5",
      auto_router_shunt_code_write_model: "gpt-5.6-luna",
    });
    expect(state).toEqual({ minLines: 200, bulkReadModel: "claude-haiku-4-5", codeWriteModel: "gpt-5.6-luna" });
  });

  it("round-trips through buildAutoRouterShuntParams", () => {
    const original = {
      auto_router_shunt_min_lines: 350,
      auto_router_shunt_bulk_read_model: "claude-haiku-4-5",
      auto_router_shunt_code_write_model: "gpt-5.6-luna",
    };
    const rebuilt = buildAutoRouterShuntParams(hydrateAutoRouterShunt(original));
    expect(rebuilt).toEqual(original);
  });
});
