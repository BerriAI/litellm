import { describe, expect, it } from "vitest";
import { routingSources, routingTiers, trafficShare, type RoutingUsage } from "./routingUsage";

const rows: RoutingUsage[] = [
  { model: "fast", router_name: "router-a", router_type: "complexity", tier: "SIMPLE", requests: 2, spend: 0.4 },
  { model: "strong", router_name: "router-a", router_type: "complexity", tier: "SIMPLE", requests: 1, spend: 1.2 },
  { model: "fast", router_name: "router-a", router_type: "complexity", tier: null, requests: 1, spend: 0.2 },
  { model: "fast", router_name: "router-b", router_type: "quality", tier: "2", requests: 1, spend: 0.2 },
  { model: "fast", router_name: null, router_type: null, tier: null, requests: 6, spend: 1.2 },
];

describe("routing usage", () => {
  it("combines a router's tiers and separates direct traffic and other routers for one model", () => {
    const sources = routingSources(rows.filter((row) => row.model === "fast"));
    expect(sources).toEqual([
      { name: null, requests: 6, spend: 1.2 },
      { name: "router-a", requests: 3, spend: expect.closeTo(0.6) },
      { name: "router-b", requests: 1, spend: 0.2 },
    ]);
    expect(trafficShare(4, 10)).toBe("40%");
  });

  it("keeps both models within a tier and includes fallback calls without inventing their tier", () => {
    expect(routingTiers(rows.filter((row) => row.router_name === "router-a"))).toEqual([
      {
        tier: "SIMPLE",
        requests: 3,
        spend: expect.closeTo(1.6),
        models: [
          { model: "fast", requests: 2, spend: 0.4 },
          { model: "strong", requests: 1, spend: 1.2 },
        ],
      },
      { tier: null, requests: 1, spend: 0.2, models: [{ model: "fast", requests: 1, spend: 0.2 }] },
    ]);
  });

  it("handles zero-cost traffic and an empty range", () => {
    expect(trafficShare(0, 0)).toBe("0%");
    expect(routingTiers([])).toEqual([]);
    expect(routingSources([{ ...rows[0], spend: 0 }])).toEqual([
      { name: null, requests: 0, spend: 0 },
      { name: "router-a", requests: 2, spend: 0 },
    ]);
  });
});
