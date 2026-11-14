import { describe, expect, it } from "vitest";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  type DailyActivityAggregatedResponse,
} from "@/components/UsagePage/dailyActivityApi";
import { customerTeamSpend } from "./customerTeamSpend";

describe("customerTeamSpend", () => {
  it("sums spend, requests and tokens across days while keeping unassigned spend", () => {
    const response: DailyActivityAggregatedResponse = {
      metadata: EMPTY_DAILY_ACTIVITY_METADATA,
      results: ["2026-01-01", "2026-01-02"].map((date) => ({
        date,
        metrics: {},
        breakdown: {
          teams: {
            "team-a": {
              metrics: { spend: 2, api_requests: 3, total_tokens: 40 },
              metadata: { team_alias: "Team A" },
            },
            Unassigned: { metrics: { spend: 1, api_requests: 1, total_tokens: 10 }, metadata: {} },
          },
        },
      })),
    };
    expect(customerTeamSpend(response)).toEqual([
      { teamId: "team-a", teamAlias: "Team A", spend: 4, requests: 6, tokens: 80 },
      { teamId: "Unassigned", teamAlias: "Unassigned", spend: 2, requests: 2, tokens: 20 },
    ]);
  });

  it("returns no rows for empty or older responses without a team breakdown", () => {
    expect(customerTeamSpend({ metadata: EMPTY_DAILY_ACTIVITY_METADATA, results: [] })).toEqual([]);
    expect(
      customerTeamSpend({
        metadata: EMPTY_DAILY_ACTIVITY_METADATA,
        results: [{ date: "2026-01-01", metrics: {}, breakdown: {} }],
      }),
    ).toEqual([]);
  });
});
