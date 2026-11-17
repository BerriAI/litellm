import { describe, expect, it } from "vitest";
import type { components } from "@/lib/http/schema";
import { dailyModelRequests } from "./migrationData";

type Day = components["schemas"]["DailySpendData"];
function day(date: string, api_requests: number): Day {
  return {
    date,
    metrics: {
      api_requests,
      spend: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      successful_requests: api_requests,
      failed_requests: 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  };
}

describe("model request trend", () => {
  it("should include zero-usage days after the model stops receiving traffic", () => {
    expect(dailyModelRequests([day("2026-11-03", 120)], { start_date: "2026-11-03", end_date: "2026-11-05" })).toEqual([
      { date: "2026-11-03", requests: 120 },
      { date: "2026-11-04", requests: 0 },
      { date: "2026-11-05", requests: 0 },
    ]);
  });
  it("should keep dates chronological across month boundaries and preserve daily totals", () => {
    expect(
      dailyModelRequests([day("2026-11-01", 5), day("2026-10-31", 50)], {
        start_date: "2026-10-31",
        end_date: "2026-11-02",
      }),
    ).toEqual([
      { date: "2026-10-31", requests: 50 },
      { date: "2026-11-01", requests: 5 },
      { date: "2026-11-02", requests: 0 },
    ]);
  });
});
