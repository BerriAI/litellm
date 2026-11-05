import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import ModelUsageChart from "./ModelUsageChart";

vi.mock("@/components/shared/charts", () => ({
  StackedUsageChart: ({ data, labels }: { data: unknown; labels: string[] }) => (
    <div role="img" aria-label="Daily key usage chart">
      {JSON.stringify({ data, labels })}
    </div>
  ),
}));
const range = { start_date: "2026-10-07", end_date: "2026-11-05" };
function mockActivity() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(
    async () =>
      new Response(
        JSON.stringify({
          results: [
            {
              date: "2026-11-03",
              metrics: { api_requests: 100, total_tokens: 1000, spend: 10 },
              breakdown: {
                api_keys: {
                  "key-a": {
                    metrics: { api_requests: 30, spend: 3, total_tokens: 300 },
                    metadata: { key_alias: "Support key" },
                  },
                  "key-b": {
                    metrics: { api_requests: 20, spend: 2, total_tokens: 200 },
                    metadata: { key_alias: "Research key" },
                  },
                },
              },
            },
          ],
        }),
        { headers: { "Content-Type": "application/json" } },
      ),
  );
}

describe("model daily usage", () => {
  afterEach(() => vi.restoreAllMocks());
  it("should render the model's daily requests stacked by key and preserve remaining usage", async () => {
    const fetch = mockActivity();
    renderWithProviders(
      <ModelUsageChart model="anthropic/retiring-model" team={null} scope="test-global" range={range} />,
    );
    const chart = await screen.findByRole("img", { name: "Daily key usage chart" });
    expect(chart).toHaveTextContent('"labels":["Support key","Research key","Other"]');
    expect(chart).toHaveTextContent('"s0":30,"s1":20,"s2":50');
    const request = fetch.mock.calls[0][0] as Request;
    expect(new URL(request.url).searchParams.get("model")).toBe("anthropic/retiring-model");
  });
  it("should apply the selected team and let users switch to spend", async () => {
    const fetch = mockActivity();
    const user = userEvent.setup();
    renderWithProviders(
      <ModelUsageChart model="anthropic/retiring-model" team="support" scope="test-team" range={range} />,
    );
    await screen.findByRole("img", { name: "Daily key usage chart" });
    const request = fetch.mock.calls[0][0] as Request;
    expect(new URL(request.url).pathname).toBe("/team/daily/activity/aggregated");
    expect(new URL(request.url).searchParams.get("team_ids")).toBe("support");
    await user.click(screen.getByRole("radio", { name: "Spend", exact: true }));
    await waitFor(() =>
      expect(screen.getByRole("img", { name: "Daily key usage chart" })).toHaveTextContent('"s0":3,"s1":2,"s2":5'),
    );
  });
});
