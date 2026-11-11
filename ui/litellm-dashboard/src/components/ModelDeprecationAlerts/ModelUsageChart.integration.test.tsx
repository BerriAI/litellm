import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import ModelUsageChart from "./ModelUsageChart";

vi.mock("@/components/shared/charts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/shared/charts")>()),
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
  it("should apply the selected team and show requests without metric controls", async () => {
    const fetch = mockActivity();
    renderWithProviders(
      <ModelUsageChart model="anthropic/retiring-model" team="support" scope="test-team" range={range} />,
    );
    await screen.findByRole("img", { name: "Daily key usage chart" });
    const request = fetch.mock.calls[0][0] as Request;
    expect(new URL(request.url).pathname).toBe("/team/daily/activity/aggregated");
    expect(new URL(request.url).searchParams.get("team_ids")).toBe("support");
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Daily key usage chart" })).toHaveTextContent('"s0":30,"s1":20,"s2":50');
  });
  it("should keep all keys in the stacks while allowing the key labels to expand", async () => {
    const fetch = mockActivity();
    const keys = Object.fromEntries(
      Array.from({ length: 12 }, (_, index) => [
        `key-${index}`,
        {
          metrics: { api_requests: 10, spend: 1, total_tokens: 100 },
          metadata: { key_alias: `Key ${index + 1}` },
        },
      ]),
    );
    const response = {
      results: [
        {
          date: "2026-11-03",
          metrics: { api_requests: 120, spend: 12, total_tokens: 1200 },
          breakdown: { api_keys: keys },
        },
      ],
    };
    fetch.mockResolvedValueOnce(
      new Response(JSON.stringify(response), { headers: { "Content-Type": "application/json" } }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <ModelUsageChart model="anthropic/retiring-model" team={null} scope="test-legend" range={range} />,
    );
    expect(await screen.findByRole("img", { name: "Daily key usage chart" })).toHaveTextContent('"s11":10');
    expect(screen.queryByText("Key 12", { exact: true })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show all key labels" }));
    expect(screen.getByText("Key 12", { exact: true })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show fewer key labels" }));
    expect(screen.queryByText("Key 12", { exact: true })).not.toBeInTheDocument();
  });
});
