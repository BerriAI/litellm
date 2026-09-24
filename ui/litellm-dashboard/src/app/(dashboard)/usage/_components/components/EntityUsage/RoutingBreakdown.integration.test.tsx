import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@/lib/http/schema";
import RoutingBreakdown from "./RoutingBreakdown";

type Row = components["schemas"]["RoutingUsageRow"];
const rows: Row[] = [
  {
    attribution: "router",
    router_name: "retired-router",
    model: "actual-model",
    provider: "provider",
    requests: 2,
    failed_attempts: 1,
    cache_hits: 1,
    inference_spend: 0.002,
    classifier_spend: 0.00025,
    classifier_cost_known_requests: 1,
  },
  {
    attribution: "router",
    router_name: "other-router",
    model: "actual-model",
    provider: "provider",
    requests: 1,
    failed_attempts: 0,
    cache_hits: 0,
    inference_spend: 0.003,
    classifier_spend: 0,
    classifier_cost_known_requests: 0,
  },
  {
    attribution: "direct",
    router_name: null,
    model: "actual-model",
    provider: "provider",
    requests: 1,
    failed_attempts: 0,
    cache_hits: 0,
    inference_spend: 0.001,
    classifier_spend: 0,
    classifier_cost_known_requests: 0,
  },
  {
    attribution: "unattributed",
    router_name: null,
    model: "actual-model",
    provider: "provider",
    requests: 1,
    failed_attempts: 0,
    cache_hits: 0,
    inference_spend: 0.004,
    classifier_spend: 0,
    classifier_cost_known_requests: 0,
  },
];

afterEach(() => vi.unstubAllGlobals());

it("shows historical destinations and all model origins without mixing classifier spend", async () => {
  const requests: URL[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      requests.push(new URL(request.url));
      const response = {
        results: rows,
        spend_logs_disabled: false,
        configured_retention: "30d",
        coverage: "retained_spend_logs",
        start_date: "2025-01-01",
        end_date: "2025-01-03",
      };
      return new Response(JSON.stringify(response), { headers: { "Content-Type": "application/json" } });
    }),
  );
  const user = userEvent.setup();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
      <RoutingBreakdown
        accessToken="test-token"
        startTime={new Date(2025, 0, 1)}
        endTime={new Date(2025, 0, 2)}
        teamIds={["team-a"]}
        enabled
      />
    </QueryClientProvider>,
  );
  expect(await screen.findByRole("cell", { name: "retired-router" })).toBeInTheDocument();
  expect(screen.getByText("$0.005")).toBeInTheDocument();
  expect(screen.getByText("$0.00025")).toBeInTheDocument();
  expect(screen.getByText(/Classifier cost is known for 1 of 3/)).toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Router" }), { target: { value: "retired-router" } });
  expect(screen.queryByRole("cell", { name: "other-router" })).not.toBeInTheDocument();
  expect(screen.getByRole("cell", { name: "100.0%" })).toBeInTheDocument();
  await user.click(screen.getByRole("tab", { name: "Model origins" }));
  fireEvent.change(screen.getByRole("combobox", { name: "Model" }), { target: { value: "actual-model" } });
  const table = screen.getByRole("table");
  expect(within(table).getByRole("cell", { name: "Direct" })).toBeInTheDocument();
  expect(within(table).getByRole("cell", { name: "Unattributed" })).toBeInTheDocument();
  expect(within(table).getByRole("cell", { name: "other-router" })).toBeInTheDocument();
  expect(screen.getByText("$0.01")).toBeInTheDocument();
  expect(requests[0].searchParams.get("start_date")).toBe("2025-01-01");
  expect(requests[0].searchParams.get("end_date")).toBe("2025-01-03");
  expect(requests[0].searchParams.getAll("team_ids")).toEqual(["team-a"]);
});
