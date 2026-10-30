import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import CategoryROI from "./CategoryROI";
import type { Report } from "./data";

const get = vi.hoisted(() => vi.fn());
vi.mock("@/components/networking", () => ({ apiClient: { get } }));
const report: Report = {
  version: "v1",
  categories: [
    { id: "api", name: "API development", description: "Build APIs" },
    { id: "build", name: "CI/CD", description: "Run builds" },
  ],
  buckets: [
    { day: "2026-10-29", category_id: "api", model: "model-a", spend: 6, requests: 3, unknown_cost_requests: 0 },
    { day: "2026-10-29", category_id: "build", model: "model-b", spend: 2, requests: 1, unknown_cost_requests: 0 },
  ],
  sessions: [
    {
      session_key: "scoped-session",
      session_id: "session",
      user: "alice",
      started_at: "2026-10-29T12:00:00Z",
      excerpt: "Implement pagination",
      category_ids: ["api"],
      model_spend: { "model-a": 6 },
      spend: 6,
      requests: 3,
    },
    {
      session_key: "standalone",
      session_id: "",
      user: "bob",
      started_at: "2026-10-29T13:00:00Z",
      excerpt: "Repair build",
      category_ids: ["build"],
      model_spend: { "model-b": 2 },
      spend: 2,
      requests: 1,
    },
  ],
};

it("filters sessions by area and scopes drawer examples to the chosen session", async () => {
  get.mockImplementation((url: string) => {
    if (url.endsWith("/status"))
      return Promise.resolve({ configured: true, settings: { phase: "active", version: "v1" } });
    if (url.includes("/report?")) return Promise.resolve(report);
    return Promise.resolve([
      { request_id: "request-one", model: "model-a", spend: 2, excerpt: "Implement safe pagination" },
    ]);
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <CategoryROI token="fixture" start="2026-10-29T00:00:00Z" end="2026-10-30T00:00:00Z" />
    </QueryClientProvider>,
  );
  expect(await screen.findByText("Implement pagination")).toBeVisible();
  const summary = screen.getByRole("region", { name: "Spend summary" });
  expect(summary).toHaveTextContent("$8.00");
  expect(summary).toHaveTextContent("Sessions1");
  expect(summary).toHaveTextContent("Avg per session$6.00");
  expect(summary).toHaveTextContent("1 standalone requests");
  fireEvent.click(
    within(screen.getByLabelText("Filter sessions by work area")).getByRole("button", { name: "API development" }),
  );
  expect(screen.queryByText("Repair build")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Implement paginationAPI development/ }));
  expect(await screen.findByText("Implement safe pagination")).toBeVisible();
  expect(get).toHaveBeenLastCalledWith(expect.stringContaining("session_key=scoped-session"), {
    accessToken: "fixture",
  });
  expect(screen.getByRole("link", { name: "View source request" })).toHaveAttribute(
    "href",
    expect.stringContaining("log_id=request-one"),
  );
  client.clear();
});
