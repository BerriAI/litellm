import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ModelRoutingShare from "./ModelRoutingShare";

const scope = { start_date: "2026-01-01", end_date: "2026-01-02", user_id: "owner" };
const rows = [
  { model: "fast", router_name: null, router_type: null, tier: null, requests: 6, spend: 1.2 },
  { model: "fast", router_name: "router-a", router_type: "complexity", tier: "SIMPLE", requests: 4, spend: 0.8 },
];

afterEach(() => vi.restoreAllMocks());

describe("model traffic sources", () => {
  it("loads the selected model and user, then opens direct and router request/spend figures", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(rows), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = new QueryClient();
    render(
      <QueryClientProvider client={client}>
        <ModelRoutingShare model="fast" scope={scope} />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "40% via auto-router" }));
    expect(await screen.findByText("Direct")).toBeVisible();
    expect(screen.getByText("router-a")).toBeVisible();
    expect(screen.getByText("$1.20")).toBeVisible();
    expect(screen.getByText("$0.80")).toBeVisible();
    expect(screen.getByText("60%")).toBeVisible();
    expect(screen.getByText(/Based on 10 retained requests/)).toBeVisible();
    const request = fetch.mock.calls[0][0] as Request;
    const query = new URL(request.url).searchParams;
    expect(Object.fromEntries(query)).toEqual({ ...scope, destination_model: "fast" });
  });

  it("does not present unavailable or missing logs as zero-percent routed traffic", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("[]", { headers: { "Content-Type": "application/json" } }),
    );
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ModelRoutingShare model="fast" scope={scope} />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("No retained requests for traffic sources")).toBeVisible();
    expect(screen.queryByRole("button", { name: /via auto-router/ })).not.toBeInTheDocument();
  });
});
