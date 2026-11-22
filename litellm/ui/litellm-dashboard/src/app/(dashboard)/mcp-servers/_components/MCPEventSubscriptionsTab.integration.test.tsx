import React from "react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { MCPEventSubscription } from "@/components/mcp_tools/types";

const networking = vi.hoisted(() => ({
  fetchMCPEventSubscriptions: vi.fn(),
  terminateMCPEventSubscription: vi.fn(),
  sendMCPEventTestDelivery: vi.fn(),
}));
vi.mock("@/components/networking", () => networking);

import { MCPEventSubscriptionsTab } from "./MCPEventSubscriptionsTab";

const SUBSCRIPTION: MCPEventSubscription = {
  id: "sub_1",
  server_id: "srv-ci",
  event_name: "build.failed",
  arguments: { repo: "acme/web" },
  callback_host: "hooks.example.com",
  status: "active",
  user_id: "user-1",
  team_id: null,
  refresh_before: "2026-11-21T00:00:00Z",
  last_delivery_at: null,
  last_error: "timeout",
  failed_since: null,
  created_at: null,
};

function renderTab(searchParams = "") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <NuqsTestingAdapter searchParams={searchParams} hasMemory>
      <QueryClientProvider client={client}>
        <MCPEventSubscriptionsTab accessToken="sk-admin" />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );
}

describe("MCPEventSubscriptionsTab", () => {
  beforeEach(() => {
    networking.fetchMCPEventSubscriptions.mockResolvedValue({ subscriptions: [SUBSCRIPTION] });
    networking.terminateMCPEventSubscription.mockResolvedValue({ ...SUBSCRIPTION, status: "terminated" });
    networking.sendMCPEventTestDelivery.mockResolvedValue({ delivered: true, status: 200 });
  });

  it("reads a bookmarked subscription page and requests the previous page", async () => {
    renderTab("?subscriptionOffset=500");
    await userEvent.click(await screen.findByRole("button", { name: "Previous page" }));
    await screen.findByText("build.failed");
    expect(networking.fetchMCPEventSubscriptions).toHaveBeenCalledWith("sk-admin", 500, 1);
    expect(networking.fetchMCPEventSubscriptions).toHaveBeenCalledWith("sk-admin", 0, 1);
  });

  it("lists webhook subscriptions with their arguments, callback host and last error", async () => {
    renderTab();
    const row = await screen.findByRole("row", { name: /build.failed/ });
    expect(within(row).getByText("repo=acme/web")).toBeInTheDocument();
    expect(within(row).getByText("hooks.example.com")).toBeInTheDocument();
    expect(within(row).getByText("Timed out")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Terminate subscription/ })).not.toBeInTheDocument();
  });

  it("shows an empty state when nothing is subscribed", async () => {
    networking.fetchMCPEventSubscriptions.mockResolvedValue({ subscriptions: [] });
    renderTab();
    expect(await screen.findByText("No webhook event subscriptions yet.")).toBeInTheDocument();
  });
});
