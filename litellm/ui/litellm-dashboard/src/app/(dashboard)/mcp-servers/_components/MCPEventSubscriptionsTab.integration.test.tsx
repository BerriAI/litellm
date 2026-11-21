import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

function renderTab(canManage: boolean) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MCPEventSubscriptionsTab accessToken="sk-admin" canManage={canManage} />
    </QueryClientProvider>,
  );
}

describe("MCPEventSubscriptionsTab", () => {
  beforeEach(() => {
    networking.fetchMCPEventSubscriptions.mockResolvedValue({ subscriptions: [SUBSCRIPTION] });
    networking.terminateMCPEventSubscription.mockResolvedValue({ ...SUBSCRIPTION, status: "terminated" });
    networking.sendMCPEventTestDelivery.mockResolvedValue({ delivered: true, status: 200 });
  });

  it("lists webhook subscriptions with their arguments, callback host and last error", async () => {
    renderTab(false);
    const row = (await screen.findByText("build.failed")).closest("tr");
    if (!row) throw new Error("subscription row missing");
    expect(within(row).getByText("repo=acme/web")).toBeInTheDocument();
    expect(within(row).getByText("hooks.example.com")).toBeInTheDocument();
    expect(within(row).getByText("Timed out")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Terminate subscription/ })).not.toBeInTheDocument();
  });

  it("terminates a subscription after confirmation", async () => {
    const user = userEvent.setup();
    renderTab(true);
    await user.click(await screen.findByRole("button", { name: "Terminate subscription sub_1" }));
    await user.click(screen.getByRole("button", { name: "Terminate" }));
    expect(networking.terminateMCPEventSubscription).toHaveBeenCalledWith("sk-admin", "sub_1");
  });

  it("sends a signed test event and reports the callback status", async () => {
    const user = userEvent.setup();
    renderTab(true);
    await user.click(await screen.findByRole("button", { name: "Send a signed test event to subscription sub_1" }));
    expect(await screen.findByText("Test event delivered (HTTP 200)")).toBeInTheDocument();
    expect(networking.sendMCPEventTestDelivery).toHaveBeenCalledWith("sk-admin", "sub_1");
  });

  it("shows an empty state when nothing is subscribed", async () => {
    networking.fetchMCPEventSubscriptions.mockResolvedValue({ subscriptions: [] });
    renderTab(true);
    expect(await screen.findByText("No webhook event subscriptions yet.")).toBeInTheDocument();
  });
});
