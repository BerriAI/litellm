import { renderWithProviders as render } from "../../../../../tests/test-utils";
import React from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

vi.mock("@/components/networking", () => ({
  organizationListCall: vi.fn().mockResolvedValue([]),
  dailyActivityAggregatedCall: vi.fn().mockResolvedValue({ results: [], metadata: {} }),
}));

vi.mock("./UsageTab", () => ({ __esModule: true, default: () => <div data-testid="usage-tab" /> }));
vi.mock("./PromptCompressionTab", () => ({ __esModule: true, default: () => <div data-testid="compression-tab" /> }));
vi.mock("./PromptCachingTab", () => ({ __esModule: true, default: () => <div data-testid="caching-tab" /> }));
vi.mock("./AutoRouterBenchmarksTab", () => ({
  __esModule: true,
  default: () => <div data-testid="autorouter-benchmarks-tab" />,
}));

import CostOptimizationView from "./CostOptimizationView";

interface UrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
  resetUrlUpdateQueueOnMount?: boolean;
}

const renderView = (userRole = "Admin", url: UrlOptions = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const viewAs = (role: string) => {
    useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole: role });
    return (
      <QueryClientProvider client={queryClient}>
        <CostOptimizationView accessToken="test-token" userId="u1" userRole={role} />
      </QueryClientProvider>
    );
  };
  const view = render(viewAs(userRole), url);
  return { ...view, changeRole: (role: string) => view.rerender(viewAs(role)) };
};

const lastUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

describe("CostOptimizationView", () => {
  beforeEach(() => {
    useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole: "Admin" });
  });

  it("renders the standard page header with the sidebar's Cost Optimization icon", () => {
    const { container } = renderView();

    expect(screen.getByRole("heading", { level: 1, name: "Cost Optimization" })).toBeInTheDocument();
    expect(screen.getByText(/Track and configure the mechanisms that save you money/)).toBeInTheDocument();
    expect(container.querySelector(".lucide-piggy-bank")).not.toBeNull();
  });

  it("renders the four cost-optimization tabs", () => {
    renderView();

    expect(screen.getByText("Overall")).toBeInTheDocument();
    expect(screen.getByText("Prompt Compression")).toBeInTheDocument();
    expect(screen.getByText("Prompt Caching")).toBeInTheDocument();
    expect(screen.getByText("Auto-Router")).toBeInTheDocument();
  });

  it("defaults to the Overall tab and switches the active tab on click", () => {
    renderView();

    expect(screen.getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "false");

    fireEvent.click(screen.getByRole("tab", { name: "Prompt Compression" }));

    expect(screen.getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "true");
  });

  // Unlike the other three pages in this cleanup, Cost Optimization keeps its
  // nav entry for internal users: the Overall tab runs on /user/daily/activity,
  // which every role may call. Only the tabs reading proxy-wide config and
  // telemetry (/config/list, /auto_router/benchmarks, guardrail management)
  // are proxy-admin-only, so those are what disappear.
  describe("proxy-admin-only tabs", () => {
    it.each(["Internal User", "Internal Viewer", "Org Admin"])("shows %s the Overall tab only", (userRole) => {
      renderView(userRole);

      expect(screen.getByRole("tab", { name: "Overall" })).toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Prompt Compression" })).not.toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Prompt Caching" })).not.toBeInTheDocument();
      expect(screen.queryByRole("tab", { name: "Auto-Router" })).not.toBeInTheDocument();
    });

    it("never mounts the panels behind the admin-only endpoints for an internal user", () => {
      renderView("Internal User");

      expect(screen.getByTestId("usage-tab")).toBeInTheDocument();
      expect(screen.queryByTestId("compression-tab")).not.toBeInTheDocument();
      expect(screen.queryByTestId("caching-tab")).not.toBeInTheDocument();
      expect(screen.queryByTestId("autorouter-benchmarks-tab")).not.toBeInTheDocument();
    });
  });

  describe("URL state", () => {
    it.each([
      ["compression", "Prompt Compression", "compression-tab"],
      ["caching", "Prompt Caching", "caching-tab"],
      ["autorouter-usage", "Auto-Router", "autorouter-benchmarks-tab"],
    ])("opens ?tab=%s for a proxy admin", (slug, name, panel) => {
      renderView("Admin", { searchParams: `?tab=${slug}` });

      expect(screen.getByRole("tab", { name, selected: true })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Overall", selected: false })).toBeInTheDocument();
      expect(screen.getByTestId(panel)).toBeInTheDocument();
    });

    it("opens a proxy-admin-only tab for an admin viewer", () => {
      renderView("Admin Viewer", { searchParams: "?tab=caching" });

      expect(screen.getByRole("tab", { name: "Prompt Caching", selected: true })).toBeInTheDocument();
      expect(screen.getByTestId("caching-tab")).toBeInTheDocument();
    });

    it("replaces ?tab= when a tab is chosen and drops it for Overall", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("Admin", { onUrlUpdate });

      fireEvent.click(screen.getByRole("tab", { name: "Auto-Router" }));
      await waitFor(() => expect(lastUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("autorouter-usage"));
      expect(lastUpdate(onUrlUpdate)?.options.history).toBe("replace");

      fireEvent.click(screen.getByRole("tab", { name: "Overall" }));
      await waitFor(() => expect(lastUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
    });

    it("writes nothing to the URL on a bare visit", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("Admin", { onUrlUpdate, resetUrlUpdateQueueOnMount: false });

      fireEvent.click(screen.getByRole("tab", { name: "Prompt Compression" }));
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(onUrlUpdate).toHaveBeenCalledTimes(1);
      expect(lastUpdate(onUrlUpdate)?.queryString).toBe("?tab=compression");
    });

    it("keeps the tab opened from the URL mounted after switching away", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("Admin", { searchParams: "?tab=caching", onUrlUpdate });

      fireEvent.click(screen.getByRole("tab", { name: "Overall" }));
      await waitFor(() => expect(lastUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));

      expect(screen.getByRole("tab", { name: "Overall", selected: true })).toBeInTheDocument();
      expect(screen.getByTestId("caching-tab")).toBeInTheDocument();
      expect(screen.queryByTestId("compression-tab")).not.toBeInTheDocument();
    });

    it.each(["Internal User", "Internal Viewer", "Org Admin"])(
      "shows %s Overall for a proxy-admin-only link without mounting it, and keeps the link for a later admin role",
      async (userRole) => {
        const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
        const { changeRole } = renderView(userRole, {
          searchParams: "?tab=caching",
          onUrlUpdate,
          resetUrlUpdateQueueOnMount: false,
        });

        expect(screen.getByRole("tab", { name: "Overall", selected: true })).toBeInTheDocument();
        expect(screen.getByTestId("usage-tab")).toBeInTheDocument();
        expect(screen.queryByTestId("caching-tab")).not.toBeInTheDocument();

        changeRole("Admin");

        expect(await screen.findByRole("tab", { name: "Prompt Caching", selected: true })).toBeInTheDocument();
        expect(screen.getByTestId("caching-tab")).toBeInTheDocument();
        expect(onUrlUpdate).not.toHaveBeenCalled();
      },
    );

    it.each(["?tab=bogus", "?tab=CACHING"])("falls back to Overall and clears %s", async (searchParams) => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("Admin", { searchParams, onUrlUpdate, resetUrlUpdateQueueOnMount: false });

      expect(screen.getByRole("tab", { name: "Overall", selected: true })).toBeInTheDocument();
      expect(screen.queryByTestId("caching-tab")).not.toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    });

    it("uses the first ?tab= when the param repeats", () => {
      renderView("Admin", { searchParams: "?tab=caching&tab=compression" });

      expect(screen.getByRole("tab", { name: "Prompt Caching", selected: true })).toBeInTheDocument();
    });
  });
});
