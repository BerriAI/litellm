import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders } from "../../../../../tests/test-utils";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

vi.mock("@/components/networking", () => ({
  organizationListCall: vi.fn().mockResolvedValue([]),
  userDailyActivityCall: vi
    .fn()
    .mockResolvedValue({ results: [], metadata: { total_pages: 1, has_more: false, page: 1 } }),
  userDailyActivityAggregatedCall: vi
    .fn()
    .mockResolvedValue({ results: [], metadata: { total_pages: 1, has_more: false, page: 1 } }),
}));

vi.mock("./UsageTab", () => ({ __esModule: true, default: () => <div data-testid="usage-tab" /> }));
vi.mock("./PromptCompressionTab", () => ({ __esModule: true, default: () => <div data-testid="compression-tab" /> }));
vi.mock("./PromptCachingTab", () => ({ __esModule: true, default: () => <div data-testid="caching-tab" /> }));
vi.mock("./AutoRouterBenchmarksTab", () => ({
  __esModule: true,
  default: () => <div data-testid="autorouter-benchmarks-tab" />,
}));

import CostOptimizationView from "./CostOptimizationView";

const renderView = (userRole = "Admin", url: { searchParams?: string; onUrlUpdate?: OnUrlUpdateFunction } = {}) => {
  useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderWithProviders(
    <QueryClientProvider client={queryClient}>
      <CostOptimizationView accessToken="test-token" userId="u1" userRole={userRole} />
    </QueryClientProvider>,
    url,
  );
};

const renderViewKeepingMountUpdates = (userRole: string, searchParams: string, onUrlUpdate: OnUrlUpdateFunction) => {
  useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<CostOptimizationView accessToken="test-token" userId="u1" userRole={userRole} />, {
    wrapper: ({ children }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </NuqsTestingAdapter>
    ),
  });
};

const lastSearchParams = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

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
    it("opens the tab named in ?tab=", () => {
      renderView("Admin", { searchParams: "?tab=caching" });

      expect(screen.getByRole("tab", { name: "Prompt Caching" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "false");
      expect(screen.getByTestId("caching-tab")).toBeInTheDocument();
    });

    it("writes ?tab= when a tab is chosen and drops it for Overall", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("Admin", { onUrlUpdate });

      fireEvent.click(screen.getByRole("tab", { name: "Auto-Router" }));
      await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("tab")).toBe("autorouter-usage"));

      fireEvent.click(screen.getByRole("tab", { name: "Overall" }));
      await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.has("tab")).toBe(false));
    });

    it("keeps the tab opened from the URL mounted after switching away", () => {
      renderView("Admin", { searchParams: "?tab=caching" });

      fireEvent.click(screen.getByRole("tab", { name: "Overall" }));

      expect(screen.getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByTestId("caching-tab")).toBeInTheDocument();
      expect(screen.queryByTestId("compression-tab")).not.toBeInTheDocument();
    });

    it.each([
      ["an admin-only tab for an internal user", "Internal User", "?tab=caching"],
      ["an unknown tab for an admin", "Admin", "?tab=bogus"],
    ])("falls back to Overall and clears %s", async (_label, userRole, searchParams) => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderViewKeepingMountUpdates(userRole, searchParams, onUrlUpdate);

      expect(screen.getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByTestId("caching-tab")).not.toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastSearchParams(onUrlUpdate)?.has("tab")).toBe(false);
    });
  });
});
