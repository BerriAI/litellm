import React from "react";
import { fireEvent, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

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

const renderView = (userRole = "Admin") => {
  useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <CostOptimizationView accessToken="test-token" userId="u1" userRole={userRole} />
    </QueryClientProvider>,
  );
};

describe("CostOptimizationView", () => {
  beforeEach(() => {
    useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole: "Admin" });
  });

  it("renders the standard page header with the sidebar's Cost Optimization icon", () => {
    const { container, getByRole, getByText } = renderView();

    expect(getByRole("heading", { level: 1, name: "Cost Optimization" })).toBeInTheDocument();
    expect(getByText(/Track and configure the mechanisms that save you money/)).toBeInTheDocument();
    expect(container.querySelector(".lucide-piggy-bank")).not.toBeNull();
  });

  it("renders the four cost-optimization tabs", () => {
    const { getByText } = renderView();

    expect(getByText("Overall")).toBeInTheDocument();
    expect(getByText("Prompt Compression")).toBeInTheDocument();
    expect(getByText("Prompt Caching")).toBeInTheDocument();
    expect(getByText("Auto-Router")).toBeInTheDocument();
  });

  it("defaults to the Overall tab and switches the active tab on click", () => {
    const { getByRole } = renderView();

    expect(getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "false");

    fireEvent.click(getByRole("tab", { name: "Prompt Compression" }));

    expect(getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "false");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "true");
  });

  // Unlike the other three pages in this cleanup, Cost Optimization keeps its
  // nav entry for internal users: the Overall tab runs on /user/daily/activity,
  // which every role may call. Only the tabs reading proxy-wide config and
  // telemetry (/config/list, /auto_router/benchmarks, guardrail management)
  // are proxy-admin-only, so those are what disappear.
  describe("proxy-admin-only tabs", () => {
    it.each(["Internal User", "Internal Viewer", "Org Admin"])("shows %s the Overall tab only", (userRole) => {
      const { getByRole, queryByRole } = renderView(userRole);

      expect(getByRole("tab", { name: "Overall" })).toBeInTheDocument();
      expect(queryByRole("tab", { name: "Prompt Compression" })).not.toBeInTheDocument();
      expect(queryByRole("tab", { name: "Prompt Caching" })).not.toBeInTheDocument();
      expect(queryByRole("tab", { name: "Auto-Router" })).not.toBeInTheDocument();
    });

    it("never mounts the panels behind the admin-only endpoints for an internal user", () => {
      const { getByTestId, queryByTestId } = renderView("Internal User");

      expect(getByTestId("usage-tab")).toBeInTheDocument();
      expect(queryByTestId("compression-tab")).not.toBeInTheDocument();
      expect(queryByTestId("caching-tab")).not.toBeInTheDocument();
      expect(queryByTestId("autorouter-benchmarks-tab")).not.toBeInTheDocument();
    });
  });
});
