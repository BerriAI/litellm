import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import CostSavingsView, { formatUsd } from "./CostSavingsView";

function renderView() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <CostSavingsView />
    </QueryClientProvider>,
  );
}

async function clickFilter(name: string) {
  const radio = screen.getByRole("radio", { name });
  const label = radio.closest("label");
  await userEvent.click(label ?? radio);
}

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("@/components/networking", () => ({
  costSavingsActivityCall: vi.fn(),
  costSavingsRecentRequestsCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: vi.fn(),
}));

vi.mock("@/components/shared/advanced_date_picker", async () => {
  const React = await import("react");
  const AdvancedDatePicker = () => React.createElement("div", { "data-testid": "advanced-date-picker" }, "Date Picker");
  AdvancedDatePicker.displayName = "AdvancedDatePicker";
  return { default: AdvancedDatePicker };
});

const ACTIVITY_RESPONSE: networking.CostSavingsActivityResponse = {
  results: [
    {
      date: "2026-07-15",
      metrics: {
        cache_savings: 1.5,
        compression_savings: 0.5,
        total_savings: 2.0,
        spend: 10.0,
        cache_read_input_tokens: 1000,
        cache_creation_input_tokens: 0,
        compression_saved_tokens: 250,
      },
    },
  ],
  totals: {
    cache_savings: 1.5,
    compression_savings: 0.5,
    total_savings: 2.0,
    spend: 10.0,
    cache_read_input_tokens: 1000,
    cache_creation_input_tokens: 0,
    compression_saved_tokens: 250,
  },
  unpriced_models: [],
};

const RECENT_RESPONSE: networking.RecentOptimizedRequestsResponse = {
  requests: [
    {
      request_id: "req_abc123",
      start_time: "2026-07-15T12:00:00+00:00",
      model: "claude-x",
      total_tokens: 1500,
      optimizations: ["caching", "compression"],
      original_cost: 0.05,
      optimized_cost: 0.02,
      savings: 0.03,
    },
  ],
  scanned_requests: 42,
};

const MIXED_RECENT_RESPONSE: networking.RecentOptimizedRequestsResponse = {
  requests: [
    {
      request_id: "req_cache_only",
      start_time: "2026-07-15T12:00:00+00:00",
      model: "claude-x",
      total_tokens: 1000,
      optimizations: ["caching"],
      original_cost: 0.04,
      optimized_cost: 0.03,
      savings: 0.01,
    },
    {
      request_id: "req_compress_only",
      start_time: "2026-07-15T12:01:00+00:00",
      model: "claude-y",
      total_tokens: 2000,
      optimizations: ["compression"],
      original_cost: 0.08,
      optimized_cost: 0.05,
      savings: 0.03,
    },
    {
      request_id: "req_both",
      start_time: "2026-07-15T12:02:00+00:00",
      model: "claude-z",
      total_tokens: 3000,
      optimizations: ["caching", "compression"],
      original_cost: 0.12,
      optimized_cost: 0.07,
      savings: 0.05,
    },
  ],
  scanned_requests: 99,
};

describe("formatUsd", () => {
  it("formats zero, cents, and sub-cent values", () => {
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(12.345)).toBe("$12.35");
    expect(formatUsd(0.002625)).toBe("$0.002625");
    expect(formatUsd(1234.5)).toBe("$1,234.50");
  });

  it("puts the sign before the dollar symbol for negative net savings", () => {
    expect(formatUsd(-1.23)).toBe("-$1.23");
    expect(formatUsd(-0.002625)).toBe("-$0.002625");
  });
});

describe("CostSavingsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAuthorized).mockReturnValue({
      accessToken: "sk-test",
      token: "token",
      userId: "user-1",
      userRole: "Admin",
      userEmail: null,
      premiumUser: false,
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    } as any);
    vi.mocked(networking.costSavingsActivityCall).mockResolvedValue(ACTIVITY_RESPONSE);
    vi.mocked(networking.costSavingsRecentRequestsCall).mockResolvedValue(RECENT_RESPONSE);
  });

  it("renders KPI totals from the activity response", async () => {
    renderView();
    await waitFor(() => {
      expect(screen.getAllByText("$2.00").length).toBeGreaterThan(0);
    });
    expect(screen.getByText("Total Savings")).toBeInTheDocument();
    expect(screen.getAllByText("$1.50").length).toBeGreaterThan(0);
    expect(screen.getAllByText("$0.50").length).toBeGreaterThan(0);
    expect(screen.getAllByText("$10.00").length).toBeGreaterThan(0);
  });

  it("renders recent optimized requests with type badges and savings", async () => {
    renderView();
    await waitFor(() => {
      expect(screen.getByText("req_abc123")).toBeInTheDocument();
    });
    expect(screen.getByText("caching")).toBeInTheDocument();
    expect(screen.getByText("compression")).toBeInTheDocument();
    expect(screen.getByText("$0.03")).toBeInTheDocument();
    expect(screen.getByText(/scanned last 42 requests/)).toBeInTheDocument();
  });

  it("shows a warning when models are missing prices", async () => {
    vi.mocked(networking.costSavingsActivityCall).mockResolvedValue({
      ...ACTIVITY_RESPONSE,
      unpriced_models: ["mystery-model"],
    });
    renderView();
    await waitFor(() => {
      expect(screen.getByText("Some models are missing prices")).toBeInTheDocument();
    });
    expect(screen.getByText(/mystery-model/)).toBeInTheDocument();
  });

  it("shows the empty state when no requests were optimized", async () => {
    vi.mocked(networking.costSavingsRecentRequestsCall).mockResolvedValue({ requests: [], scanned_requests: 0 });
    renderView();
    await waitFor(() => {
      expect(screen.getByText(/No optimized requests in this window/)).toBeInTheDocument();
    });
  });

  it("filters the recent requests table by optimization type", async () => {
    vi.mocked(networking.costSavingsRecentRequestsCall).mockResolvedValue(MIXED_RECENT_RESPONSE);
    renderView();
    await waitFor(() => {
      expect(screen.getByText("req_both")).toBeInTheDocument();
    });
    expect(screen.getByText("req_cache_only")).toBeInTheDocument();
    expect(screen.getByText("req_compress_only")).toBeInTheDocument();

    await clickFilter("Compression");

    expect(screen.queryByText("req_cache_only")).not.toBeInTheDocument();
    expect(screen.getByText("req_compress_only")).toBeInTheDocument();
    expect(screen.getByText("req_both")).toBeInTheDocument();

    await clickFilter("Caching");

    expect(screen.getByText("req_cache_only")).toBeInTheDocument();
    expect(screen.queryByText("req_compress_only")).not.toBeInTheDocument();
    expect(screen.getByText("req_both")).toBeInTheDocument();
  });

  it("shows a type-specific empty state when the filter excludes every request", async () => {
    vi.mocked(networking.costSavingsRecentRequestsCall).mockResolvedValue({
      requests: [
        {
          request_id: "req_cache_only",
          start_time: "2026-07-15T12:00:00+00:00",
          model: "claude-x",
          total_tokens: 1000,
          optimizations: ["caching"],
          original_cost: 0.04,
          optimized_cost: 0.03,
          savings: 0.01,
        },
      ],
      scanned_requests: 7,
    });
    renderView();
    await waitFor(() => {
      expect(screen.getByText("req_cache_only")).toBeInTheDocument();
    });

    await clickFilter("Compression");

    expect(screen.queryByText("req_cache_only")).not.toBeInTheDocument();
    expect(screen.getByText("No compression requests in this window.")).toBeInTheDocument();
  });

  it("scopes the KPI cards and charts to the selected optimization type", async () => {
    renderView();
    await waitFor(() => {
      expect(screen.getAllByText("$2.00").length).toBeGreaterThan(0);
    });
    expect(screen.getByText("Savings Distribution")).toBeInTheDocument();

    await clickFilter("Compression");

    expect(screen.queryByText("Total Savings")).not.toBeInTheDocument();
    expect(screen.queryByText("Caching Savings")).not.toBeInTheDocument();
    expect(screen.getByText("Compression Savings")).toBeInTheDocument();
    expect(screen.getAllByText("$0.50").length).toBeGreaterThan(0);
    expect(screen.getByText("Tokens Compressed Away")).toBeInTheDocument();
    expect(screen.getByText("250")).toBeInTheDocument();
    expect(screen.queryByText("Savings Distribution")).not.toBeInTheDocument();

    await clickFilter("Caching");

    expect(screen.getByText("Caching Savings")).toBeInTheDocument();
    expect(screen.queryByText("Compression Savings")).not.toBeInTheDocument();
    expect(screen.getByText("Cached Tokens Read")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();

    await clickFilter("All");

    expect(screen.getByText("Total Savings")).toBeInTheDocument();
    expect(screen.getByText("Savings Distribution")).toBeInTheDocument();
  });
});
