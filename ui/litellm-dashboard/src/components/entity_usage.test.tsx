import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import EntityUsage from "./entity_usage";
import * as networking from "./networking";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

// Mock the networking module
vi.mock("./networking", () => ({
  tagDailyActivityCall: vi.fn(),
  teamDailyActivityCall: vi.fn(),
  organizationDailyActivityCall: vi.fn(),
}));

// Mock the child components to simplify testing
vi.mock("./activity_metrics", () => ({
  ActivityMetrics: () => <div>Activity Metrics</div>,
  processActivityData: () => ({ data: [], metadata: {} }),
}));

vi.mock("./top_key_view", () => ({
  default: () => <div>Top Keys</div>,
}));

vi.mock("./top_model_view", () => ({
  default: () => <div>Top Models</div>,
}));

vi.mock("./EntityUsageExport", () => ({
  UsageExportHeader: () => <div>Usage Export Header</div>,
}));

describe("EntityUsage", () => {
  const mockTagDailyActivityCall = vi.mocked(networking.tagDailyActivityCall);
  const mockTeamDailyActivityCall = vi.mocked(networking.teamDailyActivityCall);
  const mockOrganizationDailyActivityCall = vi.mocked(networking.organizationDailyActivityCall);

  const mockSpendData = {
    results: [
      {
        date: "2025-01-01",
        metrics: {
          spend: 100.5,
          api_requests: 1000,
          successful_requests: 950,
          failed_requests: 50,
          total_tokens: 50000,
          prompt_tokens: 30000,
          completion_tokens: 20000,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
        breakdown: {
          entities: {
            "tag-1": {
              metrics: {
                spend: 60.3,
                api_requests: 600,
                successful_requests: 570,
                failed_requests: 30,
                total_tokens: 30000,
                prompt_tokens: 18000,
                completion_tokens: 12000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
              metadata: {
                team_alias: "Tag 1",
              },
              api_key_breakdown: {},
            },
          },
          models: {},
          api_keys: {},
          providers: {
            openai: {
              metrics: {
                spend: 100.5,
                api_requests: 1000,
                successful_requests: 950,
                failed_requests: 50,
                total_tokens: 50000,
                prompt_tokens: 30000,
                completion_tokens: 20000,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          },
        },
      },
    ],
    metadata: {
      total_spend: 100.5,
      total_api_requests: 1000,
      total_successful_requests: 950,
      total_failed_requests: 50,
      total_tokens: 50000,
    },
  };

  const defaultProps = {
    accessToken: "test-token",
    entityType: "tag" as const,
    entityId: "test-tag",
    userID: "user-123",
    userRole: "Admin",
    entityList: [
      { label: "Tag 1", value: "tag-1" },
      { label: "Tag 2", value: "tag-2" },
    ],
    premiumUser: true,
    dateValue: {
      from: new Date("2025-01-01"),
      to: new Date("2025-01-31"),
    },
  };

  beforeEach(() => {
    mockTagDailyActivityCall.mockClear();
    mockTeamDailyActivityCall.mockClear();
    mockOrganizationDailyActivityCall.mockClear();
    mockTagDailyActivityCall.mockResolvedValue(mockSpendData);
    mockTeamDailyActivityCall.mockResolvedValue(mockSpendData);
    mockOrganizationDailyActivityCall.mockResolvedValue(mockSpendData);
  });

  it("should render with tag entity type and display spend metrics", async () => {
    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    // Check that spend metrics are displayed
    expect(screen.getByText("Tag Spend Overview")).toBeInTheDocument();
    expect(screen.getByText("Total Spend")).toBeInTheDocument();

    // Use getAllByText since $100.50 appears in multiple places
    const spendElements = screen.getAllByText("$100.50");
    expect(spendElements.length).toBeGreaterThan(0);

    expect(screen.getByText("1,000")).toBeInTheDocument(); // Total Requests
  });

  it("should render with team entity type and call team API", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
    });

    // Check that it shows team-specific label
    expect(screen.getByText("Team Spend Overview")).toBeInTheDocument();

    // Use getAllByText since $100.50 appears in multiple places
    const spendElements = screen.getAllByText("$100.50");
    expect(spendElements.length).toBeGreaterThan(0);
  });

  it("should render with organization entity type and call organization API", async () => {
    const { getByText, getAllByText } = render(<EntityUsage {...defaultProps} entityType="organization" />);

    await waitFor(() => {
      expect(mockOrganizationDailyActivityCall).toHaveBeenCalled();
    });

    expect(getByText("Organization Spend Overview")).toBeInTheDocument();

    const spendElements = getAllByText("$100.50");
    expect(spendElements.length).toBeGreaterThan(0);
  });

  it("should switch between tabs", async () => {
    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    // Check default tab (Cost) is shown
    expect(screen.getByText("Tag Spend Overview")).toBeInTheDocument();

    // Click Model Activity tab
    const modelActivityTab = screen.getByText("Model Activity");
    fireEvent.click(modelActivityTab);

    // Should show activity metrics
    expect(screen.getAllByText("Activity Metrics")[0]).toBeInTheDocument();

    // Click Key Activity tab
    const keyActivityTab = screen.getByText("Key Activity");
    fireEvent.click(keyActivityTab);

    // Should show activity metrics again
    expect(screen.getAllByText("Activity Metrics")[1]).toBeInTheDocument();
  });

  it("should handle empty data gracefully", async () => {
    const emptyData = {
      results: [],
      metadata: {
        total_spend: 0,
        total_api_requests: 0,
        total_successful_requests: 0,
        total_failed_requests: 0,
        total_tokens: 0,
      },
    };

    mockTagDailyActivityCall.mockResolvedValue(emptyData);

    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    // Check that zero values are displayed (component formats it as $0.00)
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.getByText("Total Spend")).toBeInTheDocument();
  });
});
