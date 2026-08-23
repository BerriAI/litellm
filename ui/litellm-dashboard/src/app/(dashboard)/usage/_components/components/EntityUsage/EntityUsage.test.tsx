import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import * as networking from "@/components/networking";
import EntityUsage from "./EntityUsage";

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
vi.mock("@/components/networking", () => ({
  tagDailyActivityCall: vi.fn(),
  teamDailyActivityCall: vi.fn(),
  teamDailyActivityAggregatedCall: vi.fn(),
  organizationDailyActivityCall: vi.fn(),
  customerDailyActivityCall: vi.fn(),
  agentDailyActivityCall: vi.fn(),
  userDailyActivityCall: vi.fn(),
}));

// Mock the child components to simplify testing
vi.mock("@/components/activity_metrics", () => ({
  ActivityMetrics: ({ modelMetrics }: { modelMetrics?: { __source?: string } }) => (
    <div>
      <span>Activity Metrics</span>
      <span>{`metrics-source:${modelMetrics?.__source ?? "none"}`}</span>
    </div>
  ),
  processActivityData: (_data: unknown, key: string) => ({ __source: key }),
}));

vi.mock("../EndpointUsage/EndpointUsage", () => ({
  default: () => <div>Endpoint Usage Panel</div>,
}));

vi.mock("@/components/UsagePage/components/EntityUsage/TopKeyView", () => ({
  default: ({ topKeys }: { topKeys: { api_key: string; spend: number }[] }) => (
    <div>
      <span>Top Keys</span>
      <span>{`top-keys:${topKeys.map((row) => `${row.api_key}=${row.spend}`).join("|")}`}</span>
    </div>
  ),
}));

vi.mock("./TopModelView", () => ({
  default: ({ topModels }: { topModels: { key: string; spend: number }[] }) => (
    <div>
      <span>Top Models</span>
      <span>{`top-models:${topModels.map((row) => `${row.key}=${row.spend}`).join("|")}`}</span>
    </div>
  ),
}));

vi.mock("@/components/EntityUsageExport/EntityUsageExportModal", () => ({
  default: () => <div>Entity Usage Export Modal</div>,
}));

vi.mock("@/components/EntityUsageExport", () => ({
  UsageExportHeader: ({ filterLabel, filterSlot }: { filterLabel?: string; filterSlot?: ReactNode }) => (
    <div>
      <span>Usage Export Header</span>
      <span>{filterLabel}</span>
      {filterSlot}
    </div>
  ),
}));

vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: vi.fn(),
  useUserLookup: vi.fn(() => ({ data: null })),
}));

vi.mock("@/components/common_components/team_multi_select", () => ({
  default: () => <div>Team Multi Select</div>,
}));

// Mock useTeams hook
vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(() => ({
    teams: [],
    setTeams: vi.fn(),
  })),
}));

describe("EntityUsage", () => {
  const mockTagDailyActivityCall = vi.mocked(networking.tagDailyActivityCall);
  const mockTeamDailyActivityCall = vi.mocked(networking.teamDailyActivityCall);
  const mockTeamDailyActivityAggregatedCall = vi.mocked(networking.teamDailyActivityAggregatedCall);
  const mockOrganizationDailyActivityCall = vi.mocked(networking.organizationDailyActivityCall);
  const mockCustomerDailyActivityCall = vi.mocked(networking.customerDailyActivityCall);
  const mockAgentDailyActivityCall = vi.mocked(networking.agentDailyActivityCall);
  const mockUserDailyActivityCall = vi.mocked(networking.userDailyActivityCall);
  const mockUseInfiniteUsers = vi.mocked(useInfiniteUsers);

  const infiniteUsersResult = (users: { user_id: string; user_alias: string | null; user_email: string | null }[]) =>
    ({
      data: { pages: [{ users, page: 1, total_pages: 1, total_count: users.length }], pageParams: [1] },
      fetchNextPage: vi.fn(),
      hasNextPage: false,
      isFetchingNextPage: false,
      isLoading: false,
    }) as unknown as ReturnType<typeof useInfiniteUsers>;

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

  const mockAgentSpendData = {
    results: [
      {
        date: "2025-01-01",
        metrics: {
          spend: 245.8,
          api_requests: 3200,
          successful_requests: 3100,
          failed_requests: 100,
          total_tokens: 1250000,
          prompt_tokens: 850000,
          completion_tokens: 400000,
          cache_read_input_tokens: 50000,
          cache_creation_input_tokens: 10000,
        },
        breakdown: {
          entities: {
            "agent-code-review": {
              metrics: {
                spend: 120.4,
                api_requests: 1500,
                successful_requests: 1450,
                failed_requests: 50,
                total_tokens: 620000,
                prompt_tokens: 420000,
                completion_tokens: 200000,
                cache_read_input_tokens: 30000,
                cache_creation_input_tokens: 5000,
              },
              metadata: { agent_name: "Code Review Agent" },
              api_key_breakdown: {},
            },
            "agent-customer-support": {
              metrics: {
                spend: 85.2,
                api_requests: 1200,
                successful_requests: 1170,
                failed_requests: 30,
                total_tokens: 430000,
                prompt_tokens: 290000,
                completion_tokens: 140000,
                cache_read_input_tokens: 15000,
                cache_creation_input_tokens: 3000,
              },
              metadata: { agent_name: "Customer Support Agent" },
              api_key_breakdown: {},
            },
            "agent-data-analyst": {
              metrics: {
                spend: 40.2,
                api_requests: 500,
                successful_requests: 480,
                failed_requests: 20,
                total_tokens: 200000,
                prompt_tokens: 140000,
                completion_tokens: 60000,
                cache_read_input_tokens: 5000,
                cache_creation_input_tokens: 2000,
              },
              metadata: { agent_name: "Data Analyst Agent" },
              api_key_breakdown: {},
            },
          },
          models: {
            "gpt-4o": {
              metrics: {
                spend: 180.0,
                api_requests: 2000,
                successful_requests: 1950,
                failed_requests: 50,
                total_tokens: 900000,
                prompt_tokens: 600000,
                completion_tokens: 300000,
                cache_read_input_tokens: 40000,
                cache_creation_input_tokens: 8000,
              },
              metadata: {},
              api_key_breakdown: {},
            },
            "claude-sonnet-4-20250514": {
              metrics: {
                spend: 65.8,
                api_requests: 1200,
                successful_requests: 1150,
                failed_requests: 50,
                total_tokens: 350000,
                prompt_tokens: 250000,
                completion_tokens: 100000,
                cache_read_input_tokens: 10000,
                cache_creation_input_tokens: 2000,
              },
              metadata: {},
              api_key_breakdown: {},
            },
          },
          api_keys: {},
          providers: {
            openai: {
              metrics: {
                spend: 180.0,
                api_requests: 2000,
                successful_requests: 1950,
                failed_requests: 50,
                total_tokens: 900000,
                prompt_tokens: 600000,
                completion_tokens: 300000,
                cache_read_input_tokens: 40000,
                cache_creation_input_tokens: 8000,
              },
            },
            anthropic: {
              metrics: {
                spend: 65.8,
                api_requests: 1200,
                successful_requests: 1150,
                failed_requests: 50,
                total_tokens: 350000,
                prompt_tokens: 250000,
                completion_tokens: 100000,
                cache_read_input_tokens: 10000,
                cache_creation_input_tokens: 2000,
              },
            },
          },
        },
      },
      {
        date: "2025-01-02",
        metrics: {
          spend: 198.5,
          api_requests: 2800,
          successful_requests: 2720,
          failed_requests: 80,
          total_tokens: 980000,
          prompt_tokens: 670000,
          completion_tokens: 310000,
          cache_read_input_tokens: 42000,
          cache_creation_input_tokens: 9000,
        },
        breakdown: {
          entities: {
            "agent-code-review": {
              metrics: {
                spend: 95.3,
                api_requests: 1300,
                successful_requests: 1270,
                failed_requests: 30,
                total_tokens: 510000,
                prompt_tokens: 350000,
                completion_tokens: 160000,
                cache_read_input_tokens: 25000,
                cache_creation_input_tokens: 4000,
              },
              metadata: { agent_name: "Code Review Agent" },
              api_key_breakdown: {},
            },
            "agent-customer-support": {
              metrics: {
                spend: 68.7,
                api_requests: 1000,
                successful_requests: 970,
                failed_requests: 30,
                total_tokens: 320000,
                prompt_tokens: 220000,
                completion_tokens: 100000,
                cache_read_input_tokens: 12000,
                cache_creation_input_tokens: 3000,
              },
              metadata: { agent_name: "Customer Support Agent" },
              api_key_breakdown: {},
            },
            "agent-data-analyst": {
              metrics: {
                spend: 34.5,
                api_requests: 500,
                successful_requests: 480,
                failed_requests: 20,
                total_tokens: 150000,
                prompt_tokens: 100000,
                completion_tokens: 50000,
                cache_read_input_tokens: 5000,
                cache_creation_input_tokens: 2000,
              },
              metadata: { agent_name: "Data Analyst Agent" },
              api_key_breakdown: {},
            },
          },
          models: {},
          api_keys: {},
          providers: {},
        },
      },
    ],
    metadata: {
      total_spend: 444.3,
      total_api_requests: 6000,
      total_successful_requests: 5820,
      total_failed_requests: 180,
      total_tokens: 2230000,
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
    mockTeamDailyActivityAggregatedCall.mockClear();
    mockOrganizationDailyActivityCall.mockClear();
    mockCustomerDailyActivityCall.mockClear();
    mockAgentDailyActivityCall.mockClear();
    mockUserDailyActivityCall.mockClear();
    mockTagDailyActivityCall.mockResolvedValue(mockSpendData);
    mockTeamDailyActivityCall.mockResolvedValue(mockSpendData);
    mockTeamDailyActivityAggregatedCall.mockResolvedValue(mockSpendData);
    mockOrganizationDailyActivityCall.mockResolvedValue(mockSpendData);
    mockCustomerDailyActivityCall.mockResolvedValue(mockSpendData);
    mockAgentDailyActivityCall.mockResolvedValue(mockAgentSpendData);
    mockUserDailyActivityCall.mockResolvedValue(mockSpendData);
    mockUseInfiniteUsers.mockClear();
    mockUseInfiniteUsers.mockReturnValue(
      infiniteUsersResult([
        { user_id: "user-001", user_alias: "Alice", user_email: "alice@example.com" },
        { user_id: "user-002", user_alias: null, user_email: "bob@example.com" },
      ]),
    );
  });

  it("should render with tag entity type and display spend metrics", async () => {
    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Tag Spend Overview")).toBeInTheDocument();
    expect(screen.getByText("Total Spend")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$100.50");
      expect(spendElements.length).toBeGreaterThan(0);
    });

    expect(screen.getByText("1,000")).toBeInTheDocument(); // Total Requests
  });

  it("should render with team entity type and call team API", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    // Check that it shows team-specific label
    expect(screen.getByText("Team Spend Overview")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$100.50");
      expect(spendElements.length).toBeGreaterThan(0);
    });
  });

  it("should render with organization entity type and call organization API", async () => {
    render(<EntityUsage {...defaultProps} entityType="organization" />);

    await waitFor(() => {
      expect(mockOrganizationDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Organization Spend Overview")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$100.50");
      expect(spendElements.length).toBeGreaterThan(0);
    });
  });

  it("should render with customer entity type and call customer API", async () => {
    render(<EntityUsage {...defaultProps} entityType="customer" />);

    await waitFor(() => {
      expect(mockCustomerDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Customer Spend Overview")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$100.50");
      expect(spendElements.length).toBeGreaterThan(0);
    });
  });

  it("should render with agent entity type and call agent API", async () => {
    render(<EntityUsage {...defaultProps} entityType="agent" />);

    await waitFor(() => {
      expect(mockAgentDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Agent Spend Overview")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$444.30");
      expect(spendElements.length).toBeGreaterThan(0);
    });
  });

  it("should render with user entity type and call user API", async () => {
    render(<EntityUsage {...defaultProps} entityType="user" />);

    await waitFor(() => {
      expect(mockUserDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("User Spend Overview")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$100.50");
      expect(spendElements.length).toBeGreaterThan(0);
    });
  });

  it("should switch between tabs", async () => {
    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Tag Spend Overview")).toBeInTheDocument();

    const modelActivityTab = screen.getByText("Model Activity");
    act(() => {
      fireEvent.click(modelActivityTab);
    });

    expect(screen.getAllByText("Activity Metrics")[0]).toBeInTheDocument();

    const keyActivityTab = screen.getByText("Key Activity");
    act(() => {
      fireEvent.click(keyActivityTab);
    });

    expect(screen.getAllByText("Activity Metrics")[1]).toBeInTheDocument();
  });

  // An inactive tab panel is marked aria-selected="false" by one tab library and hidden by the
  // other, so treat either as "not on screen" and the assertion holds whichever one is rendering.
  const isShowing = (element: HTMLElement): boolean => {
    for (let node: HTMLElement | null = element; node; node = node.parentElement) {
      if (node.hasAttribute("hidden")) return false;
      if (node.getAttribute("aria-selected") === "false") return false;
    }
    return true;
  };

  const showingCount = (marker: string): number => screen.queryAllByText(marker).filter(isShowing).length;

  const showingText = (text: string): HTMLElement => {
    const [element] = screen.getAllByText(text).filter(isShowing);
    expect(element).toBeDefined();
    return element;
  };

  const NON_TEAM_PANELS: [string, string][] = [
    ["Cost", "Tag Spend Overview"],
    ["Model Activity", "metrics-source:model_groups"],
    ["Key Activity", "metrics-source:api_keys"],
    ["Endpoint Activity", "Endpoint Usage Panel"],
  ];

  it.each(NON_TEAM_PANELS)("shows only the %s panel for a non-team entity type", async (tabLabel, marker) => {
    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    act(() => {
      fireEvent.click(screen.getByText(tabLabel));
    });

    expect(showingCount(marker)).toBeGreaterThan(0);
    for (const [otherLabel, otherMarker] of NON_TEAM_PANELS) {
      if (otherLabel === tabLabel) continue;
      expect(showingCount(otherMarker)).toBe(0);
    }
  });

  const TEAM_PANELS: [string, string][] = [
    ["Cost", "Team Spend Overview"],
    ["Model Activity", "metrics-source:model_groups"],
    ["Agent Activity", "metrics-source:entities"],
    ["Key Activity", "metrics-source:api_keys"],
    ["Endpoint Activity", "Endpoint Usage Panel"],
  ];

  it.each(TEAM_PANELS)("shows only the %s panel for the team entity type", async (tabLabel, marker) => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    act(() => {
      fireEvent.click(screen.getByText(tabLabel));
    });

    expect(showingCount(marker)).toBeGreaterThan(0);
    for (const [otherLabel, otherMarker] of TEAM_PANELS) {
      if (otherLabel === tabLabel) continue;
      expect(showingCount(otherMarker)).toBe(0);
    }
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

    expect(await screen.findByText("Tag Spend Overview")).toBeInTheDocument();
    expect(await screen.findByText("$0.00")).toBeInTheDocument();
    expect(screen.getByText("Total Spend")).toBeInTheDocument();
    expect(screen.getAllByText("0")[0]).toBeInTheDocument();
  });

  it("should display Model Activity tab for non-agent entity types", async () => {
    render(<EntityUsage {...defaultProps} entityType="tag" />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Model Activity")).toBeInTheDocument();
  });

  it("should display Request / Token Consumption tab for agent entity type", async () => {
    render(<EntityUsage {...defaultProps} entityType="agent" />);

    await waitFor(() => {
      expect(mockAgentDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Request / Token Consumption")).toBeInTheDocument();
  });

  it("should display Top Public Model Names title for non-agent entity types", async () => {
    render(<EntityUsage {...defaultProps} entityType="tag" />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Top Public Model Names")).toBeInTheDocument();
  });

  it("defaults Model Activity to public model names and toggles to litellm models", async () => {
    const { container } = render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    act(() => {
      fireEvent.click(screen.getByText("Model Activity"));
    });

    expect(showingCount("metrics-source:model_groups")).toBeGreaterThan(0);

    act(() => {
      fireEvent.click(showingText("Litellm Model Name"));
    });

    expect(showingCount("metrics-source:models")).toBeGreaterThan(0);

    act(() => {
      fireEvent.click(showingText("Public Model Name"));
    });

    expect(showingCount("metrics-source:model_groups")).toBeGreaterThan(0);
  });

  it("should display Top Agents title for agent entity type", async () => {
    render(<EntityUsage {...defaultProps} entityType="agent" />);

    await waitFor(() => {
      expect(mockAgentDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Top Agents")).toBeInTheDocument();
  });

  it("should use entityList label when entityList is provided and entity exists", async () => {
    const customEntityList = [
      { label: "Custom Tag Label", value: "tag-1" },
      { label: "Tag 2", value: "tag-2" },
    ];

    render(<EntityUsage {...defaultProps} entityList={customEntityList} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText("Custom Tag Label")).toBeInTheDocument();
    });
  });

  it("should fallback to team_alias when entityList is provided but entity does not exist", async () => {
    const customEntityList = [{ label: "Tag 2", value: "tag-2" }];

    render(<EntityUsage {...defaultProps} entityList={customEntityList} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getAllByText("Tag 1").length).toBeGreaterThan(0);
    });
  });

  it("should fallback to team_alias when entityList is null", async () => {
    render(<EntityUsage {...defaultProps} entityList={null} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getAllByText("Tag 1").length).toBeGreaterThan(0);
    });
  });

  it("should display Agent Activity tab for team entity type", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Agent Activity")).toBeInTheDocument();
  });

  it("should not display Agent Activity tab for non-team entity types", async () => {
    render(<EntityUsage {...defaultProps} entityType="tag" />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.queryByText("Agent Activity")).not.toBeInTheDocument();
  });

  it("should display Top Agents Driving Spend card for team entity type", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Top Agents Driving Spend")).toBeInTheDocument();
  });

  it("should not display Top Agents Driving Spend card for non-team entity types", async () => {
    render(<EntityUsage {...defaultProps} entityType="tag" />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.queryByText("Top Agents Driving Spend")).not.toBeInTheDocument();
  });

  it("should fetch agent activity data when entity type is team", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockAgentDailyActivityCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(Date),
        expect.any(Date),
        1,
        null,
      );
    });
  });

  it("should not fetch agent activity data for non-team entity types", async () => {
    render(<EntityUsage {...defaultProps} entityType="tag" />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(mockAgentDailyActivityCall).not.toHaveBeenCalled();
  });

  it("should switch to Agent Activity tab for team entity type", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
    });

    const agentActivityTab = screen.getByText("Agent Activity");
    act(() => {
      fireEvent.click(agentActivityTab);
    });

    await waitFor(() => {
      expect(screen.getAllByText("Activity Metrics").length).toBeGreaterThan(0);
    });
  });

  it("should fallback to entity value when no entityList and no team_alias", async () => {
    const spendDataWithoutAlias = {
      ...mockSpendData,
      results: [
        {
          ...mockSpendData.results[0],
          breakdown: {
            ...mockSpendData.results[0].breakdown,
            entities: {
              "tag-1": {
                ...mockSpendData.results[0].breakdown.entities["tag-1"],
                metadata: {},
              },
            },
          },
        },
      ],
    };

    mockTagDailyActivityCall.mockResolvedValue(spendDataWithoutAlias);

    render(<EntityUsage {...defaultProps} entityList={null} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getAllByText("tag-1").length).toBeGreaterThan(0);
    });
  });

  it("renders daily spend bars, per-entity bars, and the provider donut with cyan fills and a $ center total", async () => {
    const { container } = render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(container.querySelectorAll("path.recharts-rectangle")).toHaveLength(2);
    });

    const barFills = new Set(
      Array.from(container.querySelectorAll("path.recharts-rectangle")).map((rect) => rect.getAttribute("fill")),
    );
    expect(barFills).toEqual(new Set(["var(--color-cyan-500, #06b6d4)"]));

    expect(screen.getAllByText("2025-01-01").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Tag 1").length).toBeGreaterThan(1);

    const sectors = container.querySelectorAll(".recharts-pie-sector path");
    expect(sectors).toHaveLength(1);
    expect(sectors[0]).toHaveAttribute("fill", "var(--color-cyan-500, #06b6d4)");

    const centerLabels = Array.from(container.querySelectorAll("text.fill-foreground")).map((text) => text.textContent);
    expect(centerLabels).toContain("$100.50");
  });

  it("should label the chart with user_email metadata instead of the raw UUID (LIT-3889)", async () => {
    const userUuid = "c0e68be8-057e-4e2f-9d3a-000000000000";
    const spendDataForUser = {
      ...mockSpendData,
      results: [
        {
          ...mockSpendData.results[0],
          breakdown: {
            ...mockSpendData.results[0].breakdown,
            entities: {
              [userUuid]: {
                ...mockSpendData.results[0].breakdown.entities["tag-1"],
                metadata: { user_email: "spender@example.com" },
              },
            },
          },
        },
      ],
    };

    mockUserDailyActivityCall.mockResolvedValue(spendDataForUser);

    // entityList is null to simulate a spender missing from the paginated user list
    render(<EntityUsage {...defaultProps} entityType="user" entityList={null} />);

    await waitFor(() => {
      expect(mockUserDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText("spender@example.com")).toBeInTheDocument();
    });
    expect(screen.queryByText(userUuid)).not.toBeInTheDocument();
  });

  it("renders the provider spend table logo from the bundled provider map", async () => {
    render(<EntityUsage {...defaultProps} />);

    const logo = await screen.findByAltText("openai logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("openai_small"));
  });

  describe("capability gating", () => {
    it.each([
      ["organization", () => mockOrganizationDailyActivityCall, "Organization Spend Overview"],
      ["agent", () => mockAgentDailyActivityCall, "Agent Spend Overview"],
    ] as const)("fetches %s activity for an admin but not for an internal user", async (entityType, call, heading) => {
      render(<EntityUsage {...defaultProps} entityType={entityType} />);
      await waitFor(() => {
        expect(call()).toHaveBeenCalled();
      });

      cleanup();
      call().mockClear();

      render(<EntityUsage {...defaultProps} entityType={entityType} userRole="Internal User" />);
      expect(await screen.findByText(heading)).toBeInTheDocument();
      expect(call()).not.toHaveBeenCalled();
    });

    // An org admin's session role is "Internal User", so the row above cannot
    // distinguish them. Gating the fetch on the session role alone left the
    // Organization Usage panel rendered but permanently empty, because the
    // request was never issued even though the proxy would have served it.
    it.each([
      ["organization", () => mockOrganizationDailyActivityCall, true],
      ["agent", () => mockAgentDailyActivityCall, false],
    ] as const)("fetches %s activity for an org admin: %s", async (entityType, call, expected) => {
      render(<EntityUsage {...defaultProps} entityType={entityType} userRole="Internal User" isOrgAdmin={true} />);

      if (expected) {
        await waitFor(() => {
          expect(call()).toHaveBeenCalled();
        });
      } else {
        expect(await screen.findByText("Agent Spend Overview")).toBeInTheDocument();
        expect(call()).not.toHaveBeenCalled();
      }
    });

    it("keeps the team breakdown but drops its agent sub-fetch for an internal user", async () => {
      render(<EntityUsage {...defaultProps} entityType="team" userRole="Internal User" />);

      await waitFor(() => {
        expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
      });
      expect(screen.getByText("Team Spend Overview")).toBeInTheDocument();

      expect(mockAgentDailyActivityCall).not.toHaveBeenCalled();
      expect(screen.queryByText("Agent Activity")).not.toBeInTheDocument();
      expect(screen.queryByText("Top Agents Driving Spend")).not.toBeInTheDocument();
    });

    it("keeps the tag breakdown for an internal user", async () => {
      render(<EntityUsage {...defaultProps} entityType="tag" userRole="Internal User" />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalled();
      });
      expect(screen.getByText("Tag Spend Overview")).toBeInTheDocument();
    });
  });

  it("renders a letter avatar instead of an img for an unknown provider slug", async () => {
    const spendDataUnknownProvider = {
      ...mockSpendData,
      results: [
        {
          ...mockSpendData.results[0],
          breakdown: {
            ...mockSpendData.results[0].breakdown,
            providers: {
              "zzz-internal": mockSpendData.results[0].breakdown.providers.openai,
            },
          },
        },
      ],
    };
    mockTagDailyActivityCall.mockResolvedValue(spendDataUnknownProvider);

    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("zzz-internal").length).toBeGreaterThan(0);
    });
    expect(screen.queryByAltText("zzz-internal logo")).not.toBeInTheDocument();
    expect(screen.getByText("z")).toBeInTheDocument();
  });

  it("feeds the key, model and agent tables from their own breakdowns", async () => {
    const usageMetrics = {
      spend: 30.75,
      api_requests: 300,
      successful_requests: 290,
      failed_requests: 10,
      total_tokens: 15000,
      prompt_tokens: 9000,
      completion_tokens: 6000,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    };
    mockTeamDailyActivityAggregatedCall.mockResolvedValue({
      ...mockSpendData,
      results: [
        {
          ...mockSpendData.results[0],
          breakdown: {
            ...mockSpendData.results[0].breakdown,
            model_groups: { "gpt-4o": { metrics: { ...usageMetrics, spend: 70.25 }, metadata: {} } },
            api_keys: { "sk-abc": { metrics: usageMetrics, metadata: { key_alias: "prod-key", team_id: null } } },
          },
        },
      ],
    });

    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(screen.getByText("top-keys:sk-abc=30.75")).toBeInTheDocument();
    });
    expect(screen.getByText("top-models:gpt-4o=70.25")).toBeInTheDocument();
    expect(screen.getByText(/^top-models:Code Review Agent=/)).toBeInTheDocument();
  });

  it("uses the aggregated team endpoint and never drains paginated pages for teams", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityAggregatedCall).toHaveBeenCalled();
    });
    expect(mockTeamDailyActivityCall).not.toHaveBeenCalled();

    await waitFor(() => {
      expect(screen.getAllByText("$100.50").length).toBeGreaterThan(0);
    });
  });

  it("falls back to the paginated team endpoint when the aggregated call fails", async () => {
    mockTeamDailyActivityAggregatedCall.mockRejectedValue(new Error("aggregated unavailable"));

    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getAllByText("$100.50").length).toBeGreaterThan(0);
    });
  });

  describe("user filter (LIT-5654)", () => {
    const userDropdown = (): HTMLElement => screen.getByTestId("user-dropdown");
    const userCombobox = (): HTMLElement => within(userDropdown()).getByRole("combobox");

    const renderUserUsage = async () => {
      render(<EntityUsage {...defaultProps} entityType="user" entityList={null} />);
      await waitFor(() => {
        expect(mockUserDailyActivityCall).toHaveBeenCalled();
      });
    };

    it("offers a user filter even when the caller preloaded no user page", async () => {
      await renderUserUsage();

      expect(userCombobox()).toHaveAttribute("placeholder", "Search users by email…");
    });

    it("searches every user on the server rather than a preloaded page", async () => {
      const user = userEvent.setup();
      await renderUserUsage();

      expect(mockUseInfiniteUsers).toHaveBeenCalledWith(50, undefined);

      await user.type(userCombobox(), "alice");

      await waitFor(() => {
        expect(mockUseInfiniteUsers).toHaveBeenCalledWith(50, "alice");
      });
    });

    it("refetches daily activity for the picked user and drops the filter when cleared", async () => {
      const user = userEvent.setup();
      await renderUserUsage();

      expect(mockUserDailyActivityCall).toHaveBeenCalledWith("test-token", expect.any(Date), expect.any(Date), 1, null);

      await user.click(userCombobox());
      await user.click(await screen.findByText("Alice (user-001)"));

      await waitFor(() => {
        expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          1,
          "user-001",
        );
      });

      mockUserDailyActivityCall.mockClear();
      await user.click(userDropdown().querySelector('[data-slot="combobox-clear"]') as HTMLElement);

      await waitFor(() => {
        expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          1,
          null,
        );
      });
    });
  });
});
