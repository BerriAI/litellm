import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import * as networking from "@/components/networking";
import type {
  DailyData,
  KeyMetadata,
  KeyMetricWithMetadata,
  ModelActivityData,
  SpendMetrics,
} from "@/components/UsagePage/types";
import EntityUsage from "./EntityUsage";
import { getGlobalTopKeys, getTopAPIKeys } from "./entityUsageAggregations";

const emptySpendMetrics: SpendMetrics = {
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

const createKeyMetrics = (spend: number, metadata: KeyMetadata): KeyMetricWithMetadata => ({
  metrics: { ...emptySpendMetrics, spend },
  metadata,
});

const createDailyData = (date: string, apiKeys: Record<string, KeyMetricWithMetadata>): DailyData => ({
  date,
  metrics: { ...emptySpendMetrics },
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: apiKeys,
    entities: {},
  },
});

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
  dailyActivityAggregatedCall: vi.fn(),
  dailyActivityKeyPageCall: vi.fn(),
  dailyActivityKeySearchCall: vi.fn(),
  dailyActivityModelTopKeysCall: vi.fn(),
  dailyActivityExportCall: vi.fn(),
  tagListCall: vi.fn(),
}));

// Mock the child components to simplify testing
vi.mock("@/components/activity_metrics", () => ({
  ActivityMetrics: ({
    modelMetrics,
    summaryMetrics,
    summaryTitle = "Overall Usage",
    fetchTopApiKeys,
  }: {
    modelMetrics?: { __source?: string };
    summaryMetrics?: ModelActivityData;
    summaryTitle?: string;
    fetchTopApiKeys?: (model: string) => Promise<unknown>;
  }) => (
    <div>
      <span>Activity Metrics</span>
      <span>{`metrics-source:${modelMetrics?.__source ?? "none"}`}</span>
      {summaryMetrics !== undefined && <span>{summaryTitle}</span>}
      {fetchTopApiKeys !== undefined && <span>{`top-keys-fetcher:${modelMetrics?.__source ?? "none"}`}</span>}
    </div>
  ),
  processActivityData: (_data: unknown, key: string) => ({ __source: key }),
}));

vi.mock("../EndpointUsage/EndpointUsage", () => ({
  default: () => <div>Endpoint Usage Panel</div>,
}));

vi.mock("@/components/UsagePage/components/EntityUsage/TopKeyView", () => ({
  default: ({
    topKeys,
    tagsColumnHeader,
    formatTag,
  }: {
    topKeys: { api_key: string; user?: string | null; spend: number; tags?: { tag: string; usage: number }[] }[];
    tagsColumnHeader?: string;
    formatTag?: (tag: string) => string;
  }) => (
    <div>
      <span>Top Keys</span>
      {tagsColumnHeader && <span>{`top-keys-header:${tagsColumnHeader}`}</span>}
      <span>{`top-keys:${topKeys.map((row) => `${row.api_key}=${row.spend}=${row.user ?? "-"}`).join("|")}`}</span>
      {topKeys.flatMap((row) =>
        (row.tags ?? []).map((tag) => (
          <span key={`${row.api_key}-${tag.tag}`}>{formatTag ? formatTag(tag.tag) : tag.tag}</span>
        )),
      )}
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

vi.mock("./TeamUserSpendCard", () => ({
  default: ({ teamIds }: { teamIds: string[] }) => <div>{`team-user-spend:${teamIds.join("|")}`}</div>,
}));

vi.mock("@/components/EntityUsageExport/EntityUsageExportModal", () => ({
  default: () => <div>Entity Usage Export Modal</div>,
}));

vi.mock("@/components/EntityUsageExport", () => ({
  UsageExportHeader: ({
    filterLabel,
    filterSlot,
    extraSlot,
    showFilters,
  }: {
    filterLabel?: string;
    filterSlot?: ReactNode;
    extraSlot?: ReactNode;
    showFilters?: boolean;
  }) => (
    <div>
      <span>Usage Export Header</span>
      <span>{filterLabel}</span>
      <span>{`show-filters:${showFilters === true}`}</span>
      {filterSlot}
      {extraSlot}
    </div>
  ),
}));

vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: vi.fn(),
  useUserLookup: vi.fn(() => ({ data: null })),
}));

vi.mock("@/components/common_components/team_multi_select", () => ({
  default: ({ onChange }: { onChange: (value: string[]) => void }) => (
    <button onClick={() => onChange(["team-1"])}>Team Multi Select</button>
  ),
}));

vi.mock("@/components/shared/PaginatedMultiSelect", () => ({
  PaginatedMultiSelect: ({ onValueChange }: { onValueChange: (value: string[]) => void }) => (
    <button onClick={() => onValueChange(["shared"])}>Tag Multi Select</button>
  ),
}));

// Mock useTeams hook
vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(() => ({
    teams: [],
    setTeams: vi.fn(),
  })),
}));

describe("EntityUsage", () => {
  const mockDailyActivityAggregatedCall = vi.mocked(networking.dailyActivityAggregatedCall);
  const mockDailyActivityKeyPageCall = vi.mocked(networking.dailyActivityKeyPageCall);
  const mockTagDailyActivityCall = vi.fn();
  const mockTeamDailyActivityCall = vi.fn();
  const mockOrganizationDailyActivityCall = vi.fn();
  const mockCustomerDailyActivityCall = vi.fn();
  const mockAgentDailyActivityCall = vi.fn();
  const mockUserDailyActivityCall = vi.fn();
  const entityMocks: Record<string, ReturnType<typeof vi.fn>> = {
    tag: mockTagDailyActivityCall,
    team: mockTeamDailyActivityCall,
    organization: mockOrganizationDailyActivityCall,
    customer: mockCustomerDailyActivityCall,
    agent: mockAgentDailyActivityCall,
    user: mockUserDailyActivityCall,
  };
  const mockUseInfiniteUsers = vi.mocked(useInfiniteUsers);
  const mockTagListCall = vi.mocked(networking.tagListCall);

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
    mockDailyActivityAggregatedCall.mockReset();
    mockDailyActivityKeyPageCall.mockReset();
    const emptyKeyPage = {
      api_keys: [],
      total_api_keys: 0,
      offset: 0,
      limit: 50,
    };
    mockDailyActivityKeyPageCall.mockResolvedValue(emptyKeyPage);
    mockDailyActivityAggregatedCall.mockImplementation((entity, request) =>
      (
        entityMocks[entity] as unknown as (
          req: typeof request,
        ) => ReturnType<typeof networking.dailyActivityAggregatedCall>
      )(request),
    );
    Object.values(entityMocks).forEach((mock) => mock.mockClear());
    mockTagDailyActivityCall.mockResolvedValue(mockSpendData);
    mockTeamDailyActivityCall.mockResolvedValue(mockSpendData);
    mockOrganizationDailyActivityCall.mockResolvedValue(mockSpendData);
    mockCustomerDailyActivityCall.mockResolvedValue(mockSpendData);
    mockAgentDailyActivityCall.mockResolvedValue(mockAgentSpendData);
    mockUserDailyActivityCall.mockResolvedValue(mockSpendData);
    mockTagListCall.mockReset();
    mockTagListCall.mockResolvedValue({
      "scoped-tag": { name: "scoped-tag", models: [], created_at: "2025-01-01", updated_at: "2025-01-01" },
    });
    mockUseInfiniteUsers.mockClear();
    mockUseInfiniteUsers.mockReturnValue(
      infiniteUsersResult([
        { user_id: "user-001", user_alias: "Alice", user_email: "alice@example.com" },
        { user_id: "user-002", user_alias: null, user_email: "bob@example.com" },
      ]),
    );
  });

  describe("top key aggregations", () => {
    it("sums, sorts, limits, and carries email attribution for global top keys", () => {
      const results = [
        createDailyData("2025-01-01", {
          "key-low": createKeyMetrics(10, { key_alias: "Low", team_id: null, user_email: "low@example.com" }),
          "key-high": createKeyMetrics(25, { key_alias: "High", team_id: null, user_email: "high@example.com" }),
        }),
        createDailyData("2025-01-02", {
          "key-low": createKeyMetrics(30, { key_alias: "Low", team_id: null, user_email: "low@example.com" }),
        }),
      ];

      expect(getGlobalTopKeys(results, 1)).toEqual([
        {
          api_key: "key-low",
          key_alias: "Low",
          user: "low@example.com",
          tags: [],
          spend: 40,
        },
      ]);
    });

    it("falls back to user ID attribution for global and entity top keys", () => {
      const results = [
        createDailyData("2025-01-01", {
          "key-123": createKeyMetrics(12.5, { key_alias: "User ID key", team_id: null, user_id: "user-123" }),
        }),
      ];

      expect(getGlobalTopKeys(results, 5)[0]?.user).toBe("user-123");
      expect(getTopAPIKeys(results, 5)[0]?.user).toBe("user-123");
    });

    it("carries whether each key still exists for global and entity top keys", () => {
      const results = [
        createDailyData("2025-01-01", {
          "stored-key": createKeyMetrics(20, { key_alias: "Stored", team_id: null, key_exists: true }),
          "session-key": createKeyMetrics(10, { key_alias: null, team_id: null, key_exists: false }),
        }),
      ];
      const existsByKey = (rows: { api_key: string; key_exists?: boolean | null }[]) =>
        Object.fromEntries(rows.map((row) => [row.api_key, row.key_exists]));

      expect(existsByKey(getGlobalTopKeys(results, 5))).toEqual({ "stored-key": true, "session-key": false });
      expect(existsByKey(getTopAPIKeys(results, 5))).toEqual({ "stored-key": true, "session-key": false });
    });
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

    // Scoped to the active Cost tab: the keep-mounted Key Activity tab shows the same totals.
    expect(within(screen.getByRole("tabpanel")).getByText("1,000")).toBeInTheDocument(); // Total Requests
  });

  it("should render with team entity type and call team API", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
    });

    // Check that it shows team-specific label
    expect(screen.getByText("Team Spend Overview")).toBeInTheDocument();

    await waitFor(() => {
      const spendElements = screen.getAllByText("$100.50");
      expect(spendElements.length).toBeGreaterThan(0);
    });
  });

  it("feeds the per-user spend card every visible team except the dashboard team, only for teams", async () => {
    const mockUseTeams = vi.mocked(useTeams);
    const teamsResult = (teams: { team_id: string }[]) =>
      ({ teams, setTeams: vi.fn() }) as unknown as ReturnType<typeof useTeams>;
    mockUseTeams.mockReturnValue(
      teamsResult([{ team_id: "team-alpha" }, { team_id: "litellm-dashboard" }, { team_id: "team-beta" }]),
    );

    render(<EntityUsage {...defaultProps} entityType="team" />);
    expect(await screen.findByText("team-user-spend:team-alpha|team-beta")).toBeInTheDocument();

    cleanup();
    mockUseTeams.mockReturnValue(teamsResult([]));
    render(<EntityUsage {...defaultProps} entityType="tag" />);
    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });
    expect(screen.queryByText(/^team-user-spend:/)).not.toBeInTheDocument();
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

    expect(within(screen.getByRole("tabpanel")).getByRole("heading", { name: "Overall Usage" })).toBeInTheDocument();
  });

  it("loads key pages separately from the aggregate using the current entity scope", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalledTimes(1);
    });
    fireEvent.click(screen.getByText("Team Multi Select"));
    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalledTimes(2);
    });
    fireEvent.click(screen.getByText("Key Activity"));
    await waitFor(() => {
      expect(mockDailyActivityKeyPageCall).toHaveBeenCalledWith(
        "team",
        expect.objectContaining({ entityIds: ["team-1"] }),
        0,
        50,
      );
    });
    expect(mockDailyActivityAggregatedCall.mock.lastCall?.[1]).not.toHaveProperty("apiKeyLimit");

    const pageCallsBeforeFilterChange = mockDailyActivityKeyPageCall.mock.calls.length;
    fireEvent.click(screen.getByText("Team Multi Select"));

    await waitFor(() => {
      expect(mockDailyActivityKeyPageCall.mock.calls.length).toBeGreaterThan(pageCallsBeforeFilterChange);
    });
    expect(mockDailyActivityKeyPageCall.mock.lastCall?.[1]).toEqual(expect.objectContaining({ entityIds: ["team-1"] }));
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
    ["Key Activity", "Overall Usage"],
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
    ["Key Activity", "Overall Usage"],
    ["Endpoint Activity", "Endpoint Usage Panel"],
  ];

  it.each(TEAM_PANELS)("shows only the %s panel for the team entity type", async (tabLabel, marker) => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
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
    const costTab = screen.getByRole("tabpanel");
    expect(await within(costTab).findByText("$0.00")).toBeInTheDocument();
    expect(within(costTab).getByText("Total Spend")).toBeInTheDocument();
    expect(within(costTab).getAllByText("0")[0]).toBeInTheDocument();
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

  it("should still request the filter when the caller's tag scope is empty", async () => {
    render(<EntityUsage {...defaultProps} entityList={[]} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("show-filters:true")).toBeInTheDocument();
  });

  it("should not request the filter while the entity list is still unresolved", async () => {
    render(<EntityUsage {...defaultProps} entityList={null} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    expect(screen.getByText("show-filters:false")).toBeInTheDocument();
  });

  it("should display Agent Activity tab for team entity type", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
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
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
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
      expect(mockAgentDailyActivityCall).toHaveBeenCalledWith(expect.objectContaining({ accessToken: "test-token" }));
    });
  });

  it("offers per-model top keys in Model Activity but not in the agent breakdown", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
    });

    act(() => {
      fireEvent.click(screen.getByText("Model Activity"));
    });
    expect(showingCount("top-keys-fetcher:model_groups")).toBeGreaterThan(0);

    act(() => {
      fireEvent.click(screen.getByText("Agent Activity"));
    });
    expect(showingCount("metrics-source:entities")).toBeGreaterThan(0);
    expect(screen.queryByText("top-keys-fetcher:entities")).not.toBeInTheDocument();
  });

  it("shows a loader instead of zero totals while the aggregate is in flight", async () => {
    let resolveSpend: (value: typeof mockSpendData) => void = () => {};
    mockTagDailyActivityCall.mockReturnValue(
      new Promise<typeof mockSpendData>((resolve) => {
        resolveSpend = resolve;
      }),
    );
    render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });
    expect(screen.getAllByText("Loading chart data...")).toHaveLength(2);
    expect(screen.queryByText("Total Spend")).not.toBeInTheDocument();
    expect(screen.queryByText("Overall Usage")).not.toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();

    await act(async () => {
      resolveSpend(mockSpendData);
    });

    expect(screen.queryByText("Loading chart data...")).not.toBeInTheDocument();
    expect(screen.getByText("Overall Usage")).toBeInTheDocument();
    expect(screen.getByText("Total Spend")).toBeInTheDocument();
    expect(screen.getAllByText("$100.50").length).toBeGreaterThan(0);
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
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
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

  it("renders the stacked daily spend chart, the per-entity table, and the provider share bar with a $ total", async () => {
    const { container } = render(<EntityUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDailyActivityCall).toHaveBeenCalled();
    });

    // The fixture carries no model breakdown, so the day's spend stacks as a single "Other" segment.
    await waitFor(() => {
      expect(container.querySelectorAll("path.recharts-rectangle")).toHaveLength(1);
    });
    expect(container.querySelector("path.recharts-rectangle")).toHaveAttribute("fill", "#94a3b8");

    expect(screen.getAllByText("Jan 1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Tag 1").length).toBeGreaterThan(0);

    const segments = screen.getAllByTestId("provider-share-segment");
    expect(segments).toHaveLength(1);
    expect(segments[0]).toHaveStyle({ backgroundColor: "rgb(236, 72, 153)" });
    expect(screen.getByTestId("provider-spend-total")).toHaveTextContent("$100.50");
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
        expect(mockTeamDailyActivityCall).toHaveBeenCalled();
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
    mockTeamDailyActivityCall.mockResolvedValue({
      ...mockSpendData,
      results: [
        {
          ...mockSpendData.results[0],
          breakdown: {
            ...mockSpendData.results[0].breakdown,
            model_groups: { "gpt-4o": { metrics: { ...usageMetrics, spend: 70.25 }, metadata: {} } },
            api_keys: {
              "sk-abc": {
                metrics: usageMetrics,
                metadata: { key_alias: "prod-key", team_id: null, user_email: "alice@example.com" },
              },
            },
          },
        },
      ],
    });

    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(screen.getByText("top-keys:sk-abc=30.75=alice@example.com")).toBeInTheDocument();
    });
    expect(screen.getByText("top-models:gpt-4o=70.25")).toBeInTheDocument();
    expect(screen.getByText(/^top-models:Code Review Agent=/)).toBeInTheDocument();
  });

  it("uses the aggregated team endpoint and makes a single bounded request", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockDailyActivityAggregatedCall).toHaveBeenCalledWith("team", expect.anything());
    });
    expect(mockDailyActivityAggregatedCall.mock.calls.filter((c) => c[0] === "team")).toHaveLength(1);

    await waitFor(() => {
      expect(screen.getAllByText("$100.50").length).toBeGreaterThan(0);
    });
  });

  it("does not scope the agent breakdown by the selected team ids", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockTeamDailyActivityCall).toHaveBeenCalled();
      expect(mockAgentDailyActivityCall).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Team Multi Select" }));

    await waitFor(() => {
      const teamRequests = mockDailyActivityAggregatedCall.mock.calls.filter((call) => call[0] === "team");
      expect(teamRequests.some((call) => call[1].entityIds?.includes("team-1"))).toBe(true);
    });

    const agentRequests = mockDailyActivityAggregatedCall.mock.calls.filter((call) => call[0] === "agent");
    expect(agentRequests.length).toBeGreaterThan(0);
    agentRequests.forEach((call) => {
      expect(call[1].entityIds).toBeNull();
    });
  });

  it("does not refetch agent activity when the team selection changes", async () => {
    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(mockAgentDailyActivityCall).toHaveBeenCalled();
    });
    const agentCallsBefore = mockDailyActivityAggregatedCall.mock.calls.filter((call) => call[0] === "agent").length;
    const teamCallsBefore = mockDailyActivityAggregatedCall.mock.calls.filter((call) => call[0] === "team").length;

    fireEvent.click(screen.getByRole("button", { name: "Team Multi Select" }));

    await waitFor(() => {
      expect(mockDailyActivityAggregatedCall.mock.calls.filter((call) => call[0] === "team").length).toBeGreaterThan(
        teamCallsBefore,
      );
    });
    expect(mockDailyActivityAggregatedCall.mock.calls.filter((call) => call[0] === "agent")).toHaveLength(
      agentCallsBefore,
    );
  });

  it("surfaces a failure alert when the aggregated call fails instead of retrying other routes", async () => {
    mockTeamDailyActivityCall.mockRejectedValue(new Error("aggregated unavailable"));

    render(<EntityUsage {...defaultProps} entityType="team" />);

    await waitFor(() => {
      expect(screen.getAllByText(/Fetching spend data failed/).length).toBeGreaterThan(0);
    });
    expect(mockDailyActivityAggregatedCall.mock.calls.filter((c) => c[0] === "team")).toHaveLength(1);
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

      expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
        expect.objectContaining({ accessToken: "test-token", entityIds: null }),
      );

      await user.click(userCombobox());
      await user.click(await screen.findByText("Alice (user-001)"));

      await waitFor(() => {
        expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
          expect.objectContaining({ accessToken: "test-token", entityIds: ["user-001"] }),
        );
      });

      mockUserDailyActivityCall.mockClear();
      await user.click(userDropdown().querySelector('[data-slot="combobox-clear"]') as HTMLElement);

      await waitFor(() => {
        expect(mockUserDailyActivityCall).toHaveBeenCalledWith(
          expect.objectContaining({ accessToken: "test-token", entityIds: null }),
        );
      });
    });
  });

  describe("tag team filter and breakdown", () => {
    const TEAM_NOTE =
      "Requests with several tags count once per tag, so team totals here can be higher than actual team spend. See the Team tab for exact team totals.";

    it("sends selected teams as teamIds to the aggregated call and scopes the tag options", async () => {
      render(<EntityUsage {...defaultProps} />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalledWith(
          expect.objectContaining({ accessToken: "test-token", teamIds: null }),
        );
      });

      fireEvent.click(screen.getByText("Team Multi Select"));

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalledWith(
          expect.objectContaining({ teamIds: ["team-1"] }),
        );
      });
      await waitFor(() => {
        expect(mockTagListCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          expect.objectContaining({ teamIds: ["team-1"], usageOnly: true }),
        );
      });
    });

    it("does not call tagListCall while no team is selected", async () => {
      render(<EntityUsage {...defaultProps} />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalled();
      });
      expect(mockTagListCall).not.toHaveBeenCalled();
    });

    it("sends group_by=team, shows the note, and labels rows by team alias", async () => {
      const mockUseTeams = vi.mocked(useTeams);
      mockUseTeams.mockReturnValue({
        teams: [{ team_id: "tag-1", team_alias: "Team Alpha" }],
        setTeams: vi.fn(),
      } as unknown as ReturnType<typeof useTeams>);

      render(<EntityUsage {...defaultProps} />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalled();
      });
      expect(screen.queryByText(TEAM_NOTE)).not.toBeInTheDocument();
      expect(mockTagDailyActivityCall.mock.lastCall?.[0].groupBy).toBeNull();

      fireEvent.click(screen.getByRole("radio", { name: "Team" }));

      await waitFor(() => {
        expect(mockTagDailyActivityCall.mock.lastCall?.[0]).toEqual(
          expect.objectContaining({ groupBy: "team" }),
        );
      });
      expect(await screen.findByText(TEAM_NOTE)).toBeInTheDocument();
      expect(screen.getAllByText("Team Alpha").length).toBeGreaterThan(0);
    });

    it("keeps tag mode unscoped to teams and without the note", async () => {
      render(<EntityUsage {...defaultProps} />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalled();
      });
      expect(mockTagDailyActivityCall.mock.lastCall?.[0]).toEqual(
        expect.objectContaining({ teamIds: null, groupBy: null }),
      );
      expect(screen.queryByText(TEAM_NOTE)).not.toBeInTheDocument();
    });

    it("does not render the tag tab's team filter on the team tab", async () => {
      render(<EntityUsage {...defaultProps} entityType="team" />);

      await waitFor(() => {
        expect(mockTeamDailyActivityCall).toHaveBeenCalled();
      });
      // "Filter by tag" belongs to the Team tab; "Filter by team" is the entity filter, not the tag tab's slot.
      expect(screen.getByText("Filter by tag")).toBeInTheDocument();
      expect(mockTeamDailyActivityCall.mock.lastCall?.[0]).toEqual(
        expect.objectContaining({ tags: null, groupBy: null }),
      );
    });

    it("labels the spend section and breakdown column as Team in team grouping", async () => {
      render(<EntityUsage {...defaultProps} />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByRole("radio", { name: "Team" }));

      expect(await screen.findByText("Spend Per Team")).toBeInTheDocument();
      expect(screen.getByRole("columnheader", { name: "Team" })).toBeInTheDocument();
    });

    it("shows the Team header and aliases in the top keys table in team grouping", async () => {
      const mockUseTeams = vi.mocked(useTeams);
      mockUseTeams.mockReturnValue({
        teams: [{ team_id: "tag-1", team_alias: "Team Alpha" }],
        setTeams: vi.fn(),
      } as unknown as ReturnType<typeof useTeams>);
      const entityMetrics = {
        spend: 5,
        api_requests: 5,
        successful_requests: 5,
        failed_requests: 0,
        total_tokens: 500,
        prompt_tokens: 300,
        completion_tokens: 200,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      };
      mockTagDailyActivityCall.mockResolvedValue({
        results: [
          {
            date: "2025-01-01",
            metrics: { ...mockSpendData.results[0].metrics },
            breakdown: {
              entities: {
                "tag-1": {
                  metrics: { ...entityMetrics },
                  metadata: { team_alias: "Tag 1" },
                  api_key_breakdown: {
                    "key-abc": { metrics: { ...entityMetrics }, metadata: {} },
                  },
                },
              },
              models: {},
              api_keys: {
                "key-abc": { metrics: { ...entityMetrics }, metadata: {} },
              },
              providers: {},
            },
          },
        ],
        metadata: mockSpendData.metadata,
      });

      render(<EntityUsage {...defaultProps} />);

      await waitFor(() => {
        expect(mockTagDailyActivityCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByRole("radio", { name: "Team" }));

      expect(await screen.findByText("top-keys-header:Team")).toBeInTheDocument();
      await waitFor(() => {
        expect(screen.getAllByText("Team Alpha").length).toBeGreaterThan(0);
      });
    });

    it("keeps the Tag labels and shows the Break down by text in tag grouping", async () => {
      render(<EntityUsage {...defaultProps} />);

      expect(await screen.findByText("Spend Per Tag")).toBeInTheDocument();
      expect(screen.getByRole("columnheader", { name: "Tag" })).toBeInTheDocument();
      expect(screen.getByText("top-keys-header:Tags")).toBeInTheDocument();
      expect(screen.getByText("Break down by")).toBeInTheDocument();
    });
  });

  describe("team tab tag filter and breakdown", () => {
    const TEAM_TAG_NOTE =
      "These totals only count tagged requests, and a request with several tags counts once per tag, so they can differ from actual team spend. Clear the tag filter and break down by Team for exact team totals.";

    const teamProps = { ...defaultProps, entityType: "team" as const };

    it("sends no tags or group_by in the default request", async () => {
      render(<EntityUsage {...teamProps} />);

      await waitFor(() => {
        expect(mockTeamDailyActivityCall).toHaveBeenCalled();
      });
      expect(mockTeamDailyActivityCall.mock.lastCall?.[0]).toEqual(
        expect.objectContaining({ tags: null, groupBy: null }),
      );
      expect(screen.queryByText(TEAM_TAG_NOTE)).not.toBeInTheDocument();
    });

    it("sends tags when tags are picked and shows the note", async () => {
      render(<EntityUsage {...teamProps} />);

      await waitFor(() => {
        expect(mockTeamDailyActivityCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByText("Tag Multi Select"));

      await waitFor(() => {
        expect(mockTeamDailyActivityCall.mock.lastCall?.[0]).toEqual(
          expect.objectContaining({ tags: ["shared"], groupBy: null }),
        );
      });
      expect(await screen.findByText(TEAM_TAG_NOTE)).toBeInTheDocument();
    });

    it("scopes tag options to selected teams and drops the teamIds when none are selected", async () => {
      render(<EntityUsage {...teamProps} />);

      await waitFor(() => {
        expect(mockTagListCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          expect.objectContaining({ usageOnly: true }),
        );
      });
      expect(mockTagListCall.mock.lastCall?.[3]?.teamIds).toBeUndefined();

      fireEvent.click(screen.getByText("Team Multi Select"));

      await waitFor(() => {
        expect(mockTagListCall).toHaveBeenCalledWith(
          "test-token",
          expect.any(Date),
          expect.any(Date),
          expect.objectContaining({ teamIds: ["team-1"], usageOnly: true }),
        );
      });
    });

    it("sends group_by=tag, renders tag-name rows under Spend Per Tag, and shows the note", async () => {
      render(<EntityUsage {...teamProps} />);

      await waitFor(() => {
        expect(mockTeamDailyActivityCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByRole("radio", { name: "Tag" }));

      await waitFor(() => {
        expect(mockTeamDailyActivityCall.mock.lastCall?.[0]).toEqual(expect.objectContaining({ groupBy: "tag" }));
      });
      expect(await screen.findByText("Spend Per Tag")).toBeInTheDocument();
      expect(await screen.findByText(TEAM_TAG_NOTE)).toBeInTheDocument();
      // mockSpendData's entity key is the tag-like id "tag-1"; it must still render as a row label
      // while the team filter selection is empty, proving filterDataByTags is skipped.
      expect(screen.queryAllByText(/tag-1|Tag 1/).length).toBeGreaterThan(0);
    });

    it("keeps tag-name rows visible in tag breakdown even with teams selected", async () => {
      render(<EntityUsage {...teamProps} />);

      await waitFor(() => {
        expect(mockTeamDailyActivityCall).toHaveBeenCalled();
      });
      fireEvent.click(screen.getByText("Team Multi Select"));
      fireEvent.click(screen.getByRole("radio", { name: "Tag" }));

      await waitFor(() => {
        expect(mockTeamDailyActivityCall.mock.lastCall?.[0]).toEqual(
          expect.objectContaining({ entityIds: ["team-1"], groupBy: "tag" }),
        );
      });
      // Without the filterDataByTags skip, "tag-1" would be filtered against ["team-1"] and vanish.
      await waitFor(() => {
        expect(screen.queryAllByText(/tag-1|Tag 1/).length).toBeGreaterThan(0);
      });
      expect(screen.getByRole("columnheader", { name: "Tag" })).toBeInTheDocument();
    });
  });
});
