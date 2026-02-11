import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { ActivityMetrics, formatKeyLabel, processActivityData } from "./activity_metrics";
import { Team } from "./key_team_helpers/key_list";
import { DailyData, KeyMetricWithMetadata, ModelActivityData } from "./UsagePage/types";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("@tremor/react", () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Grid: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Text: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  Title: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  AreaChart: () => <div>AreaChart</div>,
  BarChart: () => <div>BarChart</div>,
}));

vi.mock("antd", () => {
  const CollapseComponent = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  const PanelComponent = ({ children, header }: { children: React.ReactNode; header: React.ReactNode }) => (
    <div>
      <div>{header}</div>
      <div>{children}</div>
    </div>
  );
  PanelComponent.displayName = "Collapse.Panel";
  CollapseComponent.Panel = PanelComponent;
  const TableComponent = ({ dataSource, columns }: { dataSource?: unknown[]; columns?: { title: string }[] }) => (
    <table>
      <thead>
        <tr>
          {columns?.map((col, i) => (
            <th key={i}>{col.title}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {dataSource?.map((_, i) => (
          <tr key={i} />
        ))}
      </tbody>
    </table>
  );
  return {
    Collapse: CollapseComponent,
    Table: TableComponent,
  };
});

vi.mock("./UsagePage/utils/value_formatters", () => ({
  valueFormatter: (value: number) => value.toString(),
}));

vi.mock("./common_components/chartUtils", () => ({
  CustomTooltip: () => null,
  CustomLegend: () => <div>Legend</div>,
}));

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: (value: number, decimals?: number) => {
    return value.toFixed(decimals || 0);
  },
}));

vi.mock("@/utils/teamUtils", () => ({
  resolveTeamAliasFromTeamID: (teamID: string, teams: any[]) => {
    const team = teams.find((team) => team.team_id === teamID);
    return team ? team.team_alias : null;
  },
}));

const EMPTY_BREAKDOWN = {
  models: {},
  model_groups: {},
  mcp_servers: {},
  providers: {},
  api_keys: {},
  entities: {},
};

const EMPTY_SPEND_METRICS = {
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

const MOCK_TEAMS: Team[] = [
  {
    team_id: "team1",
    team_alias: "Test Team 1",
    models: [],
    max_budget: null,
    budget_duration: null,
    tpm_limit: null,
    rpm_limit: null,
    organization_id: "org1",
    created_at: "2025-01-01",
    keys: [],
    members_with_roles: [],
    spend: 0,
  },
  {
    team_id: "team2",
    team_alias: "Test Team 2",
    models: [],
    max_budget: null,
    budget_duration: null,
    tpm_limit: null,
    rpm_limit: null,
    organization_id: "org2",
    created_at: "2025-01-01",
    keys: [],
    members_with_roles: [],
    spend: 0,
  },
];

const createMockDailyData = (
  date: string,
  metrics: typeof EMPTY_SPEND_METRICS,
  breakdown: typeof EMPTY_BREAKDOWN,
): DailyData => ({
  date,
  metrics,
  breakdown,
});

const createMockKeyMetricWithMetadata = (
  metadata: { key_alias: string | null; team_id: string | null },
  metrics: typeof EMPTY_SPEND_METRICS = EMPTY_SPEND_METRICS,
): KeyMetricWithMetadata => ({
  metrics,
  metadata,
});

const createMockModelActivityData = (label: string, overrides: Partial<ModelActivityData> = {}): ModelActivityData => ({
  label,
  total_requests: 100,
  total_successful_requests: 95,
  total_failed_requests: 5,
  total_tokens: 50000,
  prompt_tokens: 30000,
  completion_tokens: 20000,
  total_spend: 100.5,
  total_cache_read_input_tokens: 1000,
  total_cache_creation_input_tokens: 500,
  top_api_keys: [],
  top_models: [],
  daily_data: [
    {
      date: "2025-01-01",
      metrics: {
        prompt_tokens: 30000,
        completion_tokens: 20000,
        total_tokens: 50000,
        api_requests: 100,
        spend: 100.5,
        successful_requests: 95,
        failed_requests: 5,
        cache_read_input_tokens: 1000,
        cache_creation_input_tokens: 500,
      },
    },
  ],
  ...overrides,
});

const GPT_35_MODEL_DATA: ModelActivityData = {
  label: "GPT-3.5",
  total_requests: 50,
  total_successful_requests: 48,
  total_failed_requests: 2,
  total_tokens: 25000,
  prompt_tokens: 15000,
  completion_tokens: 10000,
  total_spend: 25.25,
  total_cache_read_input_tokens: 500,
  total_cache_creation_input_tokens: 250,
  top_api_keys: [],
  top_models: [],
  daily_data: [
    {
      date: "2025-01-01",
      metrics: {
        prompt_tokens: 15000,
        completion_tokens: 10000,
        total_tokens: 25000,
        api_requests: 50,
        spend: 25.25,
        successful_requests: 48,
        failed_requests: 2,
        cache_read_input_tokens: 500,
        cache_creation_input_tokens: 250,
      },
    },
  ],
};

describe("ActivityMetrics", () => {
  const mockModelMetrics: Record<string, ModelActivityData> = {
    "gpt-4": createMockModelActivityData("GPT-4"),
  };

  it("should render", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Overall Usage")).toBeInTheDocument();
  });

  it("should display prompt caching metrics when hidePromptCachingMetrics is false", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={false} />);
    expect(screen.getByText("Prompt Caching Metrics")).toBeInTheDocument();
  });

  it("should hide prompt caching metrics when hidePromptCachingMetrics is true", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={true} />);
    expect(screen.queryByText("Prompt Caching Metrics")).not.toBeInTheDocument();
  });

  it("should display overall usage summary metrics", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    const totalRequestsElements = screen.getAllByText("Total Requests");
    expect(totalRequestsElements.length).toBeGreaterThan(0);
    const totalSuccessfulElements = screen.getAllByText("Total Successful Requests");
    expect(totalSuccessfulElements.length).toBeGreaterThan(0);
    const totalTokensElements = screen.getAllByText("Total Tokens");
    expect(totalTokensElements.length).toBeGreaterThan(0);
    const totalSpendElements = screen.getAllByText("Total Spend");
    expect(totalSpendElements.length).toBeGreaterThan(0);
  });

  it("should display aggregated totals across all models", () => {
    const multipleModels: Record<string, ModelActivityData> = {
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        total_requests: 100,
        total_spend: 100.5,
      },
      "gpt-3.5": GPT_35_MODEL_DATA,
    };

    render(<ActivityMetrics modelMetrics={multipleModels} />);
    const totalRequestsElements = screen.getAllByText("150");
    expect(totalRequestsElements.length).toBeGreaterThan(0);
    const totalSuccessfulElements = screen.getAllByText("143");
    expect(totalSuccessfulElements.length).toBeGreaterThan(0);
  });

  it("should display model sections sorted by spend", () => {
    const multipleModels: Record<string, ModelActivityData> = {
      "gpt-3.5": GPT_35_MODEL_DATA,
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        total_spend: 100.5,
      },
    };

    render(<ActivityMetrics modelMetrics={multipleModels} />);
    const headers = screen.getAllByRole("heading", { level: 2 });
    const gpt4Index = headers.findIndex((h) => h.textContent?.includes("GPT-4"));
    const gpt35Index = headers.findIndex((h) => h.textContent?.includes("GPT-3.5"));
    expect(gpt4Index).toBeLessThan(gpt35Index);
  });

  it("should display model summary cards with correct values", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    const requestElements = screen.getAllByText("100");
    expect(requestElements.length).toBeGreaterThan(0);
    const successfulElements = screen.getAllByText("95");
    expect(successfulElements.length).toBeGreaterThan(0);
    const tokenElements = screen.getAllByText("50,000");
    expect(tokenElements.length).toBeGreaterThan(0);
  });

  it("should not display Top Virtual Keys section when model has no top_api_keys", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.queryByText("Top Virtual Keys by Spend")).not.toBeInTheDocument();
  });

  it("should display top API keys section when present", () => {
    const modelWithTopKeys: Record<string, ModelActivityData> = {
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        top_api_keys: [
          {
            api_key: "key-123",
            key_alias: "Test Key",
            team_id: "team1",
            spend: 50.25,
            requests: 25,
            tokens: 12500,
          },
        ],
      },
    };

    render(<ActivityMetrics modelMetrics={modelWithTopKeys} />);
    expect(screen.getByText("Top Virtual Keys by Spend")).toBeInTheDocument();
    expect(screen.getByText("Test Key")).toBeInTheDocument();
  });

  it("should display API key hash when alias is missing", () => {
    const modelWithTopKeys: Record<string, ModelActivityData> = {
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        top_api_keys: [
          {
            api_key: "key-1234567890",
            key_alias: null,
            team_id: null,
            spend: 50.25,
            requests: 25,
            tokens: 12500,
          },
        ],
      },
    };

    render(<ActivityMetrics modelMetrics={modelWithTopKeys} />);
    expect(screen.getByText(/key-123456/)).toBeInTheDocument();
  });

  it("should display team information for top API keys", () => {
    const modelWithTopKeys: Record<string, ModelActivityData> = {
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        top_api_keys: [
          {
            api_key: "key-123",
            key_alias: "Test Key",
            team_id: "team1",
            spend: 50.25,
            requests: 25,
            tokens: 12500,
          },
        ],
      },
    };

    render(<ActivityMetrics modelMetrics={modelWithTopKeys} />);
    expect(screen.getByText(/Team: team1/)).toBeInTheDocument();
  });

  it("should display Model Usage when model has top_models", () => {
    const modelWithTopModels: Record<string, ModelActivityData> = {
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        top_models: [
          {
            model: "gpt-4",
            spend: 100.5,
            requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            tokens: 50000,
          },
        ],
      },
    };

    render(<ActivityMetrics modelMetrics={modelWithTopModels} />);
    expect(screen.getByRole("heading", { name: "Model Usage" })).toBeInTheDocument();
  });

  it("should display Spend per day in model section", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Spend per day")).toBeInTheDocument();
  });

  it("should display Requests per day in model section", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Requests per day")).toBeInTheDocument();
  });

  it("should display Success vs Failed Requests in model section", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Success vs Failed Requests")).toBeInTheDocument();
  });

  it("should sort empty string model key last in collapse order", () => {
    const modelsWithEmptyKey: Record<string, ModelActivityData> = {
      "gpt-4": { ...mockModelMetrics["gpt-4"] },
      "": {
        ...createMockModelActivityData(""),
        label: "Unknown",
      },
    };

    render(<ActivityMetrics modelMetrics={modelsWithEmptyKey} />);
    const headings = screen.getAllByRole("heading", { level: 2 });
    const gpt4Index = headings.findIndex((h) => h.textContent?.includes("GPT-4"));
    const unknownIndex = headings.findIndex((h) => h.textContent?.includes("Unknown"));
    expect(gpt4Index).toBeLessThan(unknownIndex);
  });

  it("should display average tokens per successful request", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    const avgTokensElements = screen.getAllByText(/avg per successful request/);
    expect(avgTokensElements.length).toBeGreaterThan(0);
    expect(avgTokensElements.some((el) => el.textContent?.includes("526"))).toBe(true);
  });

  it("should display average spend per successful request", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    const avgSpendElements = screen.getAllByText(/per successful request/);
    expect(avgSpendElements.length).toBeGreaterThan(0);
    expect(avgSpendElements.some((el) => el.textContent?.includes("1.058"))).toBe(true);
  });

  it("should handle zero successful requests without division error", () => {
    const modelWithZeroRequests: Record<string, ModelActivityData> = {
      "gpt-4": {
        ...mockModelMetrics["gpt-4"],
        total_successful_requests: 0,
        total_tokens: 0,
        total_spend: 0,
      },
    };

    render(<ActivityMetrics modelMetrics={modelWithZeroRequests} />);
    const zeroElements = screen.getAllByText("0");
    expect(zeroElements.length).toBeGreaterThan(0);
  });

  it("should display prompt caching token counts when visible", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} hidePromptCachingMetrics={false} />);
    expect(screen.getByText(/Cache Read:.*tokens/)).toBeInTheDocument();
    expect(screen.getByText(/Cache Creation:.*tokens/)).toBeInTheDocument();
  });

  it("should display charts for tokens over time", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Total Tokens Over Time")).toBeInTheDocument();
    expect(screen.getAllByText("AreaChart").length).toBeGreaterThan(0);
  });

  it("should display charts for requests over time", () => {
    render(<ActivityMetrics modelMetrics={mockModelMetrics} />);
    expect(screen.getByText("Total Requests Over Time")).toBeInTheDocument();
  });

  it("should handle empty model metrics", () => {
    render(<ActivityMetrics modelMetrics={{}} />);
    expect(screen.getByText("Overall Usage")).toBeInTheDocument();
  });

  it("should display model label or fallback to Unknown Item", () => {
    const modelWithEmptyLabel: Record<string, ModelActivityData> = {
      "": {
        ...mockModelMetrics["gpt-4"],
        label: "",
      },
    };

    render(<ActivityMetrics modelMetrics={modelWithEmptyLabel} />);
    expect(screen.getByText("Unknown Item")).toBeInTheDocument();
  });
});

describe("processActivityData", () => {
  const mockDailyActivity: { results: DailyData[] } = {
    results: [
      createMockDailyData(
        "2025-01-01",
        {
          spend: 100.5,
          prompt_tokens: 30000,
          completion_tokens: 20000,
          total_tokens: 50000,
          api_requests: 100,
          successful_requests: 95,
          failed_requests: 5,
          cache_read_input_tokens: 1000,
          cache_creation_input_tokens: 500,
        },
        {
          ...EMPTY_BREAKDOWN,
          api_keys: {
            key1: createMockKeyMetricWithMetadata(
              {
                key_alias: "test-key-1",
                team_id: "team1",
              },
              {
                spend: 50.25,
                prompt_tokens: 15000,
                completion_tokens: 10000,
                total_tokens: 25000,
                api_requests: 50,
                successful_requests: 47,
                failed_requests: 3,
                cache_read_input_tokens: 500,
                cache_creation_input_tokens: 250,
              },
            ),
          },
        },
      ),
    ],
  };

  it("should process data for models key without teams parameter", () => {
    const result = processActivityData(mockDailyActivity, "models");

    expect(result).toEqual({});
  });

  it("should process data for api_keys key with teams parameter", () => {
    const result = processActivityData(mockDailyActivity, "api_keys", MOCK_TEAMS);

    expect(result).toHaveProperty("key1");
    expect(result["key1"].label).toBe("test-key-1 (team: Test Team 1)");
    expect(result["key1"].total_requests).toBe(50);
    expect(result["key1"].total_spend).toBe(50.25);
  });

  it("should process data for api_keys key without teams parameter", () => {
    const result = processActivityData(mockDailyActivity, "api_keys");

    expect(result).toHaveProperty("key1");
    expect(result["key1"].label).toBe("test-key-1 (team_id: team1)");
  });

  it("should process data for models key with data", () => {
    const dailyActivityWithModels: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 100.5,
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 500,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 100.5,
                  prompt_tokens: 30000,
                  completion_tokens: 20000,
                  total_tokens: 50000,
                  api_requests: 100,
                  successful_requests: 95,
                  failed_requests: 5,
                  cache_read_input_tokens: 1000,
                  cache_creation_input_tokens: 500,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithModels, "models");

    expect(result).toHaveProperty("gpt-4");
    expect(result["gpt-4"].label).toBe("gpt-4");
    expect(result["gpt-4"].total_requests).toBe(100);
    expect(result["gpt-4"].total_spend).toBe(100.5);
  });

  it("should process data for mcp_servers key", () => {
    const dailyActivityWithMCP: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 50.0,
            prompt_tokens: 15000,
            completion_tokens: 10000,
            total_tokens: 25000,
            api_requests: 50,
            successful_requests: 48,
            failed_requests: 2,
            cache_read_input_tokens: 500,
            cache_creation_input_tokens: 250,
          },
          breakdown: {
            models: {},
            model_groups: {},
            mcp_servers: {
              "server-1": {
                metrics: {
                  spend: 50.0,
                  prompt_tokens: 15000,
                  completion_tokens: 10000,
                  total_tokens: 25000,
                  api_requests: 50,
                  successful_requests: 48,
                  failed_requests: 2,
                  cache_read_input_tokens: 500,
                  cache_creation_input_tokens: 250,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithMCP, "mcp_servers");

    expect(result).toHaveProperty("server-1");
    expect(result["server-1"].label).toBe("server-1");
    expect(result["server-1"].total_requests).toBe(50);
  });

  it("should aggregate metrics across multiple days", () => {
    const multiDayActivity: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 50.0,
            prompt_tokens: 15000,
            completion_tokens: 10000,
            total_tokens: 25000,
            api_requests: 50,
            successful_requests: 48,
            failed_requests: 2,
            cache_read_input_tokens: 500,
            cache_creation_input_tokens: 250,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 50.0,
                  prompt_tokens: 15000,
                  completion_tokens: 10000,
                  total_tokens: 25000,
                  api_requests: 50,
                  successful_requests: 48,
                  failed_requests: 2,
                  cache_read_input_tokens: 500,
                  cache_creation_input_tokens: 250,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
        {
          date: "2025-01-02",
          metrics: {
            spend: 50.5,
            prompt_tokens: 15000,
            completion_tokens: 10000,
            total_tokens: 25000,
            api_requests: 50,
            successful_requests: 47,
            failed_requests: 3,
            cache_read_input_tokens: 500,
            cache_creation_input_tokens: 250,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 50.5,
                  prompt_tokens: 15000,
                  completion_tokens: 10000,
                  total_tokens: 25000,
                  api_requests: 50,
                  successful_requests: 47,
                  failed_requests: 3,
                  cache_read_input_tokens: 500,
                  cache_creation_input_tokens: 250,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(multiDayActivity, "models");

    expect(result["gpt-4"].total_requests).toBe(100);
    expect(result["gpt-4"].total_spend).toBe(100.5);
    expect(result["gpt-4"].total_successful_requests).toBe(95);
    expect(result["gpt-4"].total_failed_requests).toBe(5);
    expect(result["gpt-4"].daily_data).toHaveLength(2);
  });

  it("should sort daily data by date", () => {
    const unsortedDailyActivity: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-03",
          metrics: {
            spend: 50.0,
            prompt_tokens: 15000,
            completion_tokens: 10000,
            total_tokens: 25000,
            api_requests: 50,
            successful_requests: 48,
            failed_requests: 2,
            cache_read_input_tokens: 500,
            cache_creation_input_tokens: 250,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 50.0,
                  prompt_tokens: 15000,
                  completion_tokens: 10000,
                  total_tokens: 25000,
                  api_requests: 50,
                  successful_requests: 48,
                  failed_requests: 2,
                  cache_read_input_tokens: 500,
                  cache_creation_input_tokens: 250,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
        {
          date: "2025-01-01",
          metrics: {
            spend: 50.0,
            prompt_tokens: 15000,
            completion_tokens: 10000,
            total_tokens: 25000,
            api_requests: 50,
            successful_requests: 48,
            failed_requests: 2,
            cache_read_input_tokens: 500,
            cache_creation_input_tokens: 250,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 50.0,
                  prompt_tokens: 15000,
                  completion_tokens: 10000,
                  total_tokens: 25000,
                  api_requests: 50,
                  successful_requests: 48,
                  failed_requests: 2,
                  cache_read_input_tokens: 500,
                  cache_creation_input_tokens: 250,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(unsortedDailyActivity, "models");

    expect(result["gpt-4"].daily_data[0].date).toBe("2025-01-01");
    expect(result["gpt-4"].daily_data[1].date).toBe("2025-01-03");
  });

  it("should process api_key_breakdown for models", () => {
    const dailyActivityWithBreakdown: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 100.5,
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 500,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 100.5,
                  prompt_tokens: 30000,
                  completion_tokens: 20000,
                  total_tokens: 50000,
                  api_requests: 100,
                  successful_requests: 95,
                  failed_requests: 5,
                  cache_read_input_tokens: 1000,
                  cache_creation_input_tokens: 500,
                },
                metadata: {},
                api_key_breakdown: {
                  "key-1": {
                    metrics: {
                      spend: 60.0,
                      prompt_tokens: 18000,
                      completion_tokens: 12000,
                      total_tokens: 30000,
                      api_requests: 60,
                      successful_requests: 57,
                      failed_requests: 3,
                      cache_read_input_tokens: 600,
                      cache_creation_input_tokens: 300,
                    },
                    metadata: {
                      key_alias: "test-key-1",
                      team_id: "team1",
                    },
                  },
                  "key-2": {
                    metrics: {
                      spend: 40.5,
                      prompt_tokens: 12000,
                      completion_tokens: 8000,
                      total_tokens: 20000,
                      api_requests: 40,
                      successful_requests: 38,
                      failed_requests: 2,
                      cache_read_input_tokens: 400,
                      cache_creation_input_tokens: 200,
                    },
                    metadata: {
                      key_alias: "test-key-2",
                      team_id: "team2",
                    },
                  },
                },
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithBreakdown, "models");

    expect(result["gpt-4"].top_api_keys).toHaveLength(2);
    expect(result["gpt-4"].top_api_keys[0].spend).toBe(60.0);
    expect(result["gpt-4"].top_api_keys[0].api_key).toBe("key-1");
    expect(result["gpt-4"].top_api_keys[1].spend).toBe(40.5);
  });

  it("should limit top_api_keys to 5 entries", () => {
    const dailyActivityWithManyKeys: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 100.5,
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 500,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 100.5,
                  prompt_tokens: 30000,
                  completion_tokens: 20000,
                  total_tokens: 50000,
                  api_requests: 100,
                  successful_requests: 95,
                  failed_requests: 5,
                  cache_read_input_tokens: 1000,
                  cache_creation_input_tokens: 500,
                },
                metadata: {},
                api_key_breakdown: {
                  "key-1": {
                    metrics: {
                      spend: 20.0,
                      prompt_tokens: 6000,
                      completion_tokens: 4000,
                      total_tokens: 10000,
                      api_requests: 20,
                      successful_requests: 19,
                      failed_requests: 1,
                      cache_read_input_tokens: 200,
                      cache_creation_input_tokens: 100,
                    },
                    metadata: { key_alias: "key-1", team_id: null },
                  },
                  "key-2": {
                    metrics: {
                      spend: 19.0,
                      prompt_tokens: 5700,
                      completion_tokens: 3800,
                      total_tokens: 9500,
                      api_requests: 19,
                      successful_requests: 18,
                      failed_requests: 1,
                      cache_read_input_tokens: 190,
                      cache_creation_input_tokens: 95,
                    },
                    metadata: { key_alias: "key-2", team_id: null },
                  },
                  "key-3": {
                    metrics: {
                      spend: 18.0,
                      prompt_tokens: 5400,
                      completion_tokens: 3600,
                      total_tokens: 9000,
                      api_requests: 18,
                      successful_requests: 17,
                      failed_requests: 1,
                      cache_read_input_tokens: 180,
                      cache_creation_input_tokens: 90,
                    },
                    metadata: { key_alias: "key-3", team_id: null },
                  },
                  "key-4": {
                    metrics: {
                      spend: 17.0,
                      prompt_tokens: 5100,
                      completion_tokens: 3400,
                      total_tokens: 8500,
                      api_requests: 17,
                      successful_requests: 16,
                      failed_requests: 1,
                      cache_read_input_tokens: 170,
                      cache_creation_input_tokens: 85,
                    },
                    metadata: { key_alias: "key-4", team_id: null },
                  },
                  "key-5": {
                    metrics: {
                      spend: 16.0,
                      prompt_tokens: 4800,
                      completion_tokens: 3200,
                      total_tokens: 8000,
                      api_requests: 16,
                      successful_requests: 15,
                      failed_requests: 1,
                      cache_read_input_tokens: 160,
                      cache_creation_input_tokens: 80,
                    },
                    metadata: { key_alias: "key-5", team_id: null },
                  },
                  "key-6": {
                    metrics: {
                      spend: 15.0,
                      prompt_tokens: 4500,
                      completion_tokens: 3000,
                      total_tokens: 7500,
                      api_requests: 15,
                      successful_requests: 14,
                      failed_requests: 1,
                      cache_read_input_tokens: 150,
                      cache_creation_input_tokens: 75,
                    },
                    metadata: { key_alias: "key-6", team_id: null },
                  },
                },
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithManyKeys, "models");

    expect(result["gpt-4"].top_api_keys).toHaveLength(5);
    expect(result["gpt-4"].top_api_keys[0].spend).toBe(20.0);
    expect(result["gpt-4"].top_api_keys[4].spend).toBe(16.0);
  });

  it("should return empty object when results array is empty", () => {
    const result = processActivityData({ results: [] }, "models");
    expect(result).toEqual({});
  });

  it("should populate top_models for api_keys when models breakdown contains api_key_breakdown for that key", () => {
    const dailyActivityWithModelsForKey: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: EMPTY_SPEND_METRICS,
          breakdown: {
            models: {
              "gpt-4": {
                metrics: EMPTY_SPEND_METRICS,
                metadata: {},
                api_key_breakdown: {
                  "api-key-hash-1": {
                    metrics: {
                      spend: 60.0,
                      prompt_tokens: 18000,
                      completion_tokens: 12000,
                      total_tokens: 30000,
                      api_requests: 60,
                      successful_requests: 57,
                      failed_requests: 3,
                      cache_read_input_tokens: 0,
                      cache_creation_input_tokens: 0,
                    },
                    metadata: { key_alias: "key-alias-1", team_id: "team1" },
                  },
                },
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {
              "api-key-hash-1": {
                metrics: {
                  spend: 60.0,
                  prompt_tokens: 18000,
                  completion_tokens: 12000,
                  total_tokens: 30000,
                  api_requests: 60,
                  successful_requests: 57,
                  failed_requests: 3,
                  cache_read_input_tokens: 0,
                  cache_creation_input_tokens: 0,
                },
                metadata: { key_alias: "key-alias-1", team_id: "team1" },
              },
            },
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithModelsForKey, "api_keys", MOCK_TEAMS);

    expect(result["api-key-hash-1"].top_models).toHaveLength(1);
    expect(result["api-key-hash-1"].top_models[0].model).toBe("gpt-4");
    expect(result["api-key-hash-1"].top_models[0].spend).toBe(60.0);
    expect(result["api-key-hash-1"].top_models[0].requests).toBe(60);
  });

  it("should not process api_key_breakdown when key is api_keys", () => {
    const dailyActivityWithBreakdown: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 100.5,
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 500,
          },
          breakdown: {
            models: {},
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {
              "key-1": {
                metrics: {
                  spend: 50.25,
                  prompt_tokens: 15000,
                  completion_tokens: 10000,
                  total_tokens: 25000,
                  api_requests: 50,
                  successful_requests: 47,
                  failed_requests: 3,
                  cache_read_input_tokens: 500,
                  cache_creation_input_tokens: 250,
                },
                metadata: {
                  key_alias: "test-key-1",
                  team_id: "team1",
                },
              },
            },
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithBreakdown, "api_keys", MOCK_TEAMS);

    expect(result["key-1"].top_api_keys).toEqual([]);
  });

  it("should handle missing cache tokens gracefully", () => {
    const dailyActivityWithoutCache: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 100.5,
            prompt_tokens: 30000,
            completion_tokens: 20000,
            total_tokens: 50000,
            api_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
          breakdown: {
            models: {
              "gpt-4": {
                metrics: {
                  spend: 100.5,
                  prompt_tokens: 30000,
                  completion_tokens: 20000,
                  total_tokens: 50000,
                  api_requests: 100,
                  successful_requests: 95,
                  failed_requests: 5,
                  cache_read_input_tokens: 0,
                  cache_creation_input_tokens: 0,
                },
                metadata: {},
                api_key_breakdown: {},
              },
            },
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {},
            entities: {},
          },
        },
      ],
    };

    const result = processActivityData(dailyActivityWithoutCache, "models");

    expect(result["gpt-4"].total_cache_read_input_tokens).toBe(0);
    expect(result["gpt-4"].total_cache_creation_input_tokens).toBe(0);
  });

  it("should handle empty breakdown gracefully", () => {
    const emptyDailyActivity: { results: DailyData[] } = {
      results: [
        {
          date: "2025-01-01",
          metrics: {
            spend: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            api_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
          breakdown: EMPTY_BREAKDOWN,
        },
      ],
    };

    const result = processActivityData(emptyDailyActivity, "models");

    expect(result).toEqual({});
  });
});

describe("formatKeyLabel", () => {
  it("should return key_alias when no team_id is present", () => {
    const modelData = createMockKeyMetricWithMetadata({
      key_alias: "test-key",
      team_id: null,
    });

    const result = formatKeyLabel(modelData, "test-key", MOCK_TEAMS);
    expect(result).toBe("test-key");
  });

  it("should return key_alias with team alias when team_id matches", () => {
    const modelData = createMockKeyMetricWithMetadata({
      key_alias: "test-key",
      team_id: "team1",
    });

    const result = formatKeyLabel(modelData, "test-key", MOCK_TEAMS);
    expect(result).toBe("test-key (team: Test Team 1)");
  });

  it("should return key_alias with team_id when team is not found", () => {
    const modelData = createMockKeyMetricWithMetadata({
      key_alias: "test-key",
      team_id: "nonexistent-team",
    });

    const result = formatKeyLabel(modelData, "test-key", MOCK_TEAMS);
    expect(result).toBe("test-key (team_id: nonexistent-team)");
  });

  it("should use key-hash fallback when key_alias is null", () => {
    const modelData = createMockKeyMetricWithMetadata({
      key_alias: null,
      team_id: "team1",
    });

    const result = formatKeyLabel(modelData, "actual-key", MOCK_TEAMS);
    expect(result).toBe("key-hash-actual-key (team: Test Team 1)");
  });

  it("should return key_alias with team_id when teams array is empty", () => {
    const modelData = createMockKeyMetricWithMetadata({
      key_alias: "my-key",
      team_id: "team1",
    });

    const result = formatKeyLabel(modelData, "my-key", []);
    expect(result).toBe("my-key (team_id: team1)");
  });
});
