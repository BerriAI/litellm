import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { paths } from "@/lib/http/schema";
import type { DailyActivityRange } from "./useDailyActivityRange";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({ $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) } }));
vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: () => <div data-testid="date-picker" />,
}));

import CacheLeakageCard from "./CacheLeakageCard";

type CacheLeakageKeysResponse =
  paths["/user/daily/activity/cache_leakage"]["get"]["responses"][200]["content"]["application/json"];
type ServerKeyRow = CacheLeakageKeysResponse["results"][number];

const serverKey = (apiKey: string, overrides: Partial<ServerKeyRow> = {}): ServerKeyRow => ({
  api_key: apiKey,
  key_alias: null,
  team_id: null,
  prompt_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  uncached_prompt_tokens: 0,
  cache_hit_ratio: 0,
  prompt_caching_savings_spend: 0,
  ...overrides,
});

const serverResponse = (
  results: ServerKeyRow[],
  overrides: Partial<CacheLeakageKeysResponse["metadata"]> = {},
): CacheLeakageKeysResponse => ({
  results,
  metadata: {
    total_api_keys: results.length,
    limit: 10,
    total_cached_tokens: 0,
    total_prompt_caching_savings_spend: 0,
    ...overrides,
  },
});

const mockKeyQuery = (result: {
  data?: CacheLeakageKeysResponse;
  isPending?: boolean;
  isError?: boolean;
  isFetching?: boolean;
  refetch?: () => unknown;
}) => {
  const queryResult = {
    data: result.data,
    isPending: result.isPending ?? false,
    isError: result.isError ?? false,
    isFetching: result.isFetching ?? false,
    refetch: result.refetch ?? vi.fn(),
  };
  useQueryMock.mockReturnValue(queryResult);
};

const baseMetrics = (overrides: Partial<SpendMetrics>): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...overrides,
});

const dayWithModels = (date: string, models: Record<string, Partial<SpendMetrics>>): DailyData => ({
  date,
  metrics: baseMetrics({}),
  breakdown: {
    models: Object.fromEntries(
      Object.entries(models).map(([name, m]) => [
        name,
        { metrics: baseMetrics(m), metadata: {}, api_key_breakdown: {} },
      ]),
    ),
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  },
});

const renderWith = (results: DailyData[], overrides: Partial<DailyActivityRange> = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <CacheLeakageCard
        accessToken="sk-test"
        scopeUserId={null}
        activity={{
          dateValue: {},
          onDateChange: vi.fn(),
          results,
          loading: false,
          isFetchingMore: false,
          progress: { currentPage: 1, totalPages: 1 },
          cancelled: false,
          failed: false,
          cancel: vi.fn(),
          ...overrides,
        }}
      />
    </QueryClientProvider>,
  );
};

const leakyOverrides: Partial<ServerKeyRow> = {
  key_alias: "leaky-key",
  prompt_tokens: 10000,
  uncached_prompt_tokens: 10000,
  cache_hit_ratio: 0,
};

const cachingOverrides: Partial<ServerKeyRow> = {
  key_alias: "caching-key",
  prompt_tokens: 1000,
  cache_read_input_tokens: 900,
  uncached_prompt_tokens: 100,
  cache_hit_ratio: 0.9,
};

describe("CacheLeakageCard", () => {
  it("ranks leaking keys by the server ordering and shows cache hit ratio", () => {
    mockKeyQuery({
      data: serverResponse([serverKey("hash-leaky", leakyOverrides), serverKey("hash-caching", cachingOverrides)]),
    });
    renderWith([]);

    expect(screen.getByText("leaky-key")).toBeInTheDocument();
    expect(screen.getByText("0.0%")).toBeInTheDocument();
    expect(screen.getByText("90.0%")).toBeInTheDocument();
    [
      "Input tokens you sent in this range that weren't served from or written to the cache",
      "Share of your input tokens that were served from the cache",
      "About how much you'd save if this uncached input used prompt caching. Estimated as uncached input tokens times what your cached traffic already nets per cached token (realized cache savings, after write premiums, ÷ cache read and write tokens). Blank when caching is not currently saving anything overall.",
    ].forEach((info) => expect(screen.getByLabelText(info)).toBeInTheDocument());
  });

  it("sorts by the clicked column, worst cache hit rate first", () => {
    mockKeyQuery({
      data: serverResponse([
        serverKey("hash-a", {
          key_alias: "alpha",
          uncached_prompt_tokens: 1000,
          cache_hit_ratio: 0.9,
        }),
        serverKey("hash-b", {
          key_alias: "bravo",
          uncached_prompt_tokens: 450,
          cache_hit_ratio: 0.1,
        }),
      ]),
    });
    renderWith([]);
    const firstDataRow = () => screen.getAllByRole("row")[1];

    expect(firstDataRow()).toHaveTextContent("alpha");

    fireEvent.click(screen.getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("bravo");

    fireEvent.click(screen.getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("alpha");
  });

  it("switches to the model view and lists models from every provider", () => {
    mockKeyQuery({ data: serverResponse([]) });
    renderWith([
      dayWithModels("2026-07-12", {
        "claude-sonnet-5": { prompt_tokens: 5000, cache_read_input_tokens: 0 },
        "vertex_ai/gemini-2.5-pro": { prompt_tokens: 8000, cache_read_input_tokens: 2000 },
      }),
    ]);

    fireEvent.click(screen.getByText("By model"));

    expect(screen.getByText("Cache leakage by model")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-5")).toBeInTheDocument();
    expect(screen.getByText("vertex_ai/gemini-2.5-pro")).toBeInTheDocument();
  });

  it("shows an empty state when no key leaked input in the range", () => {
    mockKeyQuery({ data: serverResponse([]) });
    renderWith([]);

    expect(screen.getByText("No key usage in this range.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("says the key ranking is loading while the server read is pending", () => {
    mockKeyQuery({ isPending: true });
    renderWith([]);

    expect(screen.getByRole("status")).toHaveTextContent("Loading key ranking...");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("offers a retry when the key ranking fails to load", () => {
    const refetch = vi.fn();
    mockKeyQuery({ isError: true, refetch });
    renderWith([]);

    expect(screen.getByRole("alert")).toHaveTextContent("Could not load the key ranking");
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(refetch).toHaveBeenCalled();
  });

  it("tells the user the model table is still filling in while fallback pages stream", () => {
    mockKeyQuery({ data: serverResponse([]) });
    const day = dayWithModels("2026-07-12", {
      "claude-sonnet-5": { prompt_tokens: 10000, cache_read_input_tokens: 0 },
    });
    renderWith([day], { isFetchingMore: true });

    fireEvent.click(screen.getByText("By model"));

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(
      screen.getByText("Data is still loading; rows and totals will update as the rest of the range arrives."),
    ).toBeInTheDocument();
  });

  it("keeps the streaming note off the model view while a fresh range loads over the previous range's rows", () => {
    mockKeyQuery({ data: serverResponse([]) });
    const day = dayWithModels("2026-07-12", {
      "claude-sonnet-5": { prompt_tokens: 10000, cache_read_input_tokens: 0 },
    });
    renderWith([day], { loading: true });

    fireEvent.click(screen.getByText("By model"));

    expect(
      screen.queryByText("Data is still loading; rows and totals will update as the rest of the range arrives."),
    ).not.toBeInTheDocument();
  });

  it("drops the streaming note once the range has settled", () => {
    mockKeyQuery({ data: serverResponse([]) });
    const day = dayWithModels("2026-07-12", {
      "claude-sonnet-5": { prompt_tokens: 10000, cache_read_input_tokens: 0 },
    });
    renderWith([day]);

    fireEvent.click(screen.getByText("By model"));

    expect(
      screen.queryByText("Data is still loading; rows and totals will update as the rest of the range arrives."),
    ).not.toBeInTheDocument();
  });

  it("never shows a spend-cap note for the key ranking", () => {
    mockKeyQuery({
      data: serverResponse([serverKey("hash-leaky", { key_alias: "leaky-key", uncached_prompt_tokens: 10000 })]),
    });
    renderWith([]);

    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });
});
