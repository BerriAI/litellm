import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import { all_admin_roles } from "@/utils/roles";
import { LiteAdminFrame } from "@/components/liteadmin/LiteAdmin";
import HomePage from "./HomePage";
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

const auth = vi.hoisted(() => ({
  accessToken: "sk-test",
  userRole: "",
  userId: "u1",
  isViewOnly: false,
  isAuthorized: true,
  isLoading: false,
  premiumUser: true,
  token: "test-token",
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));

const network = vi.hoisted(() => ({
  aggregated: vi.fn(),
  gateway: vi.fn(),
  userInfo: vi.fn(),
}));
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  dailyActivityAggregatedCall: (...args: unknown[]) => network.aggregated(...args),
  gatewayDailyActivityCall: (...args: unknown[]) => network.gateway(...args),
  userGetInfoV2: (...args: unknown[]) => network.userInfo(...args),
  getProxyBaseUrl: () => "http://proxy.test",
}));

const requestUrl = (input: RequestInfo | URL): string => {
  if (typeof input === "string") return input;
  return input instanceof URL ? input.href : input.url;
};

const fakeFetch = (input: RequestInfo | URL) => {
  if (requestUrl(input).endsWith("/models")) {
    return Promise.resolve(Response.json({ data: [{ id: "claude-sonnet-5-5" }] }));
  }
  if (requestUrl(input).includes("/model_group/info")) {
    return Promise.resolve(Response.json({ data: [{ model_group: "claude-sonnet-5-5", mode: "chat" }] }));
  }
  if (requestUrl(input).endsWith("/sso/get/ui_settings")) {
    return Promise.resolve(Response.json({ PROXY_BASE_URL: "http://proxy.test" }));
  }
  return Promise.reject(new Error(`Unexpected request: ${requestUrl(input)}`));
};

const dayResult = (date: string, model: string, tokens: number) => {
  const metrics = {
    spend: 1,
    prompt_tokens: tokens,
    completion_tokens: 0,
    total_tokens: tokens,
    api_requests: 1,
    successful_requests: 1,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
  return {
    date,
    metrics,
    breakdown: {
      models: {},
      model_groups: { [model]: { metrics, metadata: {}, api_key_breakdown: {} } },
      providers: {},
      api_keys: {},
      entities: {},
      mcp_servers: {},
    },
  };
};

const activity: DailyActivityAggregatedResponse = {
  results: [
    dayResult("2026-10-08", "gpt-6.1-sol", 900),
    dayResult("2026-10-09", "claude-haiku-5-5", 100),
  ] as unknown as DailyActivityAggregatedResponse["results"],
  metadata: {
    ...EMPTY_DAILY_ACTIVITY_METADATA,
    total_spend: 501.74,
    total_api_requests: 5118,
    total_successful_requests: 5075,
    total_failed_requests: 43,
    total_tokens: 98_600_000,
  },
};

const renderHome = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <HomePage />
    </QueryClientProvider>,
  );

describe("HomePage", () => {
  beforeEach(() => {
    auth.isViewOnly = false;
    auth.userRole = all_admin_roles[0]!;
    network.aggregated.mockReset().mockResolvedValue(activity);
    network.gateway.mockReset().mockResolvedValue({ total_successful_requests: 0, total_failed_requests: 0 });
    network.userInfo.mockReset().mockResolvedValue({ user_info: { max_budget: null } });
    vi.stubGlobal("fetch", vi.fn(fakeFetch));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the usage totals from the daily activity API and links Create New Virtual Key to the key creation flow", async () => {
    renderHome();
    expect(screen.getByRole("heading", { level: 1, name: "Home" })).toBeInTheDocument();
    expect(await screen.findByText("$501.74")).toBeInTheDocument();
    expect(screen.getByText("5,118")).toBeInTheDocument();
    expect(screen.getByText("98.6M")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /create new virtual key/i })).toHaveAttribute(
      "href",
      expect.stringMatching(/\/api-keys\?create=true$/),
    );
    expect(network.aggregated).toHaveBeenCalledWith("user", expect.objectContaining({ accessToken: "sk-test" }));
  });

  it("hides Create New Virtual Key from view-only users", () => {
    auth.isViewOnly = true;
    renderHome();
    expect(screen.queryByRole("link", { name: /create new virtual key/i })).not.toBeInTheDocument();
  });

  it("keeps Home focused on usage without an embedded Aria composer", async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <LiteAdminFrame>
          <HomePage />
        </LiteAdminFrame>
      </QueryClientProvider>,
    );
    expect(screen.queryByRole("region", { name: "Ask Aria Agent" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Send message" })).not.toBeInTheDocument();
    expect(await screen.findByText("$501.74")).toBeInTheDocument();
  });

  it("loads usage without requesting promotional content on Home", async () => {
    renderHome();
    expect(await screen.findByText("$501.74")).toBeVisible();
    expect(vi.mocked(fetch).mock.calls.map(([input]) => requestUrl(input))).not.toContain(
      "http://proxy.test/public/litellm_blog_posts",
    );
    expect(screen.queryByRole("heading", { name: /what's new/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Updates" })).not.toBeInTheDocument();
  });

  it("shows an error with a retry instead of zero totals when the usage request fails", async () => {
    network.aggregated.mockRejectedValueOnce(new Error("boom"));
    renderHome();
    expect(await screen.findByText("Failed to load usage")).toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("$501.74")).toBeInTheDocument();
    expect(network.aggregated).toHaveBeenCalledTimes(2);
  });

  it("ranks model groups by token share on the leaderboard", async () => {
    renderHome();
    const board = screen.getByTestId("home-leaderboard");
    expect(await within(board).findByText("gpt-6.1-sol")).toBeInTheDocument();
    const rows = within(board).getAllByRole("listitem");
    expect(within(rows[0]!).getByText("gpt-6.1-sol")).toBeInTheDocument();
    expect(within(rows[0]!).getByText("90.0%")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("claude-haiku-5-5")).toBeInTheDocument();
  });
});
