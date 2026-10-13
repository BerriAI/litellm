import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import { all_admin_roles } from "@/utils/roles";
import { LiteAdminFrame } from "@/components/liteadmin/LiteAdmin";
import HomePage from "./HomePage";

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

const post = (title: string, date: string, tags: string[] = []) => ({
  title,
  description: `${title} description`,
  date,
  url: title === "Post one" ? "https://x/1" : `https://docs.litellm.ai/blog/${title}`,
  tags,
});

const blogPosts = vi.hoisted(() => ({ posts: [] as unknown[] }));

const requestUrl = (input: RequestInfo | URL): string => {
  if (typeof input === "string") return input;
  return input instanceof URL ? input.href : input.url;
};

const fakeFetch = (input: RequestInfo | URL) => {
  if (requestUrl(input).includes("/model_group/info")) {
    return Promise.resolve(Response.json({ data: [{ model_group: "claude-sonnet-5-5", mode: "chat" }] }));
  }
  if (requestUrl(input).endsWith("/sso/get/ui_settings")) {
    return Promise.resolve(Response.json({ PROXY_BASE_URL: "http://proxy.test" }));
  }
  return Promise.resolve(
    new Response(JSON.stringify(blogPosts), { status: 200, headers: { "Content-Type": "application/json" } }),
  );
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
    blogPosts.posts = [post("Post one", "2026-10-09"), post("mid", "2026-10-06", ["day 0 support"])];
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

  it("offers LiteAdmin on Home inside the dashboard frame", async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <LiteAdminFrame>
          <HomePage />
        </LiteAdminFrame>
      </QueryClientProvider>,
    );
    const entry = screen.getByRole("region", { name: "Ask LiteAdmin" });
    expect(await within(entry).findByPlaceholderText("Ask LiteAdmin…")).toBeInTheDocument();
    expect(within(entry).getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(await screen.findByText("$501.74")).toBeInTheDocument();
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

  it("renders the three newest day 0 support blog posts as What's new, skipping untagged posts", async () => {
    blogPosts.posts = [
      post("mid", "2026-10-06", ["openai", "day 0 support"]),
      post("townhall", "2026-10-12", ["announcement"]),
      post("new", "2026-10-11", ["day 0 support"]),
      post("old", "2026-09-30", ["day 0 support"]),
      post("oldest", "2026-09-01", ["day 0 support"]),
    ];
    renderHome();
    const section = await screen.findByTestId("home-whats-new");
    expect(await within(section).findByText("new")).toBeInTheDocument();
    expect(
      within(section)
        .getAllByRole("link")
        .map((link) => link.getAttribute("href")),
    ).toEqual([
      "https://docs.litellm.ai/blog/new",
      "https://docs.litellm.ai/blog/mid",
      "https://docs.litellm.ai/blog/old",
    ]);
    expect(within(section).getByText("Oct 11")).toBeInTheDocument();
    expect(within(section).queryByText("townhall")).not.toBeInTheDocument();
  });

  it("hides What's new when no blog post is tagged day 0 support", async () => {
    blogPosts.posts = [post("Post one", "2026-10-09", ["announcement"])];
    renderHome();
    expect(await screen.findByRole("link", { name: /post one/i })).toBeInTheDocument();
    expect(screen.queryByTestId("home-whats-new")).not.toBeInTheDocument();
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

  it("renders the blog posts fetched from /public/litellm_blog_posts in the Updates feed", async () => {
    renderHome();
    const updates = screen.getByTestId("home-updates");
    expect(await within(updates).findByRole("link", { name: /post one/i })).toHaveAttribute("href", "https://x/1");
    expect(fetch).toHaveBeenCalledWith("http://proxy.test/public/litellm_blog_posts");
  });
});
