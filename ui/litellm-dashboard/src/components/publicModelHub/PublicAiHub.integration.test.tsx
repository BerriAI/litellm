import { fireEvent, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Plugin } from "@/components/claude_code_plugins/types";
import type { apiClient } from "@/components/networking";

import { chooseSelectOption, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import PublicAiHub from "./PublicAiHub";

const { apiGetMock, realGet, navigation, uiSettings, replaceMock, hubInfo } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
  realGet: {
    current: (..._args: Parameters<typeof apiClient.get>): Promise<unknown> => Promise.reject(new Error("unset")),
  },
  navigation: { params: new URLSearchParams() },
  uiSettings: { values: {} as Record<string, unknown> },
  replaceMock: vi.fn(),
  hubInfo: {
    docs_title: "",
    custom_docs_description: null,
    litellm_version: "1.90.0",
    useful_links: { "Getting started": { url: "https://docs.example.com", index: 0 } },
  },
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => navigation.params,
  usePathname: () => "/model_hub_table",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("next/font/local", () => ({ default: () => ({ variable: "catalog-display-font" }) }));

vi.mock("@/components/navbar", () => ({
  default: ({ publicNavigation }: { publicNavigation?: ReactNode }) => (
    <nav aria-label="Top bar">{publicNavigation}</nav>
  ),
}));

vi.mock("@/lib/telemetry/uiEvents", () => ({ recordUiEvent: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: uiSettings, isLoading: false }),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  realGet.current = actual.apiClient.get;
  return {
    ...actual,
    apiClient: { ...actual.apiClient, get: apiGetMock },
    getUiConfig: vi.fn().mockResolvedValue({}),
    getProxyBaseUrl: vi.fn(() => "https://gateway.example.com"),
    getPublicModelHubInfo: vi.fn().mockResolvedValue(hubInfo),
  };
});

const MODEL_HUB_PATH = "/public/v1/model_hub";
const SKILL_HUB_PATH = "/public/skill_hub";

const MODELS = [
  {
    model_group: "claude-sonnet-4-5",
    providers: ["anthropic"],
    mode: "chat",
    input_cost_per_token: 0.000003,
    max_input_tokens: 200000,
    description: "Best for agentic coding",
  },
  {
    model_group: "gpt-5",
    providers: ["openai"],
    mode: "chat",
    input_cost_per_token: 0,
    output_cost_per_token: 0,
    max_input_tokens: 0,
    description: "General purpose",
    deployment_pricing: [
      {
        provider: "azure",
        costs: { output_cost_per_token: 1e-5, output_cost_per_token_priority: 2e-5 },
        tiered_pricing: [],
      },
    ],
  },
];

const SKILLS: Plugin[] = [
  {
    download_count: 5,
    id: "skill-research",
    name: "research-assistant",
    description: "Answers research questions",
    version: "1.4.0",
    source: { source: "github", repo: "example/research-tools" },
    namespace: "knowledge",
    domain: "Research",
    category: "Analysis",
    keywords: ["evidence"],
    author: { name: "Research maintainers" },
    created_by: "writer@example.test",
    approved_by: "reviewer@example.test",
    security_scan_passed: true,
    enabled: true,
    created_at: null,
    updated_at: null,
  },
  {
    download_count: 9,
    id: "skill-review",
    name: "review-changes",
    description: "Checks pull requests",
    version: "2.0.0",
    source: { source: "github", repo: "example/review-tools" },
    namespace: "development",
    domain: "Engineering",
    category: "Code review",
    keywords: ["quality"],
    enabled: true,
    created_at: null,
    updated_at: null,
  },
  {
    download_count: 2,
    id: "skill-release",
    name: "release-notes",
    description: "Summarizes shipped changes",
    version: "1.0.0",
    source: { source: "github", repo: "example/release-tools" },
    namespace: "delivery",
    domain: "Engineering",
    category: "Documentation",
    keywords: ["changelog"],
    enabled: true,
    created_at: null,
    updated_at: null,
  },
];

const FACETS: Record<string, string[]> = {
  [`${MODEL_HUB_PATH}/providers`]: ["anthropic", "openai"],
  [`${MODEL_HUB_PATH}/modes`]: ["chat"],
  [`${MODEL_HUB_PATH}/features`]: ["vision", "function_calling"],
};

const page = (data: unknown[]) => ({
  data,
  meta: { total_count: data.length, page: 1, page_size: 50, total_pages: 1 },
  links: { self: MODEL_HUB_PATH, first: MODEL_HUB_PATH, prev: null, next: null, last: MODEL_HUB_PATH },
});

const OFF_PAGE_MODEL = { model_group: "o1-pro", providers: ["openai"], mode: "chat", description: "Deep reasoning" };

type Query = NonNullable<Parameters<typeof apiClient.get>[1]>;

const respond = (path: string, options?: Query): Promise<unknown> => {
  if (path === SKILL_HUB_PATH) return Promise.resolve({ plugins: SKILLS });
  if (path in FACETS) return Promise.resolve(page(FACETS[path]));
  if (path !== MODEL_HUB_PATH) return Promise.reject(new Error(`Unexpected API read: ${path}`));
  const q = String(options?.query?.q ?? "");
  if (q === "o1-pro") return Promise.resolve(page([OFF_PAGE_MODEL]));
  return Promise.resolve(page(MODELS.filter((model) => model.model_group.includes(q))));
};

const respondWithModels = () => apiGetMock.mockImplementation(respond);

const failModelsWith = (next: (options?: Query) => Promise<unknown>) =>
  apiGetMock.mockImplementation((path: string, options?: Query) =>
    path === MODEL_HUB_PATH ? next(options) : respond(path, options),
  );

const tokenExpiringIn = (seconds: number) => {
  const encode = (value: object) => btoa(JSON.stringify(value)).replace(/=+$/, "");
  return `${encode({ alg: "HS256" })}.${encode({ exp: Math.floor(Date.now() / 1000) + seconds })}.signature`;
};

beforeEach(() => {
  vi.clearAllMocks();
  testQueryClient.clear();
  uiSettings.values = {};
  navigation.params = new URLSearchParams();
  document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
  respondWithModels();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 404 }));
  Object.defineProperty(window, "location", {
    writable: true,
    value: { ...window.location, replace: replaceMock },
  });
});

describe("PublicAiHub", () => {
  it("shows only Models and Skills without selecting an item or reading retired hubs", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByRole("link", { name: /claude-sonnet-4-5/ })).toBeInTheDocument();
    expect(screen.getByText("2 models")).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(2);
    expect(screen.getByRole("tab", { name: "Model Hub" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Skills Hub" })).toHaveAttribute("aria-selected", "false");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Base URL" })).toHaveValue("https://gateway.example.com");

    await user.click(screen.getByRole("tab", { name: "Skills Hub" }));

    expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
    expect(screen.getByText("3 skills")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(apiGetMock.mock.calls.map(([path]) => path)).not.toEqual(
      expect.arrayContaining([expect.stringMatching(/^\/public\/(?:agent|mcp)_hub$/)]),
    );
  });

  it("opens a model from the list and records it in the URL", async () => {
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { onUrlUpdate });

    await user.click(await screen.findByRole("link", { name: /gpt-5/ }));

    const dialog = await screen.findByRole("dialog", { name: "Details" });
    expect(within(dialog).getByRole("heading", { level: 2, name: "gpt-5" })).toBeInTheDocument();
    expect(within(dialog).getByText("General purpose")).toBeInTheDocument();
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenCalledWith(expect.objectContaining({ queryString: "?open=gpt-5" })),
    );
  });

  it("explains an empty model search and clears the server query", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />);

    fireEvent.change(await screen.findByRole("textbox", { name: "Search models" }), { target: { value: "billing" } });

    expect(await screen.findByText('No models match "billing"')).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(await screen.findByRole("link", { name: /gpt-5/ })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Search models" })).toHaveValue("");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("lists the admin's links and the gateway version", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />);

    await user.click(await screen.findByRole("button", { name: "Links" }));

    expect(await screen.findByRole("menuitem", { name: "Getting started" })).toHaveAttribute(
      "href",
      "https://docs.example.com",
    );
    expect(screen.getByText("LiteLLM v1.90.0")).toBeInTheDocument();
  });

  it("keeps skills reachable when only the models call fails", async () => {
    const user = userEvent.setup();
    failModelsWith(() => Promise.reject(new Error("No models configured")));
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByText("Couldn't load models")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Search models" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Skills Hub" }));
    expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
  });

  it("says the hub is unavailable when every hub call fails", async () => {
    apiGetMock.mockRejectedValue(new Error("Not found"));
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByText("This AI Hub isn't available")).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Search models" })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows an HTTP failure for Skills and retries without losing access to Models", async () => {
    const user = userEvent.setup();
    vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify({ detail: "Database error" }), { status: 500 }));
    apiGetMock.mockImplementation((path: string, options?: Query) =>
      path === SKILL_HUB_PATH ? realGet.current(path, options) : respond(path, options),
    );
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByRole("link", { name: /gpt-5/ })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Skills Hub" }));

    expect(await screen.findByText("Couldn't load skills")).toBeInTheDocument();
    expect(screen.queryByText("No skills published yet")).not.toBeInTheDocument();
    vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify({ plugins: SKILLS }), { status: 200 }));
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
    expect(screen.queryByText("Couldn't load skills")).not.toBeInTheDocument();
  });

  it("shows loading rather than an empty hub while Retry reloads models", async () => {
    const user = userEvent.setup();
    let finishRetry: (value: unknown) => void = () => {};
    failModelsWith(() => Promise.reject(new Error("Gateway timeout")));
    renderWithProviders(<PublicAiHub accessToken={null} />);
    await screen.findByText("Couldn't load models");
    failModelsWith(() => new Promise((resolve) => (finishRetry = resolve)));

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(screen.getByText("Loading models…")).toBeInTheDocument();
    expect(screen.queryByText("No models published yet")).not.toBeInTheDocument();
    expect(screen.queryByText("Couldn't load models")).not.toBeInTheDocument();
    finishRetry(page(MODELS));
    expect(await screen.findByRole("link", { name: /claude-sonnet-4-5/ })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps model details and the selected snippet when an earlier search finishes", async () => {
    const user = userEvent.setup();
    let finishSearch: (value: unknown) => void = () => {};
    failModelsWith((options) =>
      options?.query?.q === "claude"
        ? new Promise((resolve) => (finishSearch = resolve))
        : respond(MODEL_HUB_PATH, options),
    );
    renderWithProviders(<PublicAiHub accessToken={null} />);
    await screen.findByRole("link", { name: /gpt-5/ });
    fireEvent.change(screen.getByRole("textbox", { name: "Search models" }), { target: { value: "claude" } });
    fireEvent.click(screen.getByRole("link", { name: /gpt-5/ }));
    await screen.findByRole("heading", { level: 2, name: "gpt-5" });
    await user.click(screen.getByRole("tab", { name: "cURL" }));

    await waitFor(() =>
      expect(apiGetMock).toHaveBeenCalledWith(
        MODEL_HUB_PATH,
        expect.objectContaining({ query: expect.objectContaining({ q: "claude" }) }),
      ),
    );
    finishSearch(page([MODELS[0]]));

    expect(await screen.findByText("1 model")).toBeInTheDocument();
    expect(screen.getAllByText("gpt-5")).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 2, name: "gpt-5" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "cURL" })).toHaveAttribute("aria-selected", "true");
    expect(within(screen.getByRole("dialog")).getByRole("tabpanel", { name: "cURL" })).toHaveTextContent(
      '"model": "gpt-5"',
    );
    expect(apiGetMock).not.toHaveBeenCalledWith(
      MODEL_HUB_PATH,
      expect.objectContaining({ query: expect.objectContaining({ q: "gpt-5" }) }),
    );
  });

  it("looks up shared model details even when the catalog list is empty", async () => {
    failModelsWith((options) => Promise.resolve(page(options?.query?.q === "gpt-5" ? [MODELS[1]] : [])));
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "gpt-5" } });

    expect(await screen.findByRole("heading", { level: 2, name: "gpt-5" })).toBeInTheDocument();
    expect(screen.queryByText("“gpt-5” isn't listed here")).not.toBeInTheDocument();
  });

  it("distinguishes zero and missing prices in the model table", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />);

    const freeRow = await screen.findByRole("row", { name: /gpt-5/ });
    const pricedRow = screen.getByRole("row", { name: /claude-sonnet-4-5/ });

    expect(within(freeRow).getAllByRole("cell", { name: "Free" })).toHaveLength(2);
    expect(within(freeRow).getByRole("cell", { name: "0" })).toBeInTheDocument();
    expect(within(pricedRow).getByRole("cell", { name: "$3.00" })).toBeInTheDocument();
    expect(within(pricedRow).getByRole("cell", { name: "-" })).toBeInTheDocument();
    expect(within(pricedRow).queryByRole("cell", { name: "Free" })).not.toBeInTheDocument();
  });

  it("shows the open model's prices for each of its deployments", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />);

    await user.click(await screen.findByRole("link", { name: /gpt-5/ }));

    expect(await screen.findByRole("heading", { name: "Pricing by deployment" })).toBeInTheDocument();
    const azure = screen.getByRole("region", { name: "Azure pricing" });
    expect(within(azure).getByRole("row", { name: /Output \/ 1M tokens/ })).toHaveTextContent("$10.00$20.00");
  });

  it("leaves the pricing breakdown out for a model without deployment prices", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "claude-sonnet-4-5" } });

    expect(await screen.findByRole("heading", { level: 2, name: "claude-sonnet-4-5" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Pricing by deployment" })).not.toBeInTheDocument();
  });

  it("treats an empty open param as nothing opened", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "" } });

    expect(await screen.findByRole("link", { name: /claude-sonnet-4-5/ })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it.each(["models", "skills"])("reports a missing shared item in the %s dialog", async (tab) => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab, open: "retired-item" } });

    const dialog = await screen.findByRole("dialog", { name: "Details" });

    expect(await within(dialog).findByText("“retired-item” isn't listed here")).toBeInTheDocument();
    expect(within(dialog).queryByRole("heading", { level: 2, name: "claude-sonnet-4-5" })).not.toBeInTheDocument();
  });

  it.each([
    { tab: "models", name: "gpt-5" },
    { tab: "skills", name: "research-assistant" },
  ])("preserves query params and native modified clicks on $tab links", async ({ tab, name }) => {
    const onUrlUpdate = vi.fn();
    const searchParams = { tab, key: "abc", ref: "shared catalog" };
    navigation.params = new URLSearchParams(searchParams);
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams, onUrlUpdate });

    const link = await screen.findByRole("link", { name: new RegExp(name) });

    expect(link).toHaveAttribute("href", `?tab=${tab}&key=abc&ref=shared+catalog&open=${name}`);
    for (const modifier of ["ctrlKey", "metaKey", "shiftKey", "altKey"]) {
      expect(fireEvent.click(link, { [modifier]: true })).toBe(true);
    }
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("clears the detail URL on close and opens Skills without the old model selection", async () => {
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "gpt-5" }, onUrlUpdate });

    expect(await screen.findByRole("heading", { level: 2, name: "gpt-5" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() => expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: "" })));
    await user.click(screen.getByRole("tab", { name: "Skills Hub" }));

    expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: "?tab=skills" })),
    );
  });

  it("opens a shared link to a model that is not on the first page", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "o1-pro" } });

    expect(await screen.findByRole("heading", { level: 2, name: "o1-pro" })).toBeInTheDocument();
    expect(screen.getByText("Deep reasoning")).toBeInTheDocument();
    expect(apiGetMock).toHaveBeenCalledWith(
      MODEL_HUB_PATH,
      expect.objectContaining({ query: { q: "o1-pro", page_size: 50 } }),
    );
  });

  it.each(["agents", "mcp"])("immediately discards incompatible selection from a retired %s tab", async (tab) => {
    const onUrlUpdate = vi.fn();
    renderWithProviders(<PublicAiHub accessToken={null} />, {
      searchParams: { tab, open: "gpt-5", ref: "shared" },
      onUrlUpdate,
      resetUrlUpdateQueueOnMount: false,
    });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(await screen.findByRole("link", { name: /gpt-5/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Model Hub" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: "?ref=shared" })),
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(apiGetMock).not.toHaveBeenCalledWith(
      MODEL_HUB_PATH,
      expect.objectContaining({ query: expect.objectContaining({ q: "gpt-5" }) }),
    );
  });

  it.each(["missing", "expired"])("redirects a visitor with a %s session when sign-in is required", async (session) => {
    uiSettings.values = { require_auth_for_public_ai_hub: true };
    if (session === "expired") document.cookie = `token=${tokenExpiringIn(-3600)}; path=/`;
    renderWithProviders(<PublicAiHub accessToken={null} />);

    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith(expect.stringContaining("login")));
  });

  it("keeps a signed-in visitor on the hub when the admin requires sign-in", async () => {
    uiSettings.values = { require_auth_for_public_ai_hub: true };
    document.cookie = `token=${tokenExpiringIn(3600)}; path=/`;
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByRole("link", { name: /claude-sonnet-4-5/ })).toBeInTheDocument();
    expect(replaceMock).not.toHaveBeenCalled();
  });

  it("wires model search, provider, capability, mode, and sorting controls into the list API", async () => {
    const user = userEvent.setup();
    failModelsWith((options) =>
      options?.query?.["filter[providers][in]"] ? Promise.resolve(page([MODELS[1]])) : respond(MODEL_HUB_PATH, options),
    );
    renderWithProviders(<PublicAiHub accessToken={null} />);
    await screen.findByRole("link", { name: /claude-sonnet-4-5/ });

    await user.click(screen.getByRole("checkbox", { name: /OpenAI/ }));
    await user.click(screen.getByRole("checkbox", { name: "Vision" }));
    await user.click(screen.getByRole("checkbox", { name: "Chat" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Search models" }), { target: { value: "gpt" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Sort by" }), "Input price");

    await waitFor(() =>
      expect(apiGetMock).toHaveBeenCalledWith(
        MODEL_HUB_PATH,
        expect.objectContaining({
          query: {
            page: 1,
            page_size: 50,
            q: "gpt",
            sort: "input_cost_per_token",
            "filter[providers][in]": "openai",
            "filter[features][in]": "vision",
            "filter[mode][in]": "chat",
          },
        }),
      ),
    );
    expect(await screen.findByText("1 model")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /gpt-5/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /claude-sonnet-4-5/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Sort descending" }));

    await waitFor(() =>
      expect(apiGetMock).toHaveBeenCalledWith(
        MODEL_HUB_PATH,
        expect.objectContaining({ query: expect.objectContaining({ sort: "-input_cost_per_token", q: "gpt" }) }),
      ),
    );
    await user.click(screen.getByRole("button", { name: "Clear filters" }));

    await waitFor(() =>
      expect(apiGetMock).toHaveBeenLastCalledWith(
        MODEL_HUB_PATH,
        expect.objectContaining({ query: { page: 1, page_size: 50, q: "gpt", sort: "-input_cost_per_token" } }),
      ),
    );
    expect(screen.getByRole("checkbox", { name: /OpenAI/ })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Vision" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Chat" })).not.toBeChecked();
  });

  it("combines skill domain and search filters and clears an empty result", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab: "skills" } });
    await screen.findByRole("link", { name: /research-assistant/ });

    await user.click(screen.getByRole("checkbox", { name: "Engineering" }));
    expect(screen.queryByRole("link", { name: /research-assistant/ })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /review-changes/ })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /release-notes/ })).toBeInTheDocument();
    expect(screen.getByText("2 skills")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: /Search skills/ }), { target: { value: "changelog" } });
    expect(screen.getByRole("link", { name: /release-notes/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /review-changes/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: /Search skills/ }), { target: { value: "evidence" } });
    expect(screen.getByText('No skills match "evidence"')).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^All domains/ }));
    expect(screen.getByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /release-notes/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getByText("3 skills")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: /Search skills/ })).toHaveValue("");
    expect(screen.getByRole("checkbox", { name: "Engineering" })).not.toBeChecked();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("finds skills by name, description, namespace, domain, and keywords", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab: "skills" } });
    await screen.findByRole("link", { name: /research-assistant/ });
    const search = screen.getByRole("textbox", { name: /Search skills/ });

    for (const value of ["research-assistant", "Answers research questions", "knowledge", "Research", "evidence"]) {
      fireEvent.change(search, { target: { value } });
      expect(screen.getByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
      expect(screen.queryByRole("link", { name: /review-changes/ })).not.toBeInTheDocument();
      expect(screen.queryByRole("link", { name: /release-notes/ })).not.toBeInTheDocument();
    }
  });

  it("opens the chosen skill with its source and install commands and updates the URL", async () => {
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab: "skills" }, onUrlUpdate });
    await user.click(await screen.findByRole("link", { name: /research-assistant/ }));

    const dialog = await screen.findByRole("dialog", { name: "Details" });
    expect(within(dialog).getByRole("heading", { name: "research-assistant", level: 2 })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "github.com/example/research-tools" })).toHaveAttribute(
      "href",
      "https://github.com/example/research-tools",
    );
    expect(within(dialog).getByText("knowledge")).toBeInTheDocument();
    expect(within(dialog).getByText("Research")).toBeInTheDocument();
    expect(
      within(dialog).getByText("/plugin marketplace add https://gateway.example.com/claude-code/marketplace.json"),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("/plugin install research-assistant@litellm")).toBeInTheDocument();
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(
        expect.objectContaining({ queryString: "?tab=skills&open=research-assistant" }),
      ),
    );
  });

  it("selects the skill named by a shared link instead of the first skill", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, {
      searchParams: { tab: "skills", open: "review-changes" },
    });

    expect(await screen.findByRole("heading", { level: 2, name: "review-changes" })).toBeInTheDocument();
    expect(screen.getByText("/plugin install review-changes@litellm")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 2, name: "release-notes" })).not.toBeInTheDocument();
  });
});

it("filters skills by tag and scan status, searches people, and scopes filters to Skills", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab: "skills" }, onUrlUpdate });
  await screen.findByRole("link", { name: /research-assistant/ });
  await user.click(screen.getByRole("button", { name: "Downloads" }));
  expect(within(screen.getAllByRole("row")[1]).getByRole("link", { name: /review-changes/ })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Downloads" }));
  expect(within(screen.getAllByRole("row")[1]).getByRole("link", { name: /release-notes/ })).toBeInTheDocument();
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by tag" }), "evidence");
  expect(screen.queryByRole("link", { name: /review-changes/ })).not.toBeInTheDocument();
  expect(screen.getByText("writer@example.test")).toBeInTheDocument();
  expect(screen.getByText("reviewer@example.test")).toBeInTheDocument();
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Security scan" }), "Security scan: No");
  expect(screen.queryByRole("link", { name: /research-assistant/ })).not.toBeInTheDocument();
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Security scan" }), "Security scan: Yes");
  fireEvent.change(screen.getByRole("textbox", { name: /Search skills/ }), {
    target: { value: "reviewer@example.test" },
  });
  expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
  await waitFor(() =>
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({
        queryString: expect.stringContaining("skills_filter_tag=evidence"),
      }),
    ),
  );
  await user.click(screen.getByRole("tab", { name: "Model Hub" }));
  expect(screen.queryByRole("combobox", { name: "Filter by tag" })).not.toBeInTheDocument();
});
