import { fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { agentHubPublicModelsCall, mcpHubPublicServersCall, skillHubPublicCall } from "@/components/networking";

import { renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import PublicAiHub from "./PublicAiHub";

const { apiGetMock, navigation, uiSettings, replaceMock, hubInfo } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
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

vi.mock("@/components/navbar", () => ({ default: () => <nav aria-label="Top bar" /> }));

vi.mock("@/lib/telemetry/uiEvents", () => ({ recordUiEvent: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: uiSettings, isLoading: false }),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    apiClient: { ...actual.apiClient, get: apiGetMock },
    getUiConfig: vi.fn().mockResolvedValue({}),
    getProxyBaseUrl: vi.fn(() => "https://gateway.example.com"),
    getPublicModelHubInfo: vi.fn().mockResolvedValue(hubInfo),
    agentHubPublicModelsCall: vi.fn().mockResolvedValue([
      {
        name: "research-assistant",
        description: "Answers research questions",
        version: "1.4.0",
        url: "https://gateway.example.com/a2a/research",
        protocolVersion: "0.3.0",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        skills: [],
      },
    ]),
    mcpHubPublicServersCall: vi.fn().mockResolvedValue([]),
    skillHubPublicCall: vi.fn().mockResolvedValue({ plugins: [] }),
  };
});

const MODEL_HUB_PATH = "/public/v1/model_hub";

const MODELS = [
  {
    model_group: "claude-sonnet-4-5",
    providers: ["anthropic"],
    mode: "chat",
    input_cost_per_token: 0.000003,
    description: "Best for agentic coding",
  },
  { model_group: "gpt-5", providers: ["openai"], mode: "chat", description: "General purpose" },
];

const page = (data: unknown[]) => ({
  data,
  meta: { total_count: data.length, page: 1, page_size: 50, total_pages: 1 },
  links: { self: MODEL_HUB_PATH, first: MODEL_HUB_PATH, prev: null, next: null, last: MODEL_HUB_PATH },
});

const OFF_PAGE_MODEL = { model_group: "o1-pro", providers: ["openai"], mode: "chat", description: "Deep reasoning" };

const respondWithModels = () =>
  apiGetMock.mockImplementation((path: string, options?: { query?: Record<string, unknown> }) => {
    if (path !== MODEL_HUB_PATH) return Promise.resolve({ ...page([]), meta: { page: 1, page_size: 100 } });
    return Promise.resolve(options?.query?.q === "o1-pro" ? page([OFF_PAGE_MODEL]) : page(MODELS));
  });

const tokenExpiringIn = (seconds: number) => {
  const encode = (value: object) => btoa(JSON.stringify(value)).replace(/=+$/, "");
  return `${encode({ alg: "HS256" })}.${encode({ exp: Math.floor(Date.now() / 1000) + seconds })}.signature`;
};

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    }),
  });
});

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
  it("counts every hub and shows the first model's details", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByRole("heading", { level: 2, name: "claude-sonnet-4-5" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /^Models\s*2$/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /^Agents\s*1$/ })).toBeInTheDocument();
    expect(screen.getByText("Best for agentic coding")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Base URL" })).toHaveValue("https://gateway.example.com");
  });

  it("opens a model from the list and records it in the URL", async () => {
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { onUrlUpdate });

    await user.click(await screen.findByRole("link", { name: /gpt-5/ }));

    expect(await screen.findByRole("heading", { level: 2, name: "gpt-5" })).toBeInTheDocument();
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenCalledWith(expect.objectContaining({ queryString: "?open=gpt-5" })),
    );
  });

  it("explains an empty search and clears it", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab: "agents" } });

    fireEvent.change(await screen.findByRole("textbox", { name: "Search agents" }), { target: { value: "billing" } });

    expect(await screen.findByText('No agents match "billing"')).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
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

  it("keeps agents reachable when only the models call fails", async () => {
    const user = userEvent.setup();
    apiGetMock.mockRejectedValue(new Error("No models configured"));
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByText("Couldn't load models")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Search models" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: /^Agents\s*1$/ }));
    expect(await screen.findByRole("link", { name: /research-assistant/ })).toBeInTheDocument();
  });

  it("says the hub is unavailable when every hub call fails", async () => {
    apiGetMock.mockRejectedValue(new Error("Not found"));
    vi.mocked(agentHubPublicModelsCall).mockRejectedValueOnce(new Error("Not found"));
    vi.mocked(mcpHubPublicServersCall).mockRejectedValueOnce(new Error("Not found"));
    vi.mocked(skillHubPublicCall).mockRejectedValueOnce(new Error("Not found"));
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByText("This AI Hub isn't available")).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  it("keeps the page's other query params in each row's link", async () => {
    navigation.params = new URLSearchParams("tab=agents&key=abc");
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { tab: "agents", key: "abc" } });

    expect(await screen.findByRole("link", { name: /research-assistant/ })).toHaveAttribute(
      "href",
      "?tab=agents&key=abc&open=research-assistant",
    );
  });

  it("clears the open item when switching tabs", async () => {
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "gpt-5" }, onUrlUpdate });

    expect(await screen.findByRole("heading", { level: 2, name: "gpt-5" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: /^Agents/ }));

    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: "?tab=agents" })),
    );
  });

  it("opens a shared link to a model that is not on the first page", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "o1-pro" } });

    expect(await screen.findByRole("heading", { level: 2, name: "o1-pro" })).toBeInTheDocument();
    expect(screen.getByText("Deep reasoning")).toBeInTheDocument();
  });

  it("says when a shared link names something that is no longer listed", async () => {
    renderWithProviders(<PublicAiHub accessToken={null} />, { searchParams: { open: "retired-model" } });

    expect(await screen.findByText("\u201cretired-model\u201d isn't listed here")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 2, name: "claude-sonnet-4-5" })).not.toBeInTheDocument();
  });

  it("sends visitors without a session to login when the admin requires it", async () => {
    uiSettings.values = { require_auth_for_public_ai_hub: true };
    renderWithProviders(<PublicAiHub accessToken={null} />);

    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith(expect.stringContaining("login")));
  });

  it("keeps a signed-in visitor on the hub when the admin requires sign-in", async () => {
    uiSettings.values = { require_auth_for_public_ai_hub: true };
    document.cookie = `token=${tokenExpiringIn(3600)}; path=/`;
    renderWithProviders(<PublicAiHub accessToken={null} />);

    expect(await screen.findByRole("heading", { level: 2, name: "claude-sonnet-4-5" })).toBeInTheDocument();
    expect(replaceMock).not.toHaveBeenCalled();
  });
});
