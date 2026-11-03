import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, testQueryClient, waitFor, within } from "@/../tests/test-utils";
import type { ModelHubData } from "@/components/AIHub/ModelHubTableColumns";

import ChatModelsPage from "./page";

const { apiGetMock, modelHubCallMock, shellState } = vi.hoisted(() => ({
  apiGetMock: vi.fn(),
  modelHubCallMock: vi.fn(),
  shellState: { accessToken: "chat-session-token", userRole: "proxy_admin", premiumUser: false },
}));

vi.mock("@/contexts/ChatShellContext", () => ({
  useChatShell: () => shellState,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: {} }, isLoading: false }),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  const hubInfo = {
    docs_title: "Gateway models",
    custom_docs_description: null,
    litellm_version: "1.0.0",
    useful_links: {},
  };
  return {
    ...actual,
    apiClient: { ...actual.apiClient, get: apiGetMock },
    modelHubCall: modelHubCallMock,
    getConfigFieldSetting: vi.fn().mockResolvedValue({ field_value: true }),
    getUiConfig: vi.fn().mockResolvedValue({}),
    getProxyBaseUrl: () => "https://gateway.example.com",
    getGatewayBaseUrl: () => "https://gateway.example.com",
    getPublicModelHubInfo: vi.fn().mockResolvedValue(hubInfo),
    getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
    fetchMCPServers: vi.fn().mockResolvedValue([]),
    getClaudeCodePluginsList: vi.fn().mockResolvedValue({ plugins: [] }),
    agentHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    mcpHubPublicServersCall: vi.fn().mockResolvedValue([]),
    skillHubPublicCall: vi.fn().mockResolvedValue({ plugins: [] }),
  };
});

const PRIVATE_MODEL = {
  model_group: "research-private",
  description: "An internal model for reviewing research reports.",
  providers: ["openai"],
  mode: "chat",
  max_input_tokens: 128_000,
  max_output_tokens: 4_096,
  input_cost_per_token: 0.000002,
  output_cost_per_token: 0.000008,
  supports_function_calling: true,
  supports_parallel_function_calling: false,
  supports_vision: true,
  is_public_model_group: false,
} satisfies ModelHubData;

const IMAGE_MODEL: ModelHubData = {
  ...PRIVATE_MODEL,
  model_group: "image-studio",
  description: "Create illustrations for presentations.",
  mode: "image_generation",
};

const PUBLISHED_MODEL = {
  ...PRIVATE_MODEL,
  model_group: "published-chat",
  description: "The shared model published for gateway users.",
  is_public_model_group: true,
} satisfies ModelHubData;

const PUBLIC_MODELS_PATH = "/public/v1/model_hub";
const FACETS: Record<string, string[]> = {
  [`${PUBLIC_MODELS_PATH}/providers`]: ["openai"],
  [`${PUBLIC_MODELS_PATH}/modes`]: ["chat"],
  [`${PUBLIC_MODELS_PATH}/features`]: ["function_calling", "vision"],
};

describe("Chat Models page", () => {
  beforeEach(() => {
    shellState.userRole = "proxy_admin";
    modelHubCallMock.mockResolvedValue({ data: [PRIVATE_MODEL, IMAGE_MODEL] });
    apiGetMock.mockImplementation(async (path: string) => {
      if (path in FACETS) {
        return {
          data: FACETS[path],
          meta: { page: 1, page_size: 100, has_more: false },
          links: { self: path, prev: null, next: null },
        };
      }
      if (path === PUBLIC_MODELS_PATH) {
        return {
          data: [PUBLISHED_MODEL],
          meta: { total_count: 1, page: 1, page_size: 50, total_pages: 1 },
          links: {
            self: path,
            first: path,
            prev: null,
            next: null,
            last: path,
          },
        };
      }
      throw new Error(`Unexpected catalog request: ${path}`);
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input) !== "https://gateway.example.com/get/ui_theme_settings") {
          throw new Error(`Unexpected fetch: ${String(input)}`);
        }
        return new Response(JSON.stringify({ values: {} }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("should render gateway models for admins and let them search and inspect model details", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ChatModelsPage />);

    expect(screen.getByRole("heading", { name: "Models", level: 1 })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /^research-private/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^image-studio/ })).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("Search model names..."), "research");
    await waitFor(() => expect(screen.queryByRole("button", { name: /^image-studio/ })).not.toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /^research-private/ }));

    const details = await screen.findByRole("dialog", { name: PRIVATE_MODEL.model_group });
    expect(within(details).getByText(PRIVATE_MODEL.description)).toBeInTheDocument();
    expect(within(details).getByText("128,000")).toBeInTheDocument();
    expect(within(details).getByText("$2.00")).toBeInTheDocument();
    expect(within(details).getByText("Vision")).toBeInTheDocument();
    expect(within(details).getByText("Function Calling")).toBeInTheDocument();
  });

  it("should show regular users the published catalog and its model capabilities", async () => {
    shellState.userRole = "internal_user";
    const user = userEvent.setup();
    renderWithProviders(<ChatModelsPage />);

    const publishedModel = await screen.findByRole("button", { name: /^published-chat/ });
    expect(screen.queryByRole("button", { name: /^research-private/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^image-studio/ })).not.toBeInTheDocument();
    await user.click(publishedModel);

    const details = await screen.findByRole("dialog", { name: PUBLISHED_MODEL.model_group });
    expect(within(details).getByText(PUBLISHED_MODEL.description)).toBeInTheDocument();
    expect(within(details).getByText("128,000")).toBeInTheDocument();
    expect(within(details).getByText("$2.0000")).toBeInTheDocument();
    expect(within(details).getByText("Vision")).toBeInTheDocument();
  });
});
