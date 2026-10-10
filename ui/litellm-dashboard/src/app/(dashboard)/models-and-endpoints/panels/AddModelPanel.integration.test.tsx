import { useSyncExternalStore } from "react";
import { act, renderWithProviders, screen, waitFor, within } from "../../../../../tests/test-utils";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddModelPanel from "./AddModelPanel";

const modelCreateCall = vi.fn();
const mockPtuEnabled = vi.fn();
const mockAuthorized = vi.fn();

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    modelCreateCall: (accessToken: string, model: unknown) => modelCreateCall(accessToken, model),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "group-a" }] }),
  };
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockAuthorized() }));

vi.mock("@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled", () => ({
  usePtuCostAttributionEnabled: () => mockPtuEnabled(),
}));

const mockUseModelCostMap = vi.fn();

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: (...args: unknown[]) => mockUseModelCostMap(...args),
}));

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => ({ data: { credentials: [] } }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: [] }),
  useInfiniteTeams: () => ({
    data: { pages: [{ teams: [], total: 0, page: 1, page_size: 20, total_pages: 1 }] },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrails", () => ({
  useGuardrails: () => ({ data: { guardrails: [{ guardrail_name: "g-1" }] }, isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/tags/useTags", () => ({
  useTags: () => ({ data: {}, isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/providers/useProviderFields", () => ({
  useProviderFields: () => ({
    data: [
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-4o",
        credential_fields: [
          { key: "api_key", label: "API Key", field_type: "password", required: false },
          { key: "api_base", label: "API Base", field_type: "text", required: false },
        ],
      },
      {
        provider: "Anthropic",
        provider_display_name: "Anthropic",
        litellm_provider: "anthropic",
        default_model_placeholder: "claude-3-opus",
        credential_fields: [{ key: "api_key", label: "API Key", field_type: "password", required: false }],
      },
      {
        provider: "TypeSafe",
        provider_display_name: "TypeSafe",
        litellm_provider: "typesafe",
        default_model_placeholder: "jev-latest",
        credential_fields: [{ key: "api_key", label: "API Key", field_type: "password", required: false }],
      },
    ],
    isLoading: false,
    error: null,
  }),
}));

vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  default: () => <div data-testid="vector-store-selector" />,
}));

type Catalog = Record<string, { litellm_provider: string }>;

const createCatalogFeed = () => {
  let current: Catalog | undefined;
  const listeners = new Set<() => void>();
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  };
  return {
    useData: () => ({ data: useSyncExternalStore(subscribe, () => current) }),
    publish: (next: Catalog) => {
      current = next;
      listeners.forEach((listener) => listener());
    },
  };
};

const lastCreatedModel = () => modelCreateCall.mock.calls.at(-1)?.[1];

const PROXY_ADMIN = {
  token: "t",
  accessToken: "test-access-token",
  userId: "user-1",
  userEmail: "a@b.c",
  userRole: "proxy_admin",
  premiumUser: true,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
};

const alwaysMounted = {
  api_key: undefined,
  api_base: undefined,
  custom_llm_provider: "openai",
  model: "gpt-4o",
};

const advancedOpenExtras = {
  guardrails: undefined,
  kubernetes_pod_discovery: undefined,
  tags: undefined,
  use_in_pass_through: undefined,
  vector_store_ids: undefined,
};

const baseModelInfo = { access_groups: undefined, mode: undefined };

const { api_base: _omitted, ...ALWAYS_MOUNTED_WITHOUT_API_BASE } = alwaysMounted;

const setup = async () => {
  const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
  renderWithProviders(<AddModelPanel />);
  await screen.findByText("Provider");

  const openAdvanced = async () => {
    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Tags");
  };

  const closeAdvanced = async () => {
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByText("Tags")).not.toBeInTheDocument());
  };

  const fillRequired = async (modelName = "gpt-4o") => {
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByText("OpenAI"));
    await user.type(await screen.findByPlaceholderText("gpt-3.5-turbo"), modelName);
  };

  const submit = async () => {
    await user.click(screen.getByTestId("add-model-btn"));
    await waitFor(() => expect(modelCreateCall).toHaveBeenCalled());
  };

  const submitExpectingRejection = async (message: string) => {
    await user.click(screen.getByTestId("add-model-btn"));
    await screen.findByText(message);
  };

  return { user, openAdvanced, closeAdvanced, fillRequired, submit, submitExpectingRejection };
};

describe("AddModelPanel submit payload contract", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseModelCostMap.mockReturnValue({ data: {} });
    mockPtuEnabled.mockReturnValue(false);
    mockAuthorized.mockReturnValue(PROXY_ADMIN);
  });

  it("lists catalog models in the model picker, not entries registered at runtime for deployments", async () => {
    mockUseModelCostMap.mockImplementation((_enabled: boolean, catalogOnly: boolean) => ({
      data: catalogOnly
        ? { "gpt-4o-2024-08-06": { litellm_provider: "openai" } }
        : {
            "gpt-4o-2024-08-06": { litellm_provider: "openai" },
            "openai-gpt-4o-deployment-id": { litellm_provider: "openai" },
          },
    }));
    const { user } = await setup();
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByText("OpenAI"));
    await user.click(await screen.findByPlaceholderText("Select models"));

    expect(await screen.findByText("gpt-4o-2024-08-06")).toBeInTheDocument();
    expect(screen.queryByText("openai-gpt-4o-deployment-id")).not.toBeInTheDocument();
  });

  it("offers the catalog models once the catalog arrives after the provider was picked", async () => {
    const catalog = createCatalogFeed();
    mockUseModelCostMap.mockImplementation(() => catalog.useData());
    const { user } = await setup();
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByText("OpenAI"));
    expect(await screen.findByPlaceholderText("gpt-3.5-turbo")).toBeInTheDocument();

    act(() => catalog.publish({ "gpt-4o-2024-08-06": { litellm_provider: "openai" } }));
    await user.click(await screen.findByPlaceholderText("Select models"));

    expect(await screen.findByText("gpt-4o-2024-08-06")).toBeInTheDocument();
  });

  it("keeps a model name typed before the catalog arrives instead of swapping the field under the user", async () => {
    const catalog = createCatalogFeed();
    mockUseModelCostMap.mockImplementation(() => catalog.useData());
    const { user } = await setup();
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByText("OpenAI"));
    await user.type(await screen.findByPlaceholderText("gpt-3.5-turbo"), "my-fine-tune");

    act(() => catalog.publish({ "gpt-4o-2024-08-06": { litellm_provider: "openai" } }));

    await waitFor(() => expect(screen.getByPlaceholderText("gpt-3.5-turbo")).toHaveValue("my-fine-tune"));
    expect(screen.queryByPlaceholderText("Select models")).not.toBeInTheDocument();
  });

  it("offers the catalog models when a name typed before the catalog arrived was cleared again", async () => {
    const catalog = createCatalogFeed();
    mockUseModelCostMap.mockImplementation(() => catalog.useData());
    const { user } = await setup();
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByText("OpenAI"));
    const typed = await screen.findByPlaceholderText("gpt-3.5-turbo");
    await user.type(typed, "my-fine-tune");
    await user.clear(typed);

    act(() => catalog.publish({ "gpt-4o-2024-08-06": { litellm_provider: "openai" } }));
    await user.click(await screen.findByPlaceholderText("Select models"));

    expect(await screen.findByText("gpt-4o-2024-08-06")).toBeInTheDocument();
  });

  it("swaps the offered models when the provider changes while the catalog stays the same", async () => {
    const catalog = createCatalogFeed();
    catalog.publish({
      "gpt-4o-2024-08-06": { litellm_provider: "openai" },
      "claude-sonnet-4-5": { litellm_provider: "anthropic" },
    });
    mockUseModelCostMap.mockImplementation(() => catalog.useData());
    const { user } = await setup();
    const provider = screen.getByRole("combobox", { name: /provider/i });
    await user.click(provider);
    await user.click(await screen.findByText("OpenAI"));
    await user.clear(provider);
    await user.type(provider, "Anthropic");
    await user.click(await screen.findByText("Anthropic"));
    await user.click(await screen.findByPlaceholderText("Select models"));

    expect(await screen.findByText("claude-sonnet-4-5")).toBeInTheDocument();
    expect(screen.queryByText("gpt-4o-2024-08-06")).not.toBeInTheDocument();
  });

  it("sends only the always-mounted fields while Advanced Settings stays closed", async () => {
    const { fillRequired, submit } = await setup();
    await fillRequired();
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted },
      model_info: { ...baseModelInfo },
    });
  });

  it("registers four more keys as undefined once Advanced Settings opens", async () => {
    const { openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted, ...advancedOpenExtras },
      model_info: { ...baseModelInfo },
    });
  });

  it("merges typed LiteLLM Params into litellm_params", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.type(screen.getByLabelText("LiteLLM Params"), '{{"rpm": 7}');
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted, ...advancedOpenExtras, rpm: 7 },
      model_info: { ...baseModelInfo },
    });
  });

  it("submits Kubernetes pod discovery in litellm_params", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.click(screen.getByRole("switch", { name: "Kubernetes pod discovery" }));
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: {
        ...alwaysMounted,
        ...advancedOpenExtras,
        kubernetes_pod_discovery: true,
      },
      model_info: { ...baseModelInfo },
    });
  });

  it("drops a collapsed section's keys and the value typed into it", async () => {
    const { user, openAdvanced, closeAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.type(screen.getByLabelText("LiteLLM Params"), '{{"rpm": 7}');
    await closeAdvanced();
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted },
      model_info: { ...baseModelInfo },
    });
  });

  it("restores the typed value when the section is expanded again", async () => {
    const { user, openAdvanced, closeAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.type(screen.getByLabelText("LiteLLM Params"), '{{"rpm": 7}');
    await closeAdvanced();
    await openAdvanced();
    expect(screen.getByLabelText("LiteLLM Params")).toHaveValue('{"rpm": 7}');

    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted, ...advancedOpenExtras, rpm: 7 },
      model_info: { ...baseModelInfo },
    });
  });

  it("converts per-million pricing to per-token and falls back to input cost for cache reads", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.click(screen.getByRole("switch", { name: "Custom Pricing" }));
    await user.type(await screen.findByLabelText("Input Cost (per 1M tokens)"), "3");
    await user.type(screen.getByLabelText("Output Cost (per 1M tokens)"), "9");
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: {
        ...alwaysMounted,
        ...advancedOpenExtras,
        input_cost_per_token: 0.000003,
        output_cost_per_token: 0.000009,
        cache_read_input_token_cost: 0.000003,
      },
      model_info: { ...baseModelInfo },
    });
  });

  it("sends the seeded injection point when cache control is switched on", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.click(screen.getByRole("switch", { name: "Cache Control Injection Points" }));
    await screen.findByText("Add Injection Point");
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: {
        ...alwaysMounted,
        ...advancedOpenExtras,
        cache_control_injection_points: [{ location: "message" }],
      },
      model_info: { ...baseModelInfo },
    });
  });

  it("carries a role picked inside the injection point editor, with the index kept a string", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.click(screen.getByRole("switch", { name: "Cache Control Injection Points" }));
    await screen.findByText("Add Injection Point");
    await user.click(screen.getByText("Select a role"));
    await user.click(await screen.findByText("System"));
    await user.type(screen.getByPlaceholderText("Optional"), "3");
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: {
        ...alwaysMounted,
        ...advancedOpenExtras,
        cache_control_injection_points: [{ location: "message", role: "system", index: "3" }],
      },
      model_info: { ...baseModelInfo },
    });
  });

  it("mounts team_id only once the Team-BYOK switch is on", async () => {
    const { user, fillRequired, submit } = await setup();
    await fillRequired();
    await user.click(screen.getByRole("switch", { name: "Team-BYOK Model" }));
    await screen.findByText("Select Team");
    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted },
      model_info: { ...baseModelInfo, team_id: undefined },
    });
  });
});

describe("AddModelPanel empty-string skip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseModelCostMap.mockReturnValue({ data: {} });
    mockPtuEnabled.mockReturnValue(false);
    mockAuthorized.mockReturnValue(PROXY_ADMIN);
  });

  it("sends a typed api_base, so the binding behind the next case is known to be live", async () => {
    const { user, fillRequired, submit } = await setup();
    await fillRequired();
    await user.type(screen.getByLabelText("API Base"), "https://example.test");
    await submit();

    expect(lastCreatedModel().litellm_params).toStrictEqual({
      ...alwaysMounted,
      api_base: "https://example.test",
    });
  });

  it("omits api_base entirely once it is cleared, rather than sending an empty string", async () => {
    const { user, fillRequired, submit } = await setup();
    await fillRequired();
    const apiBase = screen.getByLabelText("API Base");
    await user.type(apiBase, "https://example.test");
    await user.clear(apiBase);
    await submit();

    const params = lastCreatedModel().litellm_params;
    expect(params).not.toHaveProperty("api_base");
    expect(params).toStrictEqual(ALWAYS_MOUNTED_WITHOUT_API_BASE);
  });
});

describe("AddModelPanel validation gates", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPtuEnabled.mockReturnValue(true);
    mockAuthorized.mockReturnValue(PROXY_ADMIN);
  });

  it("blocks the submit when a PTU count carries no effective-from date", async () => {
    const { user, openAdvanced, fillRequired, submitExpectingRejection } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.type(screen.getByLabelText("PTU Count"), "15");
    await user.type(screen.getByLabelText("Calculated Cost per PTU / Hour (USD)"), "2");
    await submitExpectingRejection("PTU Effective From is required when PTU Count is set");

    expect(modelCreateCall).not.toHaveBeenCalled();
  });

  it("hides the PTU fields entirely when the capability is off", async () => {
    mockPtuEnabled.mockReturnValue(false);
    const { openAdvanced, fillRequired } = await setup();
    await fillRequired();
    await openAdvanced();

    expect(screen.queryByLabelText("PTU Count")).not.toBeInTheDocument();
  });

  it("requires a model before anything is sent", async () => {
    const { user, submitExpectingRejection } = await setup();
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByText("OpenAI"));
    await submitExpectingRejection("Please enter at least one model.");

    expect(modelCreateCall).not.toHaveBeenCalled();
  });

  it("blocks the submit when LiteLLM Params is not valid JSON", async () => {
    mockPtuEnabled.mockReturnValue(false);
    const { user, openAdvanced, fillRequired, submitExpectingRejection } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.type(screen.getByLabelText("LiteLLM Params"), "rpm: 7");
    await submitExpectingRejection("Please enter valid JSON");

    expect(modelCreateCall).not.toHaveBeenCalled();
  });
});

describe("AddModelPanel behaviours the removed Advanced Settings form instance never drove", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseModelCostMap.mockReturnValue({ data: {} });
    mockPtuEnabled.mockReturnValue(false);
    mockAuthorized.mockReturnValue(PROXY_ADMIN);
  });

  it("leaves LiteLLM Params untouched when pass through routes is switched on", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.click(screen.getByRole("switch", { name: "Use in pass through routes" }));
    expect(screen.getByLabelText("LiteLLM Params")).toHaveValue("");

    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: { ...alwaysMounted, ...advancedOpenExtras, use_in_pass_through: true },
      model_info: { ...baseModelInfo },
    });
  });

  it("keeps a typed cost when custom pricing is switched off and back on", async () => {
    const { user, openAdvanced, fillRequired, submit } = await setup();
    await fillRequired();
    await openAdvanced();
    await user.click(screen.getByRole("switch", { name: "Custom Pricing" }));
    await user.type(await screen.findByLabelText("Input Cost (per 1M tokens)"), "3");
    await user.click(screen.getByRole("switch", { name: "Custom Pricing" }));
    await waitFor(() => expect(screen.queryByLabelText("Input Cost (per 1M tokens)")).not.toBeInTheDocument());
    await user.click(screen.getByRole("switch", { name: "Custom Pricing" }));
    expect(await screen.findByLabelText("Input Cost (per 1M tokens)")).toHaveValue("3");

    await submit();

    expect(lastCreatedModel()).toStrictEqual({
      model_name: "gpt-4o",
      litellm_params: {
        ...alwaysMounted,
        ...advancedOpenExtras,
        input_cost_per_token: 0.000003,
        cache_read_input_token_cost: 0.000003,
      },
      model_info: { ...baseModelInfo },
    });
  });
});

describe("AddModelPanel decision models", () => {
  const DECISION_COST_MAP = {
    "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
    "gpt-4o-2024-08-06": { litellm_provider: "openai", mode: "chat" },
    "gpt-6-luna": { litellm_provider: "openai", mode: "chat", supported_endpoints: ["/v1/decisions"] },
    "claude-sonnet-4-5": { litellm_provider: "anthropic", mode: "chat" },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseModelCostMap.mockReturnValue({ data: DECISION_COST_MAP });
    mockPtuEnabled.mockReturnValue(false);
    mockAuthorized.mockReturnValue(PROXY_ADMIN);
  });

  const decisionNotice = () => screen.queryByRole("note", { name: "Decision model notice" });

  it("finds a provider by the name of one of its decision models", async () => {
    const { user } = await setup();
    await user.type(screen.getByRole("combobox", { name: /provider/i }), "jev");

    expect(await screen.findByRole("option", { name: /TypeSafe/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /Anthropic/ })).not.toBeInTheDocument();
  });

  it("points a decision-only provider at the decision routes, the docs, and the playground", async () => {
    const { user } = await setup();
    await user.type(screen.getByRole("combobox", { name: /provider/i }), "TypeSafe");
    await user.click(await screen.findByRole("option", { name: /TypeSafe/ }));

    const note = await screen.findByRole("note", { name: "Decision model notice" });
    expect(note).toHaveTextContent("/v1/decisions");
    expect(within(note).getByRole("link", { name: "How to call decision models" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/decisions",
    );
    expect(within(note).getByRole("link", { name: "test it in the System One playground" })).toHaveAttribute(
      "href",
      "/ui/playground?tab=system-one",
    );
  });

  it("shows the notice on a chat provider only once one of its decision models is picked", async () => {
    const { user } = await setup();
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }));
    await user.click(await screen.findByPlaceholderText("Select models"));
    await user.click(await screen.findByText("gpt-4o-2024-08-06"));
    expect(decisionNotice()).not.toBeInTheDocument();

    await user.click(await screen.findByText("gpt-6-luna"));

    const note = await screen.findByRole("note", { name: "Decision model notice" });
    expect(note).toHaveTextContent("/v1/decisions");
    expect(note).not.toHaveTextContent("/chat/completions");
  });
});
