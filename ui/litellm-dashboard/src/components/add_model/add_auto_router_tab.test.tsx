import { renderWithProviders, screen, waitFor, within, fireEvent, testQueryClient } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import NotificationManager from "../molecules/notifications_manager";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { getMissingTiersError } from "./build_complexity_router_config";
import { ModelGroup } from "@/components/llm_calls/fetch_models";

// Every model referenced by both bundled family presets. A caller holding all of these can select
// either preset; dropping any one greys out the preset that names it.
const ALL_FAMILY_MODELS: ModelGroup[] = [
  { model_group: "claude-haiku-4-5", mode: "chat" },
  { model_group: "claude-sonnet-4-5", mode: "chat" },
  { model_group: "claude-opus-5", mode: "chat" },
  { model_group: "gpt-5-nano", mode: "chat" },
  { model_group: "gpt-5-mini", mode: "chat" },
  { model_group: "gpt-5", mode: "chat" },
  { model_group: "o3", mode: "chat" },
];

const openTemplateDropdown = (): void => {
  fireEvent.mouseDown(screen.getByTestId("template-selector").querySelector(".ant-select-selector")!);
};

// The rendered antd option whose text starts with a preset label. Matching on text (not role +
// accessible name) sidesteps antd's list re-rendering options in place on every state change.
const optionByLabel = (label: string): HTMLElement | undefined =>
  Array.from(document.querySelectorAll<HTMLElement>(".ant-select-item-option")).find((el) =>
    el.textContent?.startsWith(label),
  );

const isOptionDisabled = (option: HTMLElement): boolean => option.classList.contains("ant-select-item-option-disabled");

const { mockFetchAvailableModels } = vi.hoisted(() => ({ mockFetchAvailableModels: vi.fn() }));

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: mockFetchAvailableModels,
}));

vi.mock("./handle_add_auto_router_submit", () => ({
  handleAddAutoRouterSubmit: vi.fn(),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: { fromBackend: vi.fn() },
}));

// Kept real by default so the "mandatory field" test still sees genuine tier validation; one
// test overrides it to reach the submit path without driving four tier selects.
vi.mock("./build_complexity_router_config", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./build_complexity_router_config")>();
  return { ...actual, getMissingTiersError: vi.fn(actual.getMissingTiersError) };
});

// A real TeamDropdown fetches teams and renders an antd Select; the wiring under test is
// whether team_id is registered, validated and forwarded, so a plain control stands in.
vi.mock("../common_components/team_dropdown", () => ({
  default: ({ value, onChange }: { value?: string; onChange?: (next: string) => void }) => (
    <select
      data-testid="team-dropdown"
      value={value ?? ""}
      onChange={(event) => onChange?.(event.target.value)}
      aria-label="Select Team"
    >
      <option value="">none</option>
      <option value="team-1">team-1</option>
    </select>
  ),
}));

const Harness = () => <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Admin" />;

describe("AddAutoRouterTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // testQueryClient is a shared singleton with staleTime: Infinity, so cached model lists would
    // otherwise bleed across tests (a later test reusing accessToken="token" would read an earlier
    // test's data instead of its own mock).
    testQueryClient.clear();
    mockFetchAvailableModels.mockResolvedValue([]);
  });

  // Nothing is filled in, so there is nothing to submit. The button reports that itself instead of
  // accepting a click and answering with a toast.
  it("offers no submit at all until every tier has a model", async () => {
    renderWithProviders(<Harness />);

    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
  });

  it("still flags the router name once the config no longer blocks the submit", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);
    renderWithProviders(<Harness />);

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Auto router name is required")).toBeInTheDocument();
    expect(NotificationManager.fromBackend).toHaveBeenCalledWith("Please enter an Auto Router Name");
  });

  it("offers no team selector to a proxy admin, who may create an unscoped router", () => {
    renderWithProviders(<Harness />);

    expect(screen.queryByTestId("team-dropdown")).not.toBeInTheDocument();
  });

  it("requires a team admin to pick a team", async () => {
    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    expect(screen.getByTestId("team-dropdown")).toBeInTheDocument();
    expect(screen.getByText("Select Team")).toBeInTheDocument();
  });

  // POST /model/new 403s an unscoped create from a non-proxy-admin, so a selected team that
  // never reaches the payload is indistinguishable from having no selector at all. The value
  // has to survive form.validateFields, which only returns the fields it is asked for.
  it("carries the selected team through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    await user.type(screen.getByPlaceholderText(/smart_router/i), "team-scoped-router");
    await user.selectOptions(screen.getByTestId("team-dropdown"), "team-1");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({ team_id: "team-1" });
  });

  // LIT-5133: "Add keyword rule" seeds a row with no keywords, and the semantic toggle that used
  // to be the only thing checking them is off by default. The row was dropped on the way to the
  // payload, so the create succeeded and the caller's rule was gone with nothing said about it.
  it("takes the submit away while a keyword rule is left empty", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));

    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
    // The row says so on its own; there is no failed submit left to surface it.
    expect(await screen.findByText("At least one keyword is required")).toBeInTheDocument();
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });

  it("gives the submit back once that keyword rule is filled", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();

    await user.type(
      within(screen.getByText("Keywords 1").closest("div") as HTMLElement).getByRole("combobox"),
      "invoice{enter}",
    );

    expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled();
    expect(screen.queryByText("At least one keyword is required")).not.toBeInTheDocument();
  });

  it("marks only the offending keyword row, leaving a filled one alone", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    await user.type(
      within(screen.getByText("Keywords 1").closest("div") as HTMLElement).getByRole("combobox"),
      "invoice{enter}",
    );
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));

    expect(await screen.findAllByText("At least one keyword is required")).toHaveLength(1);
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
  });

  it("creates the router once that keyword rule is filled in", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    const keywordsField = screen.getByText("Keywords 1").closest("div") as HTMLElement;
    await user.type(within(keywordsField).getByRole("combobox"), "invoice{enter}");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
      complexity_router_config: { keyword_tier_rules: [{ keywords: ["invoice"], tier: "COMPLEX" }] },
    });
  });

  it("blocks the submit when a team admin has not picked a team", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    await user.type(screen.getByPlaceholderText(/smart_router/i), "team-scoped-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Please select a team to continue")).toBeInTheDocument();
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });

  it("defaults a new router to session affinity off, matching the backend field default", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    await user.click(screen.getByText("Advanced: Session Affinity"));
    expect(await screen.findByRole("switch", { name: "Pin a session to its first model" })).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: false,
    });
  });

  it("carries session affinity turned on through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    await user.click(screen.getByText("Advanced: Session Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to its first model" }));

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: true,
    });
  });

  describe("template presets", () => {
    // Opens the dropdown once, then waits out the useQuery load: an open antd Select re-renders its
    // already-mounted options in place as state changes, so polling only re-reads the DOM here.
    // Re-firing the open/close mousedown on every poll (calling openTemplateDropdown inside the
    // waitFor callback) fights the dropdown's own open/close animation and hangs the test.
    const waitForPresetEnabled = async (label: string) => {
      openTemplateDropdown();
      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel(label)!)).toBe(false);
      });
    };

    it("disables every preset while the model list is loading", async () => {
      let resolveModels: (models: ModelGroup[]) => void = () => {};
      mockFetchAvailableModels.mockImplementation(
        () =>
          new Promise<ModelGroup[]>((resolve) => {
            resolveModels = resolve;
          }),
      );

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      const anthropicOption = optionByLabel("Anthropic Family")!;
      expect(isOptionDisabled(anthropicOption)).toBe(true);
      expect(anthropicOption.textContent).toContain("Checking model availability");

      // The dropdown is already open from above; polling re-reads its options in place rather than
      // reopening (openTemplateDropdown toggles, so a second call here would close it instead).
      resolveModels(ALL_FAMILY_MODELS);
      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
    });

    it("disables every preset and offers a retry when the model list fails to load", async () => {
      mockFetchAvailableModels.mockRejectedValue(new Error("network error"));

      renderWithProviders(<Harness />);

      expect(await screen.findByText("Could not load available models.")).toBeInTheDocument();
      openTemplateDropdown();
      const anthropicOption = optionByLabel("Anthropic Family")!;
      expect(isOptionDisabled(anthropicOption)).toBe(true);
      expect(anthropicOption.textContent).toContain("Cannot verify these models are available");
    });

    it("disables a preset missing one of its models, naming the missing model", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS.filter((m) => m.model_group !== "claude-opus-5"));

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await waitFor(() => {
        expect(optionByLabel("Anthropic Family")!.textContent).toContain("Missing: claude-opus-5");
      });
      expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(true);
    });

    it("enables a preset once every model it needs is available", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

      renderWithProviders(<Harness />);

      await waitForPresetEnabled("Anthropic Family");
      await waitForPresetEnabled("OpenAI Family");
    });

    it("collapses detailed configuration and shows a tier summary once a preset is applied", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");

      fireEvent.click(optionByLabel("Anthropic Family")!);

      expect(screen.queryByText("Advanced: Keyword/Semantic Matching")).not.toBeInTheDocument();
      expect(
        screen.getByText(
          "Simple: claude-haiku-4-5 · Medium: claude-sonnet-4-5 · Complex: claude-opus-5 · Reasoning: claude-opus-5",
        ),
      ).toBeInTheDocument();
    });

    it("expands detailed configuration when Custom Configuration is chosen", () => {
      renderWithProviders(<Harness />);
      openTemplateDropdown();

      fireEvent.click(optionByLabel("Custom Configuration")!);

      expect(screen.getByText("Advanced: Keyword/Semantic Matching")).toBeInTheDocument();
    });

    it("lets a caller manually re-expand a detailed configuration a preset just collapsed", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      fireEvent.click(optionByLabel("Anthropic Family")!);
      expect(screen.queryByText("Advanced: Keyword/Semantic Matching")).not.toBeInTheDocument();

      fireEvent.click(screen.getByTestId("detailed-configuration-toggle"));

      expect(screen.getByText("Advanced: Keyword/Semantic Matching")).toBeInTheDocument();
    });

    // This is the regression test for the whole feature: if handlePresetChange stopped prefilling
    // complexityRouterConfig, the real (unmocked here) getMissingTiersError would block the submit
    // and handleAddAutoRouterSubmit would never be called.
    it("carries a selected preset's tiers through to the create payload", async () => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      fireEvent.click(optionByLabel("Anthropic Family")!);

      await user.type(screen.getByPlaceholderText(/smart_router/i), "anthropic-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        auto_router_default_model: "claude-sonnet-4-5",
        complexity_router_config: {
          tiers: {
            SIMPLE: ["claude-haiku-4-5"],
            MEDIUM: ["claude-sonnet-4-5"],
            COMPLEX: ["claude-opus-5"],
            REASONING: ["claude-opus-5"],
          },
        },
      });
    });
  });
});
