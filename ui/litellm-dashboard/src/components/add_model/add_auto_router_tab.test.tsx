import { renderWithProviders, screen, waitFor, within, fireEvent, testQueryClient } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import { toast } from "@/lib/toast";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { getMissingTiersError } from "./build_complexity_router_config";
import { getSubmitBlockedReason } from "./add_auto_router_tab";
import { buildModelAvailability } from "@/lib/autorouter_presets";
import { testAutoRouterRouting } from "../networking";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { getAllPresets, getPresetByKey, getRequiredModelsInPreset } from "@/lib/autorouter_presets";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

const ANTHROPIC_PRESET = getPresetByKey("anthropic_family")!;
const ANTHROPIC_TIERS = ANTHROPIC_PRESET.complexity_router_config.tiers;

// Every model referenced by the bundled family presets, derived from the presets themselves so
// that renaming a preset's models in autorouter_presets.json does not red these tests. A caller
// holding all of these can select either preset; dropping any one greys out the preset that
// names it.
const ALL_FAMILY_MODELS: ModelGroup[] = [
  ...new Set(getAllPresets().flatMap((preset) => [...getRequiredModelsInPreset(preset)])),
].map((model_group) => ({ model_group, mode: "chat" }));

const ANTHROPIC_ONLY_MODEL = ANTHROPIC_TIERS.COMPLEX[0];

const openTemplateDropdown = (): void => {
  fireEvent.click(screen.getByTestId("template-selector"));
};

// Detailed Configuration is collapsed by default, so any test reaching into it (a tier select, an
// "Advanced: ..." sub-section) has to open it first.
const expandDetailedConfiguration = (): void => {
  fireEvent.click(screen.getByTestId("detailed-configuration-toggle"));
};

const visibleOptions = (): HTMLElement[] => screen.queryAllByRole("option");

const optionByLabel = (label: string): HTMLElement | undefined =>
  visibleOptions().find((el) => el.textContent?.startsWith(label));

const isOptionDisabled = (option: HTMLElement): boolean => option.getAttribute("aria-disabled") === "true";

const selectTemplate = async (label: string): Promise<void> => {
  await userEvent.click(optionByLabel(label)!);
};

// Opens the dropdown only when it is closed, since openTemplateDropdown toggles: waiting on a
// second preset in the same test would otherwise close the list out from under the poll.
const waitForPresetEnabled = async (label: string) => {
  if (visibleOptions().length === 0) openTemplateDropdown();
  await waitFor(() => {
    expect(isOptionDisabled(optionByLabel(label)!)).toBe(false);
  });
};

// The keyword field is a combobox that offers whatever is typed as a "Create ..." entry, so a
// keyword only lands on the rule once that entry is picked.
const addKeyword = async (user: ReturnType<typeof userEvent.setup>, field: HTMLElement, keyword: string) => {
  await user.type(within(field).getByRole("combobox"), keyword);
  await user.click(await screen.findByText(`Create "${keyword}"`));
};

const { mockFetchAvailableModels, mockFetchAllModelDeployments } = vi.hoisted(() => ({
  mockFetchAvailableModels: vi.fn(),
  mockFetchAllModelDeployments: vi.fn(),
}));

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  testAutoRouterRouting: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: mockFetchAvailableModels,
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/app/(dashboard)/hooks/models/useModels")>();
  return { ...actual, fetchAllModelDeployments: mockFetchAllModelDeployments };
});

vi.mock("./handle_add_auto_router_submit", () => ({
  handleAddAutoRouterSubmit: vi.fn(),
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
    mockFetchAllModelDeployments.mockResolvedValue([]);
  });

  // Detailed Configuration starts collapsed so the modal opens onto just Name + Template; a caller
  // opts into the full tier/classifier form rather than always seeing it up front.
  it("keeps Detailed Configuration collapsed until a caller opens it", () => {
    renderWithProviders(<Harness />);

    expect(screen.queryByText("Complexity Tier Configuration")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("detailed-configuration-toggle"));

    expect(screen.getByText("Complexity Tier Configuration")).toBeInTheDocument();
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
    expect(toast.fromError).toHaveBeenCalledWith("Please enter an Auto Router Name");
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
    expandDetailedConfiguration();
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
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();

    await addKeyword(user, screen.getByText("Keywords 1").closest("div") as HTMLElement, "invoice");

    expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled();
    expect(screen.queryByText("At least one keyword is required")).not.toBeInTheDocument();
  });

  it("marks only the offending keyword row, leaving a filled one alone", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    await addKeyword(user, screen.getByText("Keywords 1").closest("div") as HTMLElement, "invoice");
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));

    expect(await screen.findAllByText("At least one keyword is required")).toHaveLength(1);
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
  });

  it("creates the router once that keyword rule is filled in", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    const keywordsField = screen.getByText("Keywords 1").closest("div") as HTMLElement;
    await addKeyword(user, keywordsField, "invoice");
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
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Affinity"));
    expect(await screen.findByRole("switch", { name: "Pin a session to its first model" })).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: false,
    });
  });

  // The scalar floor is the one scorer knob with no group dict behind it, so its wiring into the create
  // payload is only proven end to end. 0 is the case a truthy check would silently drop.
  it("carries a reasoning override floor of 0 through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "override-floor-router");
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Classification Method"));
    await user.click(await screen.findByText("Advanced scoring"));
    fireEvent.change(await screen.findByLabelText("Minimum score"), { target: { value: "0" } });

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      reasoning_override_min_score: 0,
    });
  });

  it("carries session affinity turned on through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to its first model" }));

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: true,
    });
  });

  it("defaults a new router to deployment affinity on, matching the backend field default", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Affinity"));
    expect(
      await screen.findByRole("switch", { name: "Pin a session to one deployment per model group" }),
    ).toBeChecked();

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      deployment_affinity: true,
    });
  });

  it("carries deployment affinity turned off through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    expandDetailedConfiguration();
    await user.click(screen.getByText("Advanced: Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to one deployment per model group" }));

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      deployment_affinity: false,
    });
  });

  // Custom is the escape hatch, not the headline choice, so it's listed after every bundled preset
  // rather than first.
  it("lists Custom Configuration after the bundled presets", () => {
    renderWithProviders(<Harness />);
    openTemplateDropdown();

    const labels = visibleOptions().map((option) => option.querySelector(".font-medium")?.textContent);

    expect(labels).toEqual(["Anthropic Family", "Gemini Family", "Lite", "OpenAI Family", "Custom Configuration"]);
  });

  describe("routing test", () => {
    it("offers no routing test until the config is complete enough to route", async () => {
      const actual = await vi.importActual<typeof import("./build_complexity_router_config")>(
        "./build_complexity_router_config",
      );
      vi.mocked(getMissingTiersError).mockImplementation(actual.getMissingTiersError);

      renderWithProviders(<Harness />);

      expect(screen.getByTestId("auto-router-test-routing-btn")).toBeDisabled();
    });

    it("routes a prompt through the config on screen without creating the router", async () => {
      const user = userEvent.setup();
      vi.mocked(getMissingTiersError).mockReturnValue(null);
      vi.mocked(testAutoRouterRouting).mockResolvedValue({
        status: "success",
        result: {
          routed_model: "claude-opus-5",
          routed_model_configured: true,
          routing_decision: { routed_model: "claude-opus-5", tier: "COMPLEX", cause: "literal_keyword_match" },
        },
      });

      renderWithProviders(<Harness />);
      await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
      expandDetailedConfiguration();
      await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
      await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
      const keywordsField = screen.getByText("Keywords 1").closest("div") as HTMLElement;
      await addKeyword(user, keywordsField, "invoice");

      await user.click(screen.getByTestId("auto-router-test-routing-btn"));
      await user.type(await screen.findByTestId("auto-router-routing-test-prompt"), "reconcile this invoice");
      await user.click(screen.getByTestId("auto-router-routing-test-send"));

      await waitFor(() => expect(testAutoRouterRouting).toHaveBeenCalled());
      const [accessToken, request] = vi.mocked(testAutoRouterRouting).mock.calls.at(-1)!;
      expect(accessToken).toBe("token");
      expect(request.prompt).toBe("reconcile this invoice");
      expect(request.router_name).toBe("keyword-router");
      expect(request.complexity_router_config).toMatchObject({
        keyword_tier_rules: [{ keywords: ["invoice"], tier: "COMPLEX" }],
      });
      expect(await screen.findByTestId("auto-router-routing-test-routed-model")).toHaveTextContent("claude-opus-5");
      expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
    });

    it("forgets the last prompt and result when the modal is reopened", async () => {
      const user = userEvent.setup();
      vi.mocked(getMissingTiersError).mockReturnValue(null);
      vi.mocked(testAutoRouterRouting).mockResolvedValue({
        status: "success",
        result: {
          routed_model: "claude-opus-5",
          routed_model_configured: true,
          routing_decision: { routed_model: "claude-opus-5", tier: "COMPLEX", cause: "heuristic_scorer" },
        },
      });

      renderWithProviders(<Harness />);
      await user.click(screen.getByTestId("auto-router-test-routing-btn"));
      await user.type(await screen.findByTestId("auto-router-routing-test-prompt"), "reconcile this invoice");
      await user.click(screen.getByTestId("auto-router-routing-test-send"));
      expect(await screen.findByTestId("auto-router-routing-test-result")).toBeInTheDocument();

      await user.click(screen.getAllByRole("button", { name: /^close$/i }).at(-1)!);
      await user.click(screen.getByTestId("auto-router-test-routing-btn"));

      expect(await screen.findByTestId("auto-router-routing-test-prompt")).toHaveValue("");
      expect(screen.queryByTestId("auto-router-routing-test-result")).not.toBeInTheDocument();
    });
  });

  describe("template presets", () => {
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
      expect(anthropicOption).toHaveTextContent(/Checking model availability/);

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
      expect(anthropicOption).toHaveTextContent(/Cannot verify these models are available/);
    });

    it("keeps group-name presets selectable when only the deployment fetch fails", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
      mockFetchAllModelDeployments.mockRejectedValue(new Error("network error"));

      renderWithProviders(<Harness />);

      await waitForPresetEnabled("Anthropic Family");
      await waitForPresetEnabled("OpenAI Family");
    });

    it("disables a preset missing one of its models, naming the missing model", async () => {
      mockFetchAvailableModels.mockResolvedValue(
        ALL_FAMILY_MODELS.filter((m) => m.model_group !== ANTHROPIC_ONLY_MODEL),
      );

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await waitFor(() => {
        expect(optionByLabel("Anthropic Family")!).toHaveTextContent(new RegExp(`Missing: ${ANTHROPIC_ONLY_MODEL}`));
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

      await selectTemplate("Anthropic Family");

      expect(screen.queryByText("Advanced: Keyword/Semantic Matching")).not.toBeInTheDocument();
      expect(
        screen.getByText(
          `Simple: ${ANTHROPIC_TIERS.SIMPLE.join(", ")} · Medium: ${ANTHROPIC_TIERS.MEDIUM.join(", ")} · ` +
            `Complex: ${ANTHROPIC_TIERS.COMPLEX.join(", ")} · Reasoning: ${ANTHROPIC_TIERS.REASONING.join(", ")}`,
        ),
      ).toBeInTheDocument();
    });

    it("expands detailed configuration when Custom Configuration is chosen", async () => {
      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await selectTemplate("Custom Configuration");

      expect(screen.getByText("Advanced: Keyword/Semantic Matching")).toBeInTheDocument();
    });

    it("lets a caller manually re-expand a detailed configuration a preset just collapsed", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");
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
      await selectTemplate("Anthropic Family");

      await user.type(screen.getByPlaceholderText(/smart_router/i), "anthropic-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        auto_router_default_model: ANTHROPIC_TIERS.MEDIUM[0],
        complexity_router_config: { tiers: ANTHROPIC_TIERS },
      });
    });

    // Every step between the bundled JSON and the payload drops these params silently.
    it("carries a preset's per-tier reasoning effort through to the create payload", async () => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");

      await user.type(screen.getByPlaceholderText(/smart_router/i), "anthropic-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        complexity_router_config: {
          tier_model_configs: {
            REASONING: [{ model_name: "claude-opus-5", litellm_params: { reasoning_effort: "high" } }],
          },
        },
      });
    });

    // Bugbot-found bug: submitBlockedReason disables the button for this, but Form's onFinish
    // (wired to the same handler as the button) fires whenever the form itself is submitted,
    // independent of the button's own disabled state. Without submitRecommendedRouter re-checking
    // it, a real form submission (e.g. Enter, in browsers where that's implicit for this form)
    // could still create a router referencing a model no longer in availableModelSet.
    it("blocks a form submit when a referenced model disappears after the tiers are filled in", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

      const { container } = renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");
      fireEvent.change(screen.getByPlaceholderText(/smart_router/i), { target: { value: "stale-model-router" } });
      expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled();

      // The model list changed after the tiers were filled in (e.g. a deployment removed
      // elsewhere) - update the query cache directly rather than a real refetch, since that's the
      // one thing under test, not how the data arrived. Waiting for the button to actually reflect
      // the disabled state confirms the re-render (and availableModelSet) has settled before the
      // form submits, the same way a real user's next interaction would only happen after that.
      testQueryClient.setQueryData(["availableModels", "autoRouter", "token"], []);
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled());

      fireEvent.submit(container.querySelector("form")!);

      await waitFor(() => expect(toast.fromError).toHaveBeenCalledWith(expect.stringContaining("no longer available")));
      expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
    });
  });

  describe("default model pin", () => {
    const PINNED_MODEL = "pinned-default-model";

    const applyPresetAndPin = async (user: ReturnType<typeof userEvent.setup>) => {
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");

      // Applying a preset collapses Detailed Configuration, so the default model row is behind it.
      expandDetailedConfiguration();
      const defaultModel = screen.getByRole("combobox", { name: "Default model" });
      await user.click(defaultModel);
      await user.type(defaultModel, PINNED_MODEL);
      await user.click(await screen.findByRole("option", { name: PINNED_MODEL }));
    };

    beforeEach(() => {
      mockFetchAvailableModels.mockResolvedValue([...ALL_FAMILY_MODELS, { model_group: PINNED_MODEL, mode: "chat" }]);
    });

    it("submits the pinned model in place of the one the tiers derive", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await applyPresetAndPin(user);
      await user.type(screen.getByPlaceholderText(/smart_router/i), "pinned-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      const submitted = vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0];
      // The pin rides on litellm_params for the backend and is recorded in the config so the edit
      // modal can read it back as a pin rather than guessing from the tiers.
      expect(submitted).toMatchObject({
        auto_router_default_model: PINNED_MODEL,
        complexity_router_config: { tiers: ANTHROPIC_TIERS, default_model: PINNED_MODEL },
      });
      expect(PINNED_MODEL).not.toBe(ANTHROPIC_TIERS.MEDIUM[0]);
    });

    it("blocks a submit whose pinned model is no longer available", async () => {
      const user = userEvent.setup();
      const { container } = renderWithProviders(<Harness />);

      await applyPresetAndPin(user);
      fireEvent.change(screen.getByPlaceholderText(/smart_router/i), { target: { value: "stale-pin-router" } });
      expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled();

      // Only the pinned model disappears - the tier models all survive, so nothing but the pin can
      // be what blocks the submit.
      testQueryClient.setQueryData(["availableModels", "autoRouter", "token"], ALL_FAMILY_MODELS);
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled());

      fireEvent.submit(container.querySelector("form")!);

      await waitFor(() => expect(toast.fromError).toHaveBeenCalledWith(expect.stringContaining(PINNED_MODEL)));
      expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
    });
  });

  describe("plan-mode override", () => {
    beforeEach(() => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    });

    it("omits plan_mode_min_tier from the payload when never touched", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");
      await user.type(screen.getByPlaceholderText(/smart_router/i), "no-plan-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).not.toHaveProperty(
        "plan_mode_min_tier",
      );
    });

    it("carries the enabled override through to the create payload", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");
      expandDetailedConfiguration();
      await user.click(screen.getByText("Advanced: Plan-Mode Override"));
      await user.click(await screen.findByRole("switch", { name: "Route plan-mode requests to a minimum tier" }));

      await user.type(screen.getByPlaceholderText(/smart_router/i), "plan-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
        plan_mode_min_tier: "REASONING",
      });
    });
  });

  describe("deployment-matched presets", () => {
    const renamedDeploymentsFor = (presetKey: string) =>
      [...getRequiredModelsInPreset(getPresetByKey(presetKey)!)].map((model, index) => ({
        model_name: `renamed-${presetKey}-${index}`,
        litellm_params: { model: `someprovider/${model}` },
      }));

    const groupsFor = (deployments: { model_name: string }[]): ModelGroup[] =>
      deployments.map((deployment) => ({ model_group: deployment.model_name, mode: "chat" }));

    const ALL_RENAMED_DEPLOYMENTS = getAllPresets().flatMap((preset) => renamedDeploymentsFor(preset.key));

    const renamedGroupFor = (model: string): string =>
      ALL_RENAMED_DEPLOYMENTS.find((deployment) => deployment.litellm_params.model === `someprovider/${model}`)!
        .model_name;

    it("enables a preset whose models exist only under renamed deployments, labeling the match", async () => {
      mockFetchAvailableModels.mockResolvedValue(groupsFor(ALL_RENAMED_DEPLOYMENTS));
      mockFetchAllModelDeployments.mockResolvedValue(ALL_RENAMED_DEPLOYMENTS);

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
      expect(optionByLabel("Anthropic Family")!).toHaveTextContent(/Matches your deployments/);
    });

    it("keeps detailed configuration open and prefills the admin's group names on apply", async () => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockResolvedValue(groupsFor(ALL_RENAMED_DEPLOYMENTS));
      mockFetchAllModelDeployments.mockResolvedValue(ALL_RENAMED_DEPLOYMENTS);

      renderWithProviders(<Harness />);
      openTemplateDropdown();
      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
      await selectTemplate("Anthropic Family");

      expect(screen.getByText("Advanced: Keyword/Semantic Matching")).toBeInTheDocument();

      await user.type(screen.getByPlaceholderText(/smart_router/i), "renamed-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        complexity_router_config: {
          tiers: {
            SIMPLE: ANTHROPIC_TIERS.SIMPLE.map(renamedGroupFor),
            MEDIUM: ANTHROPIC_TIERS.MEDIUM.map(renamedGroupFor),
            COMPLEX: ANTHROPIC_TIERS.COMPLEX.map(renamedGroupFor),
            REASONING: ANTHROPIC_TIERS.REASONING.map(renamedGroupFor),
          },
        },
      });
    });

    it("lists a deployment-matched preset ahead of one that stays unavailable", async () => {
      const anthropicOnly = renamedDeploymentsFor("anthropic_family");
      mockFetchAvailableModels.mockResolvedValue(groupsFor(anthropicOnly));
      mockFetchAllModelDeployments.mockResolvedValue(anthropicOnly);

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
      const labels = visibleOptions().map((option) => option.querySelector(".font-medium")?.textContent);
      expect(labels).toEqual(["Anthropic Family", "Gemini Family", "Lite", "OpenAI Family", "Custom Configuration"]);
    });

    it.each([
      ["a wildcard group", "openai/*"],
      ["a plain group over a wildcard underlying model", "openai-wild"],
    ])("never lets %s satisfy a preset when the hub lists no expansions", async (_label, modelName) => {
      const wildcard = [{ model_name: modelName, litellm_params: { model: "openai/*" } }];
      mockFetchAvailableModels.mockResolvedValue(groupsFor(wildcard));
      mockFetchAllModelDeployments.mockResolvedValue(wildcard);

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await waitFor(() => {
        expect(optionByLabel("OpenAI Family")!).toHaveTextContent(/Missing:/);
      });
      expect(isOptionDisabled(optionByLabel("OpenAI Family")!)).toBe(true);
    });
  });

  describe("wildcard-matched presets", () => {
    const WILDCARD_DEPLOYMENTS = [{ model_name: "someprovider/*", litellm_params: { model: "someprovider/*" } }];

    const expandedGroupFor = (model: string): string => `someprovider/${model}`;

    const EXPANDED_HUB_GROUPS: ModelGroup[] = [
      { model_group: "someprovider/*", mode: "chat" },
      ...[...new Set(getAllPresets().flatMap((preset) => [...getRequiredModelsInPreset(preset)]))].map((model) => ({
        model_group: expandedGroupFor(model),
        mode: "chat",
      })),
    ];

    it("enables a preset whose models exist only as wildcard-expanded groups, labeling the match", async () => {
      mockFetchAvailableModels.mockResolvedValue(EXPANDED_HUB_GROUPS);
      mockFetchAllModelDeployments.mockResolvedValue(WILDCARD_DEPLOYMENTS);

      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
      expect(optionByLabel("Anthropic Family")!).toHaveTextContent(/Matches your deployments/);
    });

    it("prefills the expanded group names and submits them", async () => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockResolvedValue(EXPANDED_HUB_GROUPS);
      mockFetchAllModelDeployments.mockResolvedValue(WILDCARD_DEPLOYMENTS);

      renderWithProviders(<Harness />);
      openTemplateDropdown();
      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
      await selectTemplate("Anthropic Family");

      await user.type(screen.getByPlaceholderText(/smart_router/i), "wildcard-router");
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        complexity_router_config: {
          tiers: {
            SIMPLE: ANTHROPIC_TIERS.SIMPLE.map(expandedGroupFor),
            MEDIUM: ANTHROPIC_TIERS.MEDIUM.map(expandedGroupFor),
            COMPLEX: ANTHROPIC_TIERS.COMPLEX.map(expandedGroupFor),
            REASONING: ANTHROPIC_TIERS.REASONING.map(expandedGroupFor),
          },
        },
      });
    });
  });
});

describe("getSubmitBlockedReason", () => {
  const tiers = {
    SIMPLE: ["gpt-4o-mini"],
    MEDIUM: ["gpt-4o-mini"],
    COMPLEX: ["gpt-4o-mini"],
    REASONING: ["gpt-4o-mini"],
  };
  const availability = buildModelAvailability(["gpt-4o-mini"], []);
  const referenced = {
    tiers,
    classifierType: "heuristic" as const,
    classifierLlmConfig: undefined,
    semanticMatchingEnabled: false,
    embeddingModel: undefined,
    defaultModel: undefined,
  };

  it("lets a complete heuristic router through", () => {
    expect(getSubmitBlockedReason({ tiers, classifier_type: "heuristic" }, [], referenced, availability)).toBeNull();
  });

  it("blocks an LLM classifier with no model, which the button previously left enabled", () => {
    expect(getSubmitBlockedReason({ tiers, classifier_type: "llm" }, [], referenced, availability)).toContain(
      "Please select a classifier model",
    );
  });

  it("blocks a keyword rule aimed at a tier this router does not have", () => {
    const rules = [{ id: "r1", keywords: ["audit"], tier: "AUDIT" }];
    expect(getSubmitBlockedReason({ tiers, classifier_type: "heuristic" }, rules, referenced, availability)).toContain(
      "no longer has",
    );
  });
});
