import { renderWithProviders, screen, waitFor, testQueryClient, within } from "../../../tests/test-utils";
import { fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import NotificationManager from "../molecules/notifications_manager";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { getMissingTiersError } from "./build_complexity_router_config";
import { ModelGroup } from "@/components/llm_calls/fetch_models";

// Every model referenced by both bundled family presets. A caller holding all of these can
// select either preset; dropping any one greys out the preset that names it.
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

// antd marks a disabled option with a class, not aria-disabled.
const isOptionDisabled = (option: HTMLElement): boolean => option.classList.contains("ant-select-item-option-disabled");

const { mockFetchAvailableModels, mockHandleAddAutoRouterSubmit } = vi.hoisted(() => ({
  mockFetchAvailableModels: vi.fn(),
  mockHandleAddAutoRouterSubmit: vi.fn(),
}));

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: mockFetchAvailableModels,
}));

vi.mock("./handle_add_auto_router_submit", () => ({
  handleAddAutoRouterSubmit: mockHandleAddAutoRouterSubmit,
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: { fromBackend: vi.fn() },
}));

// Kept real by default so the "mandatory field" test still sees genuine tier validation; several
// tests override it with mockReturnValue(null) to reach the submit path without driving four tier
// selects. vi.clearAllMocks() only clears call history, not that override, so it leaks into
// whichever test runs next unless beforeEach restores the real implementation explicitly.
const mocks = vi.hoisted(() => ({
  realGetMissingTiersError: undefined as ((tiers: unknown) => string | null) | undefined,
}));
vi.mock("./build_complexity_router_config", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./build_complexity_router_config")>();
  mocks.realGetMissingTiersError = actual.getMissingTiersError as (tiers: unknown) => string | null;
  return { ...actual, getMissingTiersError: vi.fn(actual.getMissingTiersError) };
});

// Real bundled presets carry no deliberately-falsy fields, so a synthetic preset is appended to
// prove prefill preserves a 0 match threshold and an empty escalation list (a preset switching
// escalation off) instead of overwriting them with the create-form defaults. Defined via
// vi.hoisted so it exists when the hoisted vi.mock factory below references it.
const { FALSY_PRESET } = vi.hoisted(() => ({
  FALSY_PRESET: {
    key: "falsy_family",
    label: "Falsy Family",
    description: "Preset that deliberately disables escalation and pins a zero match threshold",
    complexity_router_config: {
      tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: ["gpt-5-mini"], COMPLEX: ["gpt-5"], REASONING: ["o3"] },
      classifier_type: "heuristic" as const,
      semantic_keyword_matching: true,
      embedding_model: "gpt-5-nano",
      match_threshold: 0,
      keyword_tier_rules: [{ keywords: ["foo"], tier: "SIMPLE" as const }],
      escalation_keywords: [],
      session_affinity: true,
    },
  },
}));

vi.mock("@/lib/autorouter_presets", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/autorouter_presets")>();
  const withSynthetic = [...actual.getAllPresets(), FALSY_PRESET];
  return {
    ...actual,
    getAllPresets: () => withSynthetic,
    getPresetByKey: (key: string) => withSynthetic.find((preset) => preset.key === key),
  };
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
    if (mocks.realGetMissingTiersError) {
      vi.mocked(getMissingTiersError).mockImplementation(mocks.realGetMissingTiersError);
    }
    // testQueryClient is a shared singleton with staleTime: Infinity, so cached model lists would
    // otherwise bleed across tests (a later test reusing accessToken="token" would read an earlier
    // test's data instead of its own mock).
    testQueryClient.clear();
    mockFetchAvailableModels.mockResolvedValue([]);
    mockHandleAddAutoRouterSubmit.mockResolvedValue(undefined);
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

  it("renders template selector as the first control", async () => {
    renderWithProviders(<Harness />);

    const templateSelector = screen.getByTestId("template-selector");
    expect(templateSelector).toBeInTheDocument();

    const templateLabel = screen.getByText("Template");
    const nameLabel = screen.getByText("Auto Router Name");

    expect(templateLabel.compareDocumentPosition(nameLabel)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("enables a preset once every model it references has loaded", async () => {
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

    renderWithProviders(<Harness />);
    openTemplateDropdown();

    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false));
    expect(optionByLabel("Anthropic Family")).not.toHaveTextContent(/Missing:/);
  });

  it("greys out only the preset whose model the caller is missing", async () => {
    // Full OpenAI family, but the Anthropic family is short claude-opus-5.
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS.filter((m) => m.model_group !== "claude-opus-5"));

    renderWithProviders(<Harness />);
    openTemplateDropdown();

    // Wait for the terminal (loaded) state; "disabled" alone is also true mid-load, so assert on
    // the missing-model text that only appears once the list has resolved.
    await waitFor(() => expect(optionByLabel("Anthropic Family")).toHaveTextContent(/Missing:.*claude-opus-5/));
    expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(true);
    // The other family, fully available, stays selectable.
    expect(isOptionDisabled(optionByLabel("OpenAI Family")!)).toBe(false);
  });

  // When the model fetch fails, we have no authoritative data to verify presets' models, so
  // both must be greyed out. This prevents submitting a router with unverifiable models.
  it("greys out all presets when the model fetch fails", async () => {
    mockFetchAvailableModels.mockRejectedValue(new Error("boom"));

    renderWithProviders(<Harness />);
    await waitFor(() => expect(mockFetchAvailableModels).toHaveBeenCalled());
    openTemplateDropdown();

    await waitFor(() => expect(optionByLabel("Anthropic Family")).toBeTruthy());
    expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(true);
    expect(isOptionDisabled(optionByLabel("OpenAI Family")!)).toBe(true);
  });

  // A failed fetch must not strand the caller: Retry re-runs the same query, and a caller who
  // only hit a transient error can recover without closing and reopening the whole modal.
  it("recovers presets once Retry re-fetches successfully", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce(ALL_FAMILY_MODELS);

    renderWithProviders(<Harness />);
    await waitFor(() => expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /retry/i }));

    openTemplateDropdown();
    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false));
    expect(mockFetchAvailableModels).toHaveBeenCalledTimes(2);
  });

  // The headline behavior: selecting a preset must pre-fill the tier config so the created
  // router carries the preset's models. Real tier validation runs here (getMissingTiersError is
  // not stubbed), so if selection stopped pre-filling, the empty tiers would either block the
  // submit or produce a config that fails the tier assertion below.
  it("pre-fills the tier config from the chosen preset and carries it into the create payload", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

    renderWithProviders(<Harness />);
    openTemplateDropdown();

    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false));
    fireEvent.click(optionByLabel("Anthropic Family")!);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "anthropic-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(mockHandleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(mockHandleAddAutoRouterSubmit.mock.calls.at(-1)?.[0]).toMatchObject({
      auto_router_name: "anthropic-router",
      auto_router_default_model: "claude-sonnet-4-5",
      complexity_router_config: {
        tiers: {
          SIMPLE: ["claude-haiku-4-5"],
          MEDIUM: ["claude-sonnet-4-5"],
          COMPLEX: ["claude-opus-5"],
          REASONING: ["claude-opus-5"],
        },
        classifier_type: "heuristic",
      },
    });
  });

  // The load-race both bots flagged: a preset must not be applied before its models are verified.
  // While the model list is still loading the option is disabled, so a click cannot pre-fill the
  // tier config. If it could, a later fetch revealing a missing model would leave a stale, invalid
  // config selected with nothing to clear it, and submit would create an unusable router.
  it("does not apply a preset while the model list is still loading", async () => {
    const user = userEvent.setup();
    let resolveModels: (models: ModelGroup[]) => void = () => undefined;
    mockFetchAvailableModels.mockReturnValue(
      new Promise<ModelGroup[]>((resolve) => {
        resolveModels = resolve;
      }),
    );

    renderWithProviders(<Harness />);
    openTemplateDropdown();

    // Mid-load, the family option is disabled and clicking it must not pre-fill anything.
    await waitFor(() => expect(optionByLabel("Anthropic Family")).toBeTruthy());
    expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(true);
    fireEvent.click(optionByLabel("Anthropic Family")!);

    // Resolve the list WITHOUT claude-opus-5, the exact race: had the click applied, the config
    // would now hold a model the caller lacks. Submit must not carry a preset config through.
    resolveModels(ALL_FAMILY_MODELS.filter((m) => m.model_group !== "claude-opus-5"));
    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(true));

    await user.type(screen.getByPlaceholderText(/smart_router/i), "raced-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    // No preset was applied, so tiers are still empty and submitBlockedReason disables the button.
    expect(mockHandleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });

  // Availability is scoped to the caller. Because the model query is keyed on accessToken, a caller
  // switch re-fetches and re-gates the presets against the NEW caller's models: a preset the first
  // caller could select greys out for a second caller who lacks one of its models. Nothing carries
  // the first caller's list forward, so no preset stays wrongly selectable across the switch.
  it("re-gates presets against the new caller when the access token changes", async () => {
    mockFetchAvailableModels
      .mockResolvedValueOnce(ALL_FAMILY_MODELS)
      .mockResolvedValueOnce(ALL_FAMILY_MODELS.filter((m) => m.model_group !== "o3"));

    const { rerender } = renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="caller-a" userRole="Admin" />,
    );
    openTemplateDropdown();
    await waitFor(() => expect(isOptionDisabled(optionByLabel("OpenAI Family")!)).toBe(false));

    rerender(<AddAutoRouterTab handleOk={vi.fn()} accessToken="caller-b" userRole="Admin" />);

    await waitFor(() => expect(optionByLabel("OpenAI Family")).toHaveTextContent(/Missing:.*o3/));
    expect(isOptionDisabled(optionByLabel("OpenAI Family")!)).toBe(true);
  });

  // A preset only prefills once; nothing re-checks selectedPreset afterward. So the config itself,
  // not the preset it came from, has to stay checked against availableModelSet - here the caller's
  // access narrows after a preset already filled in a model, and submitBlockedReason must catch it
  // the same way it would catch an empty tier, not let a stale reference through.
  it("blocks submit when a model already in the config is no longer in the caller's available list", async () => {
    mockFetchAvailableModels
      .mockResolvedValueOnce(ALL_FAMILY_MODELS)
      .mockResolvedValueOnce(ALL_FAMILY_MODELS.filter((m) => m.model_group !== "claude-opus-5"));

    const { rerender } = renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="caller-a" userRole="Admin" />,
    );
    openTemplateDropdown();
    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false));
    fireEvent.click(optionByLabel("Anthropic Family")!);

    rerender(<AddAutoRouterTab handleOk={vi.fn()} accessToken="caller-b" userRole="Admin" />);

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled());
  });

  // Prefill must preserve a preset's deliberately-falsy fields (a 0 match threshold, an empty
  // escalation list). Using `||` instead of `??` would swap the 0 for the create-form default and
  // re-enable escalation the preset meant to turn off, so this asserts the exact submitted values.
  it("preserves a preset's zero match threshold and empty escalation list through submit", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

    renderWithProviders(<Harness />);
    openTemplateDropdown();

    await waitFor(() => expect(isOptionDisabled(optionByLabel("Falsy Family")!)).toBe(false));
    fireEvent.click(optionByLabel("Falsy Family")!);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "falsy-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(mockHandleAddAutoRouterSubmit).toHaveBeenCalled());
    const payload = mockHandleAddAutoRouterSubmit.mock.calls.at(-1)?.[0];
    expect(payload.complexity_router_config.match_threshold).toBe(0);
    expect(payload.complexity_router_config.escalation_keywords).toEqual([]);
    expect(payload.complexity_router_config.session_affinity).toBe(true);
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

    openTemplateDropdown();
    fireEvent.click(optionByLabel("Custom Configuration")!);
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

    openTemplateDropdown();
    fireEvent.click(optionByLabel("Custom Configuration")!);
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

    openTemplateDropdown();
    fireEvent.click(optionByLabel("Custom Configuration")!);
    await user.type(screen.getByPlaceholderText(/smart_router/i), "team-scoped-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Please select a team to continue")).toBeInTheDocument();
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });

  it("defaults a new router to session affinity off, matching the backend field default", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    openTemplateDropdown();
    fireEvent.click(optionByLabel("Custom Configuration")!);
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

    openTemplateDropdown();
    fireEvent.click(optionByLabel("Custom Configuration")!);
    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    await user.click(screen.getByText("Advanced: Session Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to its first model" }));

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: true,
    });
  });

  // handlePresetChange fully replaces complexityRouterConfig, so a field missing from the object
  // it builds isn't "carried forward" from a prior manual edit, it's silently dropped to whatever
  // the destructuring default resolves to downstream. Turn affinity on via Custom, then apply a
  // preset whose own value is off, and confirm the preset's off wins in both the UI and the
  // payload, the same authoritative-replace behavior every other preset field already has.
  it("applies the preset's own session affinity instead of a previously toggled-on value", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

    renderWithProviders(<Harness />);

    openTemplateDropdown();
    fireEvent.click(optionByLabel("Custom Configuration")!);
    await user.click(screen.getByText("Advanced: Session Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to its first model" }));
    expect(screen.getByRole("switch", { name: "Pin a session to its first model" })).toBeChecked();

    openTemplateDropdown();
    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false));
    fireEvent.click(optionByLabel("Anthropic Family")!);

    expect(screen.getByRole("switch", { name: "Pin a session to its first model" })).not.toBeChecked();

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-then-preset-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(mockHandleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(mockHandleAddAutoRouterSubmit.mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: false,
    });
  });
});
