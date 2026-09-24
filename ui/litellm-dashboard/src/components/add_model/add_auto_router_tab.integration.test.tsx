import {
  openAutoRouterAdvanced,
  selectAutoRouterOption,
  selectAutoRouterApproach,
} from "../../../tests/autoRouterSetup";
import {
  renderWithProviders,
  screen,
  waitFor,
  within,
  fireEvent,
  testQueryClient,
  act,
  chooseSelectOption,
} from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import { toast } from "@/lib/toast";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { getMissingTiersError } from "./build_complexity_router_config";
import { getSubmitBlockedReason } from "./add_auto_router_tab";
import { buildModelAvailability } from "@/lib/autorouter_presets";
import { apiClient, modelCreateCall, testAutoRouterRouting } from "../networking";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { AutoRouterPreset, getRequiredModelsInPreset } from "@/lib/autorouter_presets";
import { BUNDLED_PRESETS, LOADED_PRESETS_QUERY, useAutoRouterPresets } from "../../../tests/mocks/autoRouterPresets";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useAutoRouterPresets",
  async () => await import("../../../tests/mocks/autoRouterPresets"),
);
const getAllPresets = (): AutoRouterPreset[] => BUNDLED_PRESETS;
const getPresetByKey = (key: string): AutoRouterPreset | undefined => BUNDLED_PRESETS.find((p) => p.key === key);

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

const expandDetailedConfiguration = (): void => {
  expect(screen.getByText("Models by tier")).toBeVisible();
};

const visibleOptions = (): HTMLElement[] => screen.queryAllByRole("option");

const optionByLabel = (label: string): HTMLElement | undefined =>
  visibleOptions().find((el) => el.textContent?.startsWith(label));

const isOptionDisabled = (option: HTMLElement): boolean => option.getAttribute("aria-disabled") === "true";

const tierChips = (tier: string): HTMLElement => {
  const placeholder = `Select model(s) for ${tier.toLowerCase()} queries`;
  const chips = screen
    .getAllByRole("toolbar")
    .find((candidate) => within(candidate).queryByLabelText(placeholder) !== null);
  if (!chips) throw new Error(`No tier row found for "${tier}"`);
  return chips;
};

const expectTierModel = (tier: string, model: string): void => {
  const chips = within(tierChips(tier)).getAllByLabelText(/.+/, { selector: '[data-slot="combobox-chip"]' });
  expect(chips.map((chip) => chip.getAttribute("aria-label"))).toEqual([model]);
};

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

const { validateAutoRouterConfig } = vi.hoisted(() => ({
  validateAutoRouterConfig: vi.fn().mockResolvedValue({ valid: true }),
}));

vi.mock("../networking", () => ({
  apiClient: {
    post: vi.fn().mockResolvedValue({
      allowances: [{ key: "heuristic_v2", limit: 1, remaining: 1, available: true }],
      error: null,
    }),
  },
  modelCreateCall: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  testAutoRouterRouting: vi.fn(),
  validateAutoRouterConfig,
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: mockFetchAvailableModels,
  fetchAutoRouterModels: mockFetchAvailableModels,
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
// whether team_id is registered, validated and forwarded, so a plain control stands in. The
// clear button mirrors the real dropdown's x, which emits null rather than a string.
vi.mock("../common_components/team_dropdown", () => ({
  default: ({ value, onChange }: { value?: string; onChange?: (next: string | null) => void }) => (
    <>
      <select
        data-testid="team-dropdown"
        value={value ?? ""}
        onChange={(event) => onChange?.(event.target.value)}
        aria-label="Select Team"
      >
        <option value="">none</option>
        <option value="team-1">team-1</option>
        <option value="team-2">team-2</option>
      </select>
      <button type="button" data-testid="team-dropdown-clear" onClick={() => onChange?.(null)}>
        clear team
      </button>
    </>
  ),
}));

const Harness = () => <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Admin" />;

describe("AddAutoRouterTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.post).mockResolvedValue({
      allowances: [{ key: "heuristic_v2", limit: 1, remaining: 1, available: true }],
      error: null,
    });
    // testQueryClient is a shared singleton with staleTime: Infinity, so cached model lists would
    // otherwise bleed across tests (a later test reusing accessToken="token" would read an earlier
    // test's data instead of its own mock).
    testQueryClient.clear();
    mockFetchAvailableModels.mockResolvedValue([]);
    mockFetchAllModelDeployments.mockResolvedValue([]);
  });

  it.each([1, 0])("defaults to Rule-based with %s v2 slots remaining", async (remaining) => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      allowances: [{ key: "heuristic_v2", limit: 1, remaining, available: true }],
      error: null,
    });
    renderWithProviders(<Harness />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Heuristic" })).toHaveTextContent("Rule-based"));
  });

  it("does not show a tuning rejection for an incomplete Rule-based draft", async () => {
    vi.mocked(apiClient.post).mockImplementationOnce(async (_url, options) => ({
      allowances: [{ key: "heuristic_v2", limit: 1, remaining: 0, available: true }],
      error: (options?.body as { complexity_router_config?: unknown })?.complexity_router_config
        ? "This change needs an available rule-based tuning allowance"
        : null,
    }));
    renderWithProviders(<Harness />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Heuristic" })).toHaveTextContent("Rule-based"));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
  });

  it("keeps the Rule-based default when the form reopens with a cached allowance", async () => {
    const first = renderWithProviders(<Harness />);
    await waitFor(() => expect(apiClient.post).toHaveBeenCalled());
    first.unmount();
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      allowances: [{ key: "heuristic_v2", limit: 1, remaining: 0, available: true }],
      error: null,
    });
    renderWithProviders(<Harness />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Heuristic" })).toHaveTextContent("Rule-based"));
  });

  it.each(["exhausted", "available", "failed"])(
    "keeps an exhausted option disabled through a background refresh that becomes %s",
    async (result) => {
      vi.mocked(apiClient.post).mockResolvedValue({
        allowances: [{ key: "heuristic_v2", limit: 1, remaining: 0, available: true }],
        error: null,
      });
      renderWithProviders(<Harness />);
      fireEvent.click(screen.getByRole("button", { name: "Heuristic" }));
      const option = screen.getByRole("menuitemradio", { name: /^Heuristic v2/ });
      await waitFor(() => expect(option).toHaveTextContent("0 of 1 available"));
      expect(option).toHaveAttribute("aria-disabled", "true");
      let complete: (() => void) | undefined;
      vi.mocked(apiClient.post).mockImplementationOnce(
        () =>
          new Promise((resolve, reject) => {
            complete = () =>
              result === "failed"
                ? reject(new Error("availability unavailable"))
                : resolve({
                    allowances: [
                      { key: "heuristic_v2", limit: 1, remaining: result === "available" ? 1 : 0, available: true },
                    ],
                    error: null,
                  });
          }),
      );
      await act(async () => {
        void testQueryClient.refetchQueries({ queryKey: ["autoRouterAvailability"] });
      });
      await waitFor(() => expect(option).toHaveTextContent("Checking availability"));
      expect(option).toHaveAttribute("aria-disabled", "true");
      fireEvent.click(option);
      expect(screen.getByRole("button", { name: "Heuristic" })).toHaveTextContent("Rule-based");
      await act(async () => complete?.());
      await waitFor(() => expect(option).not.toHaveTextContent("Checking availability"));
      if (result === "available") {
        expect(option).not.toHaveAttribute("aria-disabled", "true");
        fireEvent.click(option);
        expect(screen.getByRole("button", { name: "Heuristic" })).toHaveTextContent("Heuristic v2");
      } else {
        expect(option).toHaveAttribute("aria-disabled", "true");
      }
    },
  );

  it("clears a customization rejection after restoring tiers and keeps the selected models", async () => {
    const user = userEvent.setup();
    const blocked = "Custom tiers or classifier instructions has no available allowance";
    vi.mocked(apiClient.post).mockImplementation(async (_url, options) => {
      const body = options?.body as { complexity_router_config?: { tier_definitions?: unknown } };
      return {
        allowances: [{ key: "tier_or_classifier_prompt", limit: 1, remaining: 0, available: true }],
        error: body?.complexity_router_config?.tier_definitions ? blocked : null,
      };
    });
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    renderWithProviders(<Harness />);
    await user.click(await screen.findByRole("button", { name: "Choose models for me" }));
    await user.click(screen.getByRole("radio", { name: "Jev" }));
    await waitFor(() =>
      expect(apiClient.post).toHaveBeenLastCalledWith(
        "/auto_router/availability",
        expect.objectContaining({
          body: expect.objectContaining({
            complexity_router_config: expect.objectContaining({ classifier_type: "jev" }),
          }),
        }),
      ),
    );
    const initialRequest = vi.mocked(apiClient.post).mock.calls.at(-1)?.[1]?.body as {
      complexity_router_config: { tiers: Record<string, string[]> };
    };
    fireEvent.change(screen.getByLabelText("Auto Router Name"), { target: { value: "restored-router" } });
    await user.click(screen.getByRole("button", { name: "Edit tiers" }));
    await user.click(screen.getByRole("button", { name: "Add tier" }));
    expect(screen.getByLabelText("Name for tier 5")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Restore defaults" }));
    expect(screen.queryByLabelText("Name for tier 5")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Definition for tier 1"), { target: { value: "Only short requests" } });
    expect(await screen.findByRole("alert")).toHaveTextContent(blocked);
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    expect(within(screen.getByRole("alert")).getByRole("link", { name: "Talk to our team" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Restore defaults" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.getByRole("radio", { name: "Jev" })).toBeChecked();
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));
    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    const saved = vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config;
    expect(saved).not.toHaveProperty("tier_definitions");
    expect(saved?.classifier_type).toBe("jev");
    expect(Object.keys(saved?.tiers ?? {})).toEqual(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]);
    expect(saved?.tiers).toEqual(initialRequest.complexity_router_config.tiers);
  });

  it("blocks button and Enter submissions until the edited draft is checked", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    renderWithProviders(<Harness />);
    await user.click(await screen.findByRole("button", { name: "Choose models for me" }));
    await user.click(screen.getByRole("radio", { name: "Jev" }));
    fireEvent.change(screen.getByLabelText("Auto Router Name"), { target: { value: "checked-router" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    let complete: ((result: unknown) => void) | undefined;
    vi.mocked(apiClient.post).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    );
    await user.click(screen.getByRole("button", { name: "Edit tiers" }));
    fireEvent.change(screen.getByLabelText("Definition for tier 1"), { target: { value: "Custom definition" } });
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    fireEvent.submit(screen.getByLabelText("Auto Router Name").closest("form")!);
    await waitFor(() => expect(complete).toBeDefined());
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    await act(async () => complete?.({ allowances: [], error: "Custom tiers have no available allowance" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Custom tiers have no available allowance");
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Restore defaults" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));
    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledOnce());
  });

  it("does not overwrite a classifier chosen while availability is loading", async () => {
    let complete: ((result: unknown) => void) | undefined;
    vi.mocked(apiClient.post).mockReturnValueOnce(
      new Promise((resolve) => {
        complete = resolve;
      }),
    );
    renderWithProviders(<Harness />);
    await userEvent.click(screen.getByRole("radio", { name: "LLM" }));
    complete?.({ allowances: [{ key: "heuristic_v2", limit: 1, remaining: 0, available: true }], error: null });
    await waitFor(() => expect(screen.getByRole("radio", { name: "LLM" })).toBeChecked());
    expect(screen.getByRole("button", { name: "Routing approach" })).toHaveTextContent("Complexity");
  });

  it.each(["LLM", "Jev"])("keeps %s and the frequency when choosing models automatically", async (family) => {
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    renderWithProviders(<Harness />);
    const automatic = await screen.findByRole("button", { name: "Choose models for me" });
    await userEvent.click(screen.getByRole("radio", { name: family }));
    await selectAutoRouterOption("How often to classify", "Every new user message");
    await userEvent.click(automatic);
    expect(screen.getByRole("radio", { name: family })).toBeChecked();
    expect(screen.getByRole("combobox", { name: "How often to classify" })).toHaveTextContent("Every new user message");
    expect(screen.getByRole("button", { name: "Advanced settings" })).toHaveAttribute("aria-expanded", "false");
  });

  it.each(["Capability", "Fuse v2"])(
    "creates %s from its dedicated tab without complexity templates",
    async (label) => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockResolvedValue([
        { model_group: "efficient", mode: "chat" },
        { model_group: "capable", mode: "chat" },
        { model_group: "judge", mode: "chat" },
      ]);
      renderWithProviders(<Harness />);
      await user.type(screen.getByLabelText("Auto Router Name"), "forecast-router");
      await selectAutoRouterApproach(label);
      expect(screen.getByLabelText("Auto Router Name")).toHaveValue("forecast-router");
      expect(screen.queryByTestId("template-selector")).not.toBeInTheDocument();
      expect(screen.queryByTestId("configure-automatically-button")).not.toBeInTheDocument();
      expect(screen.queryByTestId("detailed-configuration-toggle")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Advanced settings" })).toHaveAttribute("aria-expanded", "false");
      expect(screen.queryByText("Classification Method")).not.toBeInTheDocument();
      expect(screen.queryByText("Adaptive Routing")).not.toBeInTheDocument();
      const capability = label === "Capability";
      for (const [role, model] of [
        ["Efficient", "efficient"],
        ["Capable", "capable"],
      ]) {
        await user.click(
          screen.getByRole("combobox", {
            name: capability ? `Select ${role.toLowerCase()} solver models` : `${role} solver`,
          }),
        );
        await user.click(await screen.findByRole("option", { name: model, exact: true }));
        if (capability) await user.keyboard("{Escape}");
      }
      await user.click(screen.getByRole("combobox", { name: "Judge model" }));
      await user.click(await screen.findByRole("option", { name: "judge", exact: true }));
      expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
      if (capability) {
        fireEvent.change(screen.getByLabelText("Solve probability threshold"), { target: { value: "0.7" } });
      } else {
        fireEvent.change(screen.getByLabelText("Efficient solver profile"), { target: { value: "Small solver" } });
        fireEvent.change(screen.getByLabelText("Capable solver profile"), { target: { value: "Large solver" } });
        fireEvent.change(screen.getByLabelText("Harness and budget"), { target: { value: "One attempt" } });
        fireEvent.change(screen.getByLabelText("Maximum quality gap"), { target: { value: "0.05" } });
      }
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
      openAutoRouterAdvanced("Housekeeping Routing");
      openAutoRouterAdvanced("Affinity");
      openAutoRouterAdvanced("Response Format");
      for (const label of ["Adaptive Routing", "Context Window Escalation", "Escalation Keywords"]) {
        expect(screen.queryByText(`${label}`)).not.toBeInTheDocument();
      }
      openAutoRouterAdvanced("Stalled Task Escalation");
      expect(screen.getByText("Stalled Task Escalation")).toBeInTheDocument();
      openAutoRouterAdvanced("Response Format");
      expect(screen.getByText("Response Format")).toBeInTheDocument();
      expect(screen.queryByText("Classification Method")).not.toBeInTheDocument();
      openAutoRouterAdvanced("Affinity");
      expect(screen.getByText("Affinity")).toBeInTheDocument();
      await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: "Add Auto Router" }));
      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledTimes(1));
      const expected = {
        classifier_type: capability ? "capability" : "llm_v2",
        adaptive: false,
        enable_context_window_escalation: false,
        escalation_keywords: [],
        tiers: { SIMPLE: ["efficient"], REASONING: ["capable"] },
        classifier_llm_config: { model: "judge" },
      };
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls[0][0].complexity_router_config).toMatchObject(expected);
    },
  );

  it("restores the automatic/template/detail flow on the Complexity tab", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    renderWithProviders(<Harness />);
    await screen.findByTestId("configure-automatically-button");
    await selectAutoRouterApproach("Capability");
    expect(screen.queryByTestId("configure-automatically-button")).not.toBeInTheDocument();
    await selectAutoRouterApproach("Complexity");
    expect(screen.getByTestId("configure-automatically-button")).toBeInTheDocument();
    expect(screen.getByTestId("template-selector")).toBeInTheDocument();
    expandDetailedConfiguration();
    expect(screen.getByText("Models by tier")).toBeInTheDocument();
    for (const label of ["Adaptive Routing", "Context Window Escalation", "Escalation Keywords"]) {
      openAutoRouterAdvanced(label);
      expect(screen.getAllByText(label)).not.toHaveLength(0);
    }
    openAutoRouterAdvanced("Classification Method");
    expect(screen.queryByRole("radio", { name: /^Capability/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: /^Fuse v2/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Routing approach" })).toHaveTextContent("Complexity");
  });

  it.each(["Capability", "Fuse v2"])(
    "retries failed model loading on %s without losing entered settings",
    async (label) => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockRejectedValueOnce(new Error("Model list unavailable")).mockResolvedValue([
        { model_group: "efficient", mode: "chat" },
        { model_group: "capable", mode: "chat" },
        { model_group: "judge", mode: "chat" },
      ]);
      renderWithProviders(<Harness />);
      await selectAutoRouterApproach(label);
      await user.type(screen.getByLabelText("Auto Router Name"), "forecast-retry");
      const capability = label === "Capability";
      const policyField = capability ? "Solve probability threshold" : "Efficient solver profile";
      fireEvent.change(screen.getByLabelText(policyField), {
        target: { value: capability ? "0.7" : "Small solver" },
      });
      expect(await screen.findByText("Could not load available models.")).toBeVisible();
      expect(screen.queryByTestId("template-selector")).not.toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Retry", exact: true }));
      await waitFor(() => expect(screen.queryByText("Could not load available models.")).not.toBeInTheDocument());
      expect(screen.getByRole("button", { name: "Routing approach" })).toHaveTextContent(label);
      expect(screen.getByLabelText("Auto Router Name")).toHaveValue("forecast-retry");
      expect(screen.getByLabelText(policyField)).toHaveValue(capability ? 0.7 : "Small solver");
      await user.click(screen.getByRole("combobox", { name: "Judge model" }));
      expect(await screen.findByRole("option", { name: "judge", exact: true })).toBeVisible();
      expect(mockFetchAvailableModels).toHaveBeenCalledTimes(2);
    },
  );

  // Detailed Configuration starts collapsed so the modal opens onto just Name + Template; a caller
  // opts into the full tier/classifier form rather than always seeing it up front.
  it("shows models immediately and keeps advanced settings collapsed", async () => {
    renderWithProviders(<Harness />);

    expect(screen.getByRole("button", { name: "Advanced settings" })).toHaveAttribute("aria-expanded", "false");

    expect(screen.getByText("Models by tier")).toBeVisible();

    expect(screen.getByText("Models by tier")).toBeInTheDocument();
  });

  it("hides automatic setup when no available model is recommended", async () => {
    mockFetchAvailableModels.mockResolvedValue([
      { model_group: "unknown-model-a", mode: "chat" },
      { model_group: "unknown-model-b", mode: "chat" },
    ]);
    renderWithProviders(<Harness />);

    openTemplateDropdown();
    await waitFor(() => expect(optionByLabel("Anthropic Family")).toHaveTextContent("Missing:"));
    expect(screen.queryByTestId("configure-automatically-button")).not.toBeInTheDocument();
  });

  it("mixes preferred tier models even when one complete preset is available", async () => {
    const anthropicPreset = getPresetByKey("anthropic_family")!;
    mockFetchAvailableModels.mockResolvedValue(
      [...getRequiredModelsInPreset(anthropicPreset), "gpt-5.6-luna"].map((model_group) => ({
        model_group,
        mode: "chat",
      })),
    );
    mockFetchAllModelDeployments.mockResolvedValue([]);
    renderWithProviders(<Harness />);

    const button = await screen.findByTestId("configure-automatically-button");
    await userEvent.click(button);

    expectTierModel("Simple", "gpt-5.6-luna");
    expectTierModel("Medium", "claude-sonnet-5");
    expectTierModel("Complex", "claude-opus-5-5");
    expectTierModel("Reasoning", "claude-opus-5-5");
    expect(toast.success).not.toHaveBeenCalledWith(expect.stringContaining("Configured with"));
  });

  it("mixes available models from the preferred tier catalog when no complete template fits", async () => {
    mockFetchAvailableModels.mockResolvedValue(
      ["gpt-5.6-luna", "claude-sonnet-5", "gpt-5.6-sol"].map((model_group) => ({
        model_group,
        mode: "chat",
      })),
    );
    mockFetchAllModelDeployments.mockResolvedValue([]);
    renderWithProviders(<Harness />);

    const button = await screen.findByTestId("configure-automatically-button");
    await userEvent.click(button);

    expectTierModel("Simple", "gpt-5.6-luna");
    expectTierModel("Medium", "claude-sonnet-5");
    expectTierModel("Complex", "gpt-5.6-sol");
    expectTierModel("Reasoning", "gpt-5.6-sol");
  });

  it("opens Detailed Configuration on the tiers automatic setup just filled in", async () => {
    const simpleModel = "gpt-5.6-luna";
    mockFetchAvailableModels.mockResolvedValue([...ALL_FAMILY_MODELS, { model_group: simpleModel, mode: "chat" }]);
    renderWithProviders(<Harness />);

    expect(screen.getByRole("button", { name: "Advanced settings" })).toHaveAttribute("aria-expanded", "false");

    await userEvent.click(await screen.findByTestId("configure-automatically-button"));

    expect(screen.getByText("Models by tier")).toBeInTheDocument();
    expectTierModel("Simple", simpleModel);
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

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Auto router name is required")).toBeInTheDocument();
    expect(toast.fromError).toHaveBeenCalledWith("Please enter an Auto Router Name");
  });

  it("offers no team selector to a proxy admin, who may create an unscoped router", async () => {
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
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({ team_id: "team-1" });
  });

  it("creates a member's router with only its name, team and routing configuration", async () => {
    mockFetchAvailableModels.mockImplementation(async (_token: string, teamId?: string) =>
      teamId === "team-1" ? ALL_FAMILY_MODELS : [],
    );
    const actualSubmit = await vi.importActual<typeof import("./handle_add_auto_router_submit")>(
      "./handle_add_auto_router_submit",
    );
    vi.mocked(handleAddAutoRouterSubmit).mockImplementationOnce(actualSubmit.handleAddAutoRouterSubmit);
    const user = userEvent.setup();
    renderWithProviders(
      <AddAutoRouterTab
        handleOk={vi.fn()}
        accessToken="token"
        userRole="Internal User"
        userId="member"
        createScope="team-required"
        teams={
          [
            {
              team_id: "team-1",
              team_member_permissions: ["/auto_router/manage"],
              members_with_roles: [{ user_id: "member", user_email: "member@example.com", role: "user" }],
            },
          ] as import("../networking").Team[]
        }
      />,
    );

    await user.selectOptions(screen.getByTestId("team-dropdown"), "team-1");
    openTemplateDropdown();
    await waitForPresetEnabled(ANTHROPIC_PRESET.label);
    await selectTemplate(ANTHROPIC_PRESET.label);
    fireEvent.change(screen.getByPlaceholderText(/smart_router/i), { target: { value: "my-router" } });
    await user.selectOptions(screen.getByTestId("team-dropdown"), "team-2");
    await waitFor(() => expect(mockFetchAvailableModels).toHaveBeenLastCalledWith("token", "team-2"));
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
    expect(screen.getByPlaceholderText(/smart_router/i)).toHaveValue("my-router");
    await user.selectOptions(screen.getByTestId("team-dropdown"), "team-1");
    expandDetailedConfiguration();
    expect(screen.queryByText("Compression")).not.toBeInTheDocument();
    expect(screen.queryByText("Model Access Groups")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(modelCreateCall).toHaveBeenCalled());
    expect(modelCreateCall).toHaveBeenLastCalledWith("token", {
      model_name: "my-router",
      model_info: { team_id: "team-1" },
      litellm_params: {
        model: "auto_router/complexity_router",
        complexity_router_config: expect.objectContaining({ tiers: ANTHROPIC_TIERS }),
        complexity_router_default_model: expect.any(String),
      },
    });
    expect(mockFetchAvailableModels).toHaveBeenCalledWith("token", "team-1");
  });

  it("does not submit when the backend's dry-run rejects the config", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);
    validateAutoRouterConfig.mockResolvedValueOnce({
      valid: false,
      error: "session_affinity cannot be combined with tier_definitions",
    });

    renderWithProviders(<Harness />);
    await user.type(screen.getByPlaceholderText(/smart_router/i), "rejected-router");
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(validateAutoRouterConfig).toHaveBeenCalled());
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });

  it("creates the router once when the form is submitted again mid dry-run", async () => {
    vi.mocked(getMissingTiersError).mockReturnValue(null);
    let resolveVerdict: (verdict: { valid: boolean }) => void = () => {};
    validateAutoRouterConfig.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveVerdict = resolve;
        }),
    );

    const { container } = renderWithProviders(<Harness />);
    fireEvent.change(screen.getByPlaceholderText(/smart_router/i), { target: { value: "double-submit-router" } });

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    fireEvent.submit(container.querySelector("form")!);
    await waitFor(() => expect(validateAutoRouterConfig).toHaveBeenCalledOnce());
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
    fireEvent.submit(container.querySelector("form")!);

    resolveVerdict({ valid: true });
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    expect(validateAutoRouterConfig).toHaveBeenCalledTimes(1);
    expect(handleAddAutoRouterSubmit).toHaveBeenCalledTimes(1);
  });

  it("submits when the dry-run passes, so the gate is not simply blocking everything", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);
    validateAutoRouterConfig.mockResolvedValueOnce({ valid: true });

    renderWithProviders(<Harness />);
    await user.type(screen.getByPlaceholderText(/smart_router/i), "accepted-router");
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
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
    openAutoRouterAdvanced("Keyword/Semantic Matching");
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
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();

    await addKeyword(user, screen.getByText("Keywords 1").closest("div") as HTMLElement, "invoice");

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    expect(screen.queryByText("At least one keyword is required")).not.toBeInTheDocument();
  });

  it("shows the orphaned-rule reason in the tier editor when a rule's tier is removed", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "orphan-rule-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    await addKeyword(user, screen.getByText("Keywords 1").closest("div") as HTMLElement, "invoice");

    await user.click(screen.getByRole("button", { name: "Edit tiers" }));
    await user.click(screen.getByRole("button", { name: "Remove the COMPLEX tier" }));

    expect(await screen.findByText(/route to a tier this router no longer has/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add auto router/i })).toBeDisabled();
  });

  it("marks only the offending keyword row, leaving a filled one alone", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "keyword-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Keyword/Semantic Matching");
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
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    const keywordsField = screen.getByText("Keywords 1").closest("div") as HTMLElement;
    await addKeyword(user, keywordsField, "invoice");
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Please select a team to continue")).toBeInTheDocument();
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });

  // The shared dropdown emits null on clear while this form's schema wants a string, so the
  // form maps null back to "": the user sees the pick-a-team message, not a zod type error.
  it("treats a team picked and then cleared like no team at all", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    await user.type(screen.getByPlaceholderText(/smart_router/i), "team-scoped-router");
    await user.selectOptions(screen.getByTestId("team-dropdown"), "team-1");
    await user.click(screen.getByTestId("team-dropdown-clear"));
    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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
    openAutoRouterAdvanced("Classification Method");
    expect(await screen.findByRole("combobox", { name: "How often to classify" })).not.toHaveTextContent(
      "Once per session",
    );

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: false,
    });
  });

  it("blocks invalid success thresholds and creates a heuristic v2 router with explicit zero", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);
    renderWithProviders(<Harness />);
    fireEvent.change(screen.getByLabelText("Auto Router Name"), { target: { value: "threshold-router" } });
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("Heuristic", "Heuristic v2");

    const threshold = screen.getByRole("textbox", { name: "Success threshold" });
    expect(threshold).toHaveValue("");
    fireEvent.change(threshold, { target: { value: "invalid" } });
    fireEvent.blur(threshold);
    expect(threshold).toHaveValue("invalid");
    expect(threshold).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    expect(screen.getByTestId("auto-router-test-routing-btn")).toBeDisabled();

    await selectAutoRouterOption("Heuristic", "Rule-based");
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    await selectAutoRouterOption("Heuristic", "Heuristic v2");
    fireEvent.change(screen.getByRole("textbox", { name: "Success threshold" }), { target: { value: "1.01" } });
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Success threshold" }), { target: { value: "0" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledOnce());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      classifier_type: "heuristic_v2",
      heuristic_v2_success_threshold: 0,
    });
  });

  it("preserves classifier tuning when choosing models automatically", async () => {
    const user = userEvent.setup();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    renderWithProviders(<Harness />);
    const automaticSetup = await screen.findByRole("button", { name: "Choose models for me" });
    await waitFor(() => expect(automaticSetup).toBeEnabled());
    await user.click(automaticSetup);
    fireEvent.change(screen.getByLabelText("Auto Router Name"), { target: { value: "reset-threshold-router" } });
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("Heuristic", "Heuristic v2");
    fireEvent.change(screen.getByRole("textbox", { name: "Success threshold" }), { target: { value: "1.1" } });
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();

    await user.click(automaticSetup);
    expect(screen.getByRole("textbox", { name: "Success threshold" })).toHaveValue("1.1");
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Success threshold" }), { target: { value: "" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));
    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledOnce());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).not.toHaveProperty(
      "heuristic_v2_success_threshold",
    );
  });

  it("clears an invalid inactive threshold before creating the router", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);
    renderWithProviders(<Harness />);
    fireEvent.change(screen.getByLabelText("Auto Router Name"), { target: { value: "clear-threshold-router" } });
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("Heuristic", "Heuristic v2");
    fireEvent.change(screen.getByRole("textbox", { name: "Success threshold" }), { target: { value: "invalid" } });
    await selectAutoRouterOption("Heuristic", "Rule-based");
    expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Clear Heuristic v2 threshold" }));
    expect(screen.queryByRole("region", { name: "Inactive Heuristic v2 threshold" })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));
    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledOnce());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).not.toHaveProperty(
      "heuristic_v2_success_threshold",
    );
  });

  it("starts context-window escalation disabled and carries an explicit opt-in to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "ctx-window-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Context Window Escalation");
    const toggle = await screen.findByRole("switch", { name: "Escalate oversized prompts to a tier that fits" });
    expect(toggle).not.toBeChecked();
    expect(screen.queryByLabelText("Window fit buffer")).not.toBeInTheDocument();
    await user.click(toggle);

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      enable_context_window_escalation: true,
    });
  });

  it("clamps the context-window buffer to 1 and keeps an untouched buffer out of the payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "ctx-buffer-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Context Window Escalation");
    await user.click(screen.getByRole("switch", { name: "Escalate oversized prompts to a tier that fits" }));
    const buffer = await screen.findByLabelText("Window fit buffer");
    fireEvent.change(buffer, { target: { value: "1.5" } });
    fireEvent.blur(buffer, { target: { value: "1.5" } });

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    const config = vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config;
    expect(config).toMatchObject({ context_window_escalation_buffer: 1 });
    expect(config).toHaveProperty("enable_context_window_escalation", true);
  });

  it("clearing the buffer removes it from the payload so the router tracks the backend default", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "ctx-clear-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Context Window Escalation");
    await user.click(screen.getByRole("switch", { name: "Escalate oversized prompts to a tier that fits" }));
    const buffer = await screen.findByLabelText("Window fit buffer");
    fireEvent.change(buffer, { target: { value: "0.8" } });
    fireEvent.blur(buffer, { target: { value: "0.8" } });
    fireEvent.change(buffer, { target: { value: "" } });
    fireEvent.blur(buffer, { target: { value: "" } });

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).not.toHaveProperty(
      "context_window_escalation_buffer",
    );
  });

  describe("prompt compression", () => {
    it("leaves both compression keys out of the create payload when the section is untouched", async () => {
      const user = userEvent.setup();
      vi.mocked(getMissingTiersError).mockReturnValue(null);

      renderWithProviders(<Harness />);

      await user.type(screen.getByPlaceholderText(/smart_router/i), "no-compression-router");
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      const submitted = vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0];
      expect(submitted).not.toHaveProperty("auto_router_routing_compression");
      expect(submitted).not.toHaveProperty("auto_router_model_compression");
    });

    it("mirrors an explicit no-compression routing choice onto the model call by default", async () => {
      const user = userEvent.setup();
      vi.mocked(getMissingTiersError).mockReturnValue(null);

      renderWithProviders(<Harness />);

      await user.type(screen.getByPlaceholderText(/smart_router/i), "no-compression-explicit-router");
      expandDetailedConfiguration();
      openAutoRouterAdvanced("Compression");
      await chooseSelectOption(
        user,
        screen.getByRole("combobox", { name: "Routing decision compression" }),
        "None (no compression)",
      );

      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      const submitted = vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0];
      expect(submitted?.auto_router_routing_compression).toBe("none");
      expect(submitted?.auto_router_model_compression).toBe("none");
    });

    it("defaults the model call to none when different is chosen but nothing is picked there", async () => {
      const user = userEvent.setup();
      vi.mocked(getMissingTiersError).mockReturnValue(null);

      renderWithProviders(<Harness />);

      await user.type(screen.getByPlaceholderText(/smart_router/i), "different-compression-router");
      expandDetailedConfiguration();
      openAutoRouterAdvanced("Compression");
      await chooseSelectOption(
        user,
        screen.getByRole("combobox", { name: "Routing decision compression" }),
        "None (no compression)",
      );
      await user.click(screen.getByText("Use a different compression"));
      expect(screen.getByRole("combobox", { name: "Model call compression" })).toBeInTheDocument();

      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      const submitted = vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0];
      expect(submitted?.auto_router_routing_compression).toBe("none");
      expect(submitted?.auto_router_model_compression).toBe("none");
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
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("Heuristic", "Rule-based");
    await user.click(await screen.findByText("Advanced scoring"));
    fireEvent.change(await screen.findByLabelText("Minimum score"), { target: { value: "0" } });

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      reasoning_override_min_score: 0,
    });
  });

  it("carries session affinity turned on and its idle window through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("How often to classify", "Once per session");
    openAutoRouterAdvanced("Affinity");
    const ttl = await screen.findByLabelText("How long a pin survives idle (seconds)");
    fireEvent.change(ttl, { target: { value: "300" } });
    fireEvent.blur(ttl);

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      session_affinity: true,
      session_affinity_ttl_seconds: 300,
    });
  });

  it("carries every new user message through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "user-turn-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("How often to classify", "Every new user message");

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      classification_mode: "user_turn",
    });
  });

  it("writes every_request into the create payload when the default frequency stays selected", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "default-timing-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Classification Method");
    expect(await screen.findByRole("combobox", { name: "How often to classify" })).toHaveTextContent("Every request");

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(
      vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config.classification_mode,
    ).toBe("every_request");
  });

  it("defaults a new router to deployment affinity on, matching the backend field default", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    await user.type(screen.getByPlaceholderText(/smart_router/i), "affinity-router");
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Affinity");
    expect(await screen.findByRole("switch", { name: "Pin one model deployment per tier" })).toBeChecked();

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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
    openAutoRouterAdvanced("Affinity");
    await user.click(await screen.findByRole("switch", { name: "Pin one model deployment per tier" }));

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      deployment_affinity: false,
    });
  });

  it("writes both modality flags as false into the create payload when the panel stays untouched", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    fireEvent.change(screen.getByPlaceholderText(/smart_router/i), { target: { value: "modality-router" } });

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      modality_routing: false,
      modality_pin_override: false,
    });
  });

  it("carries the pin override through to the create payload once image routing unlocks it", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(<Harness />);

    fireEvent.change(screen.getByPlaceholderText(/smart_router/i), { target: { value: "modality-router" } });
    expandDetailedConfiguration();
    openAutoRouterAdvanced("Modality Routing");
    await user.click(await screen.findByRole("switch", { name: "Route image requests to vision-capable models" }));
    await user.click(await screen.findByRole("switch", { name: "Override session pin for image requests" }));

    await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0].complexity_router_config).toMatchObject({
      modality_routing: true,
      modality_pin_override: true,
    });
  });

  // Custom is the escape hatch, not the headline choice, so it's listed after every bundled preset
  // rather than first.
  it("lists Custom Configuration after the bundled presets", async () => {
    renderWithProviders(<Harness />);
    openTemplateDropdown();

    const labels = visibleOptions().map((option) => option.querySelector(".font-medium")?.textContent);

    expect(labels).toEqual([
      "1M Context",
      "Anthropic Family",
      "Gemini Family",
      "Lite",
      "OpenAI Family",
      "Custom Configuration",
    ]);
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
      openAutoRouterAdvanced("Keyword/Semantic Matching");
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

    it("shows the template's model choices while advanced settings stay collapsed", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");
      expect(screen.getByRole("button", { name: "Advanced settings" })).toHaveAttribute("aria-expanded", "false");
      expectTierModel("simple", ANTHROPIC_TIERS.SIMPLE[0]);
      expectTierModel("reasoning", ANTHROPIC_TIERS.REASONING[0]);
    });

    it("expands detailed configuration when Custom Configuration is chosen", async () => {
      renderWithProviders(<Harness />);
      openTemplateDropdown();

      await selectTemplate("Custom Configuration");

      openAutoRouterAdvanced("Keyword/Semantic Matching");

      expect(screen.getByText("Keyword/Semantic Matching")).toBeInTheDocument();
    });

    it("lets a caller manually re-expand a detailed configuration a preset just collapsed", async () => {
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");
      expect(screen.queryByText("Keyword/Semantic Matching")).not.toBeInTheDocument();

      expect(screen.getByText("Models by tier")).toBeVisible();

      openAutoRouterAdvanced("Keyword/Semantic Matching");

      expect(screen.getByText("Keyword/Semantic Matching")).toBeInTheDocument();
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
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        auto_router_default_model: ANTHROPIC_TIERS.MEDIUM[0],
        complexity_router_config: { tiers: ANTHROPIC_TIERS },
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
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());

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

    it("carries a preset's per-tier reasoning effort through to the create payload", async () => {
      const user = userEvent.setup();
      mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);

      renderWithProviders(<Harness />);
      await waitForPresetEnabled("Anthropic Family");
      await selectTemplate("Anthropic Family");

      await user.type(screen.getByPlaceholderText(/smart_router/i), "anthropic-router");
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        complexity_router_config: {
          tier_model_configs: ANTHROPIC_PRESET.complexity_router_config.tier_model_configs,
        },
      });
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
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());

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
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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
      openAutoRouterAdvanced("Plan-Mode Override");
      await user.click(await screen.findByRole("switch", { name: "Route plan-mode requests to a minimum tier" }));

      await user.type(screen.getByPlaceholderText(/smart_router/i), "plan-router");
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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

    it("keeps detailed configuration open and submits native group names when cloud twins are available", async () => {
      const user = userEvent.setup();
      const nativeDeployments = renamedDeploymentsFor("anthropic_family").map((deployment) => ({
        ...deployment,
        litellm_params: { model: deployment.litellm_params.model.replace("someprovider/", "anthropic/") },
      }));
      const nativeGroupFor = (model: string): string =>
        nativeDeployments.find((deployment) => deployment.litellm_params.model === `anthropic/${model}`)!.model_name;
      const cloudDeployments = nativeDeployments.map((deployment) => ({
        model_name: `a-cloud-${deployment.model_name}`,
        litellm_params: {
          model: `bedrock/us.anthropic.${deployment.litellm_params.model.split("/")[1]}-v1:0`,
        },
      }));
      const deployments = [...cloudDeployments, ...nativeDeployments];
      mockFetchAvailableModels.mockResolvedValue(groupsFor(deployments));
      mockFetchAllModelDeployments.mockResolvedValue(deployments);

      renderWithProviders(<Harness />);
      openTemplateDropdown();
      await waitFor(() => {
        expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(false);
      });
      await selectTemplate("Anthropic Family");
      expectTierModel("Complex", nativeGroupFor(ANTHROPIC_TIERS.COMPLEX[0]));

      openAutoRouterAdvanced("Keyword/Semantic Matching");

      expect(screen.getByText("Keyword/Semantic Matching")).toBeInTheDocument();

      await user.type(screen.getByPlaceholderText(/smart_router/i), "renamed-router");
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
      await user.click(screen.getByRole("button", { name: /add auto router/i }));

      await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
      expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({
        complexity_router_config: {
          tiers: {
            SIMPLE: ANTHROPIC_TIERS.SIMPLE.map(nativeGroupFor),
            MEDIUM: ANTHROPIC_TIERS.MEDIUM.map(nativeGroupFor),
            COMPLEX: ANTHROPIC_TIERS.COMPLEX.map(nativeGroupFor),
            REASONING: ANTHROPIC_TIERS.REASONING.map(nativeGroupFor),
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
      expect(labels).toEqual([
        "Anthropic Family",
        "1M Context",
        "Gemini Family",
        "Lite",
        "OpenAI Family",
        "Custom Configuration",
      ]);
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
      await waitFor(() => expect(screen.getByRole("button", { name: /add auto router/i })).toBeEnabled());
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

  it("lets a complete heuristic router through", async () => {
    expect(getSubmitBlockedReason({ tiers, classifier_type: "heuristic" }, [], referenced, availability)).toBeNull();
  });

  it("blocks an LLM classifier with no model, which the button previously left enabled", async () => {
    expect(getSubmitBlockedReason({ tiers, classifier_type: "llm" }, [], referenced, availability)).toContain(
      "Please select a classifier model",
    );
  });

  it("blocks an edited tier set with no classifier model, since the set forces the LLM classifier", async () => {
    const config = {
      tiers,
      classifier_type: "heuristic" as const,
      custom_tier_set: {
        tiers: [
          { id: "a", name: "CASUAL", definition: "d", models: ["gpt-4o-mini"] },
          { id: "b", name: "AUDIT", definition: "d", models: ["gpt-4o-mini"] },
        ],
        fallback_tier_id: "a",
      },
    };
    expect(getSubmitBlockedReason(config, [], referenced, availability)).toContain(
      "an edited tier set routes with the LLM classifier",
    );
  });

  it("blocks a keyword rule aimed at a tier this router does not have", async () => {
    const rules = [{ id: "r1", keywords: ["audit"], tier: "AUDIT" }];
    expect(getSubmitBlockedReason({ tiers, classifier_type: "heuristic" }, rules, referenced, availability)).toContain(
      "no longer has",
    );
  });
});

describe("preset catalog fetch states", () => {
  afterEach(() => vi.mocked(useAutoRouterPresets).mockReturnValue(LOADED_PRESETS_QUERY));

  it("preserves a JEV preset's per-turn bound in the create request", async () => {
    vi.clearAllMocks();
    testQueryClient.clear();
    vi.mocked(handleAddAutoRouterSubmit).mockReset();
    mockFetchAvailableModels.mockResolvedValue(ALL_FAMILY_MODELS);
    vi.mocked(useAutoRouterPresets).mockReturnValue({
      ...LOADED_PRESETS_QUERY,
      data: [
        {
          ...ANTHROPIC_PRESET,
          key: "bounded_jev",
          label: "Bounded JEV",
          complexity_router_config: {
            ...ANTHROPIC_PRESET.complexity_router_config,
            classifier_type: "jev",
            jev_classifier_config: { model: "jev-test", timeout_ms: 3000 },
            classifier_context_per_turn_chars: 450,
          },
        },
      ],
    });
    renderWithProviders(<Harness />);
    await waitForPresetEnabled("Bounded JEV");
    await selectTemplate("Bounded JEV");
    fireEvent.change(screen.getByLabelText("Auto Router Name"), { target: { value: "bounded-router" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Add Auto Router" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Add Auto Router" }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalledOnce());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls[0][0].complexity_router_config).toMatchObject({
      classifier_type: "jev",
      classifier_context_per_turn_chars: 450,
    });
  });

  it("keeps showing cached presets without the error banner when only a refetch fails", async () => {
    vi.mocked(useAutoRouterPresets).mockReturnValue({
      ...LOADED_PRESETS_QUERY,
      isError: true,
    } as never);
    renderWithProviders(<Harness />);

    expect(screen.queryByText(/Could not load templates/)).not.toBeInTheDocument();

    openTemplateDropdown();
    expect(screen.queryAllByRole("option").length).toBeGreaterThan(1);
  });

  it("shows a loading hint while the catalog fetch is pending", async () => {
    vi.mocked(useAutoRouterPresets).mockReturnValue({
      ...LOADED_PRESETS_QUERY,
      data: undefined,
      isPending: true,
    } as never);
    renderWithProviders(<Harness />);

    expect(screen.getByText("Loading templates...")).toBeInTheDocument();
  });

  it("degrades to Custom Configuration with a retry hint that refetches the catalog", async () => {
    const refetch = vi.fn();
    vi.mocked(useAutoRouterPresets).mockReturnValue({
      ...LOADED_PRESETS_QUERY,
      data: undefined,
      isError: true,
      refetch,
    } as never);
    renderWithProviders(<Harness />);

    expect(await screen.findByText(/Could not load templates/)).toBeInTheDocument();

    openTemplateDropdown();
    const options = screen.queryAllByRole("option");
    expect(options).toHaveLength(1);
    expect(options[0]).toHaveTextContent("Custom Configuration");

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(refetch).toHaveBeenCalled();
  });
});
