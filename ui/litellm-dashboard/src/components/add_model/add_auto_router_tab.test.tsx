import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
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
    mockFetchAvailableModels.mockResolvedValue([]);
    mockHandleAddAutoRouterSubmit.mockResolvedValue(undefined);
  });

  it("flags every mandatory field when Add Auto Router is clicked with nothing filled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Harness />);

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Auto router name is required")).toBeInTheDocument();
    expect(screen.getAllByText("This tier is required")).toHaveLength(4);
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

    await waitFor(() => expect(isOptionDisabled(optionByLabel("Anthropic Family")!)).toBe(true));
    expect(optionByLabel("Anthropic Family")).toHaveTextContent(/Missing:.*claude-opus-5/);
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

    // No preset was applied, so the tiers are empty and tier validation blocks the submit.
    expect(mockHandleAddAutoRouterSubmit).not.toHaveBeenCalled();
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
});
