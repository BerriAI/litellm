import { renderWithProviders, screen, waitFor, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import NotificationManager from "../molecules/notifications_manager";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { getMissingTiersError } from "./build_complexity_router_config";

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
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
});
